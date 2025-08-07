/**
 * Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
 * Serow is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, version 3.
 *
 * Serow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Serow. If not,
 * see <https://www.gnu.org/licenses/>.
 **/
/**
 * @brief IMU Orientation Estimation with the Mahony Filter
 * @author Stylianos Piperakis
 * @details estimates the IMU frame orientation with respect to a world frame of reference with IMU
 * measurements
 * @note updateIMU() is based on the https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
 * repository
 */

#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <iostream>
#include <optional>

namespace serow {

class Mahony {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** @fn  Mahony(double freq, double kp, double ki = 0.0)
     *  @brief Initializes parameters and sets the sampling frequency and gains of the algorithm
     *  @param freq sampling frequency
     *  @param kp Proportional gain
     *  @param ki Integral gain
     */
    Mahony(double freq, double kp, double ki = 0.0) {
        nominal_dt_ = 1.0 / freq;
        twoKp_ = 2.0 * kp;
        twoKi_ = 2.0 * ki;
        integralFBx_ = 0.0;
        integralFBy_ = 0.0;
        integralFBz_ = 0.0;
    }

    /** @fn Eigen::Quaterniond getQ()
     *  @returns the orientation of IMU w.r.t the world frame as a quaternion
     */
    Eigen::Quaterniond getQ() const {
        return q_;
    }

    /** @fn  Eigen::Vector3d getAcc()
     *  @returns the linear acceleration of IMU in the world frame
     */
    Eigen::Vector3d getAcc() const {
        return acc_;
    }

    /** @fn  Eigen::Vector3d getGyro()
     *  @returns the angular velocity of IMU in the world frame
     */
    Eigen::Vector3d getGyro() const {
        return gyro_;
    }

    /** @fn Eigen::Matrix3d getR()
     *  @returns the orientation of IMU w.r.t the world frame as a rotation matrix
     */
    Eigen::Matrix3d getR() const {
        return R_;
    }

    /** @fn Eigen::Vector3d getEuler()
     *  @returns the orientation of IMU w.r.t the world frame as  euler angles in the RPY
     * convention
     */
    Eigen::Vector3d getEuler() const {
        return q_.toRotationMatrix().eulerAngles(0, 1, 2);
    }

    /** @fn filter(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc)
     *  @brief Computes the IMU orientation w.r.t the world frame of reference
     *  @param gyro angular velocity as measured by the IMU
     *  @param acc linea acceleration as measured by the IMU
     *  @param timestamp timestamp of the measurement
     */
    void filter(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double timestamp) {
        double dt = nominal_dt_;
        if (timestamp_) {
            dt = timestamp - timestamp_.value();

            if (dt < nominal_dt_ / 2.0) {
                std::cout << "[SEROW/Mahony]: Sample time is abnormal " << dt
                          << " while the nominal sample time is " << nominal_dt_
                          << " setting to nominal" << std::endl;
                dt = nominal_dt_;
            }
            timestamp_ = timestamp;
        } else {
            timestamp_ = timestamp;
        }

        double recipNorm;
        double halfvx, halfvy, halfvz;
        double halfex, halfey, halfez;
        double qa, qb, qc;
        double ax, ay, az, gx, gy, gz;

        gx = gyro(0);
        gy = gyro(1);
        gz = gyro(2);
        ax = acc(0);
        ay = acc(1);
        az = acc(2);
        // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer
        // normalization)
        if (!(std::abs(ax) < 1e-6 && std::abs(ay) < 1e-6 && std::abs(az) < 1e-6)) {
            // Normalize accelerometer measurement
            recipNorm = 1.0 / std::sqrt(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // Estimated direction of gravity and vector perpendicular to magnetic flux
            halfvx = q1_ * q3_ - q0_ * q2_;
            halfvy = q0_ * q1_ + q2_ * q3_;
            halfvz = q0_ * q0_ - 0.5f + q3_ * q3_;

            // Error is sum of cross product between estimated and measured direction of gravity
            halfex = (ay * halfvz - az * halfvy);
            halfey = (az * halfvx - ax * halfvz);
            halfez = (ax * halfvy - ay * halfvx);

            // Compute and apply integral feedback if enabled
            if (twoKi_ > 0.0f) {
                // integral error scaled by Ki
                integralFBx_ += twoKi_ * halfex * dt;
                integralFBy_ += twoKi_ * halfey * dt;
                integralFBz_ += twoKi_ * halfez * dt;
                // apply integral feedback
                gx += integralFBx_;
                gy += integralFBy_;
                gz += integralFBz_;
            } else {
                // prevent integral windup
                integralFBx_ = 0.0f;
                integralFBy_ = 0.0f;
                integralFBz_ = 0.0f;
            }

            // Apply proportional feedback
            gx += twoKp_ * halfex;
            gy += twoKp_ * halfey;
            gz += twoKp_ * halfez;
        }

        // Integrate rate of change of quaternion
        gx *= (0.5f * dt);  // pre-multiply common factors
        gy *= (0.5f * dt);
        gz *= (0.5f * dt);
        qa = q0_;
        qb = q1_;
        qc = q2_;
        q0_ += (-qb * gx - qc * gy - q3_ * gz);
        q1_ += (qa * gx + qc * gz - q3_ * gy);
        q2_ += (qa * gy - qb * gz + q3_ * gx);
        q3_ += (qa * gz + qb * gy - qc * gx);

        // Normalize quaternion
        recipNorm = 1.0 / sqrt(q0_ * q0_ + q1_ * q1_ + q2_ * q2_ + q3_ * q3_);
        q0_ *= recipNorm;
        q1_ *= recipNorm;
        q2_ *= recipNorm;
        q3_ *= recipNorm;
        q_.x() = q1_;
        q_.y() = q2_;
        q_.z() = q3_;
        q_.w() = q0_;
        R_ = q_.toRotationMatrix();
        acc_ = R_ * acc;
        gyro_ = R_ * gyro;
    }

    /// Set the state of the Mahony filter
    void setState(const Eigen::Quaterniond& q) {
        q_ = q;
        R_ = q_.toRotationMatrix();
    }

private:
    /// IMU orientation w.r.t the world frame as a quaternion
    Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
    /// IMU orientation w.r.t the world frame as a rotation matrix
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
    /// IMU linear acceleration and angular velocity in the world frame
    Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_ = Eigen::Vector3d::Zero();
    /// Algorithm propotional gain
    double twoKp_ = 0;
    /// Algorithm Integral gain
    double twoKi_ = 0;
    /// Quaternion of sensor frame relative to auxiliary frame
    double q0_ = 1, q1_ = 0, q2_ = 0, q3_ = 0;
    /// Integral of angular velocity in x,y,z
    double integralFBx_ = 0, integralFBy_ = 0, integralFBz_ = 0;
    /// Nominal sample time
    double nominal_dt_ = 0;
    /// Timestamp of the last measurement
    std::optional<double> timestamp_ = std::nullopt;
};

}  // namespace serow
