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
     *  @param verbose whether to print verbose output
     */
    Mahony(double freq, double kp, double ki = 0.0, bool verbose = false) {
        nominal_dt_ = 1.0 / freq;
        twoKp_ = 2.0 * kp;
        twoKi_ = 2.0 * ki;
        integralFBx_ = 0.0;
        integralFBy_ = 0.0;
        integralFBz_ = 0.0;
        verbose_ = verbose;
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
     *  @param acc linear acceleration as measured by the IMU
     *  @param timestamp timestamp of the measurement
     */
    void filter(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc, double timestamp) {
        double dt = nominal_dt_;
        double gx = gyro(0);
        double gy = gyro(1);
        double gz = gyro(2);
        double ax = acc(0);
        double ay = acc(1);
        double az = acc(2);

        // Valid accelerometer check
        if (!(std::abs(ax) < 1e-6 && std::abs(ay) < 1e-6 && std::abs(az) < 1e-6)) {
            // Normalize accelerometer
            const double recipNorm = 1.0 / std::sqrt(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // Estimated gravity direction
            const double halfvx = q1_ * q3_ - q0_ * q2_;
            const double halfvy = q0_ * q1_ + q2_ * q3_;
            const double halfvz = q0_ * q0_ - 0.5 + q3_ * q3_;

            // Error between measured and estimated gravity
            const double halfex = ay * halfvz - az * halfvy;
            const double halfey = az * halfvx - ax * halfvz;
            const double halfez = ax * halfvy - ay * halfvx;

            // Integral feedback with anti-windup decay
            if (twoKi_ > 0.0) {
                integralFBx_ = 0.98 * integralFBx_ + twoKi_ * halfex * dt;
                integralFBy_ = 0.98 * integralFBy_ + twoKi_ * halfey * dt;
                integralFBz_ = 0.98 * integralFBz_ + twoKi_ * halfez * dt;
                gx += integralFBx_;
                gy += integralFBy_;
                gz += integralFBz_;
            } else {
                integralFBx_ = 0.0;
                integralFBy_ = 0.0;
                integralFBz_ = 0.0;
            }

            // Proportional feedback
            gx += twoKp_ * halfex;
            gy += twoKp_ * halfey;
            gz += twoKp_ * halfez;
        }

        // Integrate rate of change of quaternion
        const double qa = q0_;
        const double qb = q1_;
        const double qc = q2_;
        gx *= 0.5 * dt;
        gy *= 0.5 * dt;
        gz *= 0.5 * dt;
        q0_ += (-qb * gx - qc * gy - q3_ * gz);
        q1_ += (qa * gx + qc * gz - q3_ * gy);
        q2_ += (qa * gy - qb * gz + q3_ * gx);
        q3_ += (qa * gz + qb * gy - qc * gx);

        // Normalize quaternion
        const double recipNorm = 1.0 / std::sqrt(q0_ * q0_ + q1_ * q1_ + q2_ * q2_ + q3_ * q3_);
        q0_ *= recipNorm;
        q1_ *= recipNorm;
        q2_ *= recipNorm;
        q3_ *= recipNorm;

        // Update state variables
        q_.x() = q1_;
        q_.y() = q2_;
        q_.z() = q3_;
        q_.w() = q0_;
        R_ = q_.toRotationMatrix();
        acc_ = R_ * acc;
        gyro_ = R_ * gyro;
        timestamp_ = timestamp;
    }

    /// Set the state of the Mahony filter
    void setState(const Eigen::Quaterniond& q) {
        q_ = q;
        q0_ = q.w();
        q1_ = q.x();
        q2_ = q.y();
        q3_ = q.z();
        acc_ = Eigen::Vector3d::Zero();
        gyro_ = Eigen::Vector3d::Zero();
        R_ = q_.toRotationMatrix();
        integralFBx_ = 0.0;
        integralFBy_ = 0.0;
        integralFBz_ = 0.0;
        timestamp_.reset();
    }

    /// Reset the filter
    void reset() {
        q_ = Eigen::Quaterniond::Identity();
        q0_ = 1.0;
        q1_ = 0.0;
        q2_ = 0.0;
        q3_ = 0.0;
        acc_ = Eigen::Vector3d::Zero();
        gyro_ = Eigen::Vector3d::Zero();
        R_ = Eigen::Matrix3d::Identity();
        integralFBx_ = 0.0;
        integralFBy_ = 0.0;
        integralFBz_ = 0.0;
        timestamp_.reset();
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
    double twoKp_ = 0.0;
    /// Algorithm Integral gain
    double twoKi_ = 0.0;
    /// Quaternion of sensor frame relative to auxiliary frame
    double q0_ = 1.0;
    double q1_ = 0.0;
    double q2_ = 0.0;
    double q3_ = 0.0;
    /// Integral of angular velocity in x,y,z
    double integralFBx_ = 0.0;
    double integralFBy_ = 0.0;
    double integralFBz_ = 0.0;
    /// Nominal sample time
    double nominal_dt_ = 0.0;
    /// Timestamp of the last measurement
    std::optional<double> timestamp_ = std::nullopt;
    /// Whether to print verbose output
    bool verbose_{};
};

}  // namespace serow
