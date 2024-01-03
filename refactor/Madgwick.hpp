/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
/**
 * @brief IMU Orientation Estimation with the Madgwick Filter
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

namespace serow {

class Madgwick {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /** @fn   Madgwick(double freq_, double beta_)
     *  @brief Initializes parameters and sets the sampling frequency and gain of the algorithm
     *  @param freq sampling frequency
     *  @param beta Proportional gain
     */
    Madgwick(double freq, double beta) {
        freq_ = freq;
        beta_ = beta;  // 2 * proportional gain (Kp)
    }

    /** @fn Eigen::Quaterniond getQ()
     *  @returns the orientation of IMU  w.r.t the world frame as a quaternion
     */
    Eigen::Quaterniond getQ() const { return q_; }

    /** @fn  Eigen::Vector3d getAcc()
     *  @returns the linear acceleration of IMU  in the world frame
     */
    Eigen::Vector3d getAcc() const { return acc_; }

    /** @fn  Eigen::Vector3d getGyro()
     *  @returns the angular velocity  of IMU  in the world frame
     */
    Eigen::Vector3d getGyro() const { return gyro_; }

    /** @fn Eigen::Matrix3d getR()
     *  @returns the orientation of IMU  w.r.t the world frame as a rotation matrix
     */
    Eigen::Matrix3d getR() const { return R_; }

    /** @fn Eigen::Matrix3d getEuler()
     *  @returns the orientation of IMU  w.r.t the world frame as  euler angles in the RPY
     * convention
     */
    Eigen::Vector3d getEuler() const { return q_.toRotationMatrix().eulerAngles(0, 1, 2); }

    /** @fn updateIMU(Eigen::Vector3d gyro_, Eigen::Vector3d acc_)
     *  @brief Computes the IMU orientation w.r.t the world frame of reference
     *  @param gyro_ angular velocity as measured by the IMU
     *  @param acc_ linea acceleration as measured by the IMU
     */
    void updateIMU(const Eigen::Vector3d& gyro, const Eigen::Vector3d& acc) {
        double gx = gyro(0);
        double gy = gyro(1);
        double gz = gyro(2);

        double ax = acc(0);
        double ay = acc(1);
        double az = acc(2);

        double recipNorm;
        double s0, s1, s2, s3;
        double qDot1, qDot2, qDot3, qDot4;
        double _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

        // Rate of change of quaternion from gyroscope
        qDot1 = 0.5f * (-q1_ * gx - q2_ * gy - q3_ * gz);
        qDot2 = 0.5f * (q0_ * gx + q2_ * gz - q3_ * gy);
        qDot3 = 0.5f * (q0_ * gy - q1_ * gz + q3_ * gx);
        qDot4 = 0.5f * (q0_ * gz + q1_ * gy - q2_ * gx);

        // Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer
        // normalization)
        if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f))) {
            // Normalize accelerometer measurement
            recipNorm = 1.0 / std::sqrt(ax * ax + ay * ay + az * az);
            ax *= recipNorm;
            ay *= recipNorm;
            az *= recipNorm;

            // Auxiliary variables to avoid repeated arithmetic
            _2q0 = 2.0f * q0_;
            _2q1 = 2.0f * q1_;
            _2q2 = 2.0f * q2_;
            _2q3 = 2.0f * q3_;
            _4q0 = 4.0f * q0_;
            _4q1 = 4.0f * q1_;
            _4q2 = 4.0f * q2_;
            _8q1 = 8.0f * q1_;
            _8q2 = 8.0f * q2_;
            q0q0 = q0_ * q0_;
            q1q1 = q1_ * q1_;
            q2q2 = q2_ * q2_;
            q3q3 = q3_ * q3_;

            // Gradient decent algorithm corrective step
            s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
            s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1_ - _2q0 * ay - _4q1 + _8q1 * q1q1 +
                 _8q1 * q2q2 + _4q1 * az;
            s2 = 4.0f * q0q0 * q2_ + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 +
                 _8q2 * q2q2 + _4q2 * az;
            s3 = 4.0f * q1q1 * q3_ - _2q1 * ax + 4.0f * q2q2 * q3_ - _2q2 * ay;
            // Normalize step magnitude
            recipNorm = 1.0 / std::sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
            s0 *= recipNorm;
            s1 *= recipNorm;
            s2 *= recipNorm;
            s3 *= recipNorm;

            // Apply feedback step
            qDot1 -= beta_ * s0;
            qDot2 -= beta_ * s1;
            qDot3 -= beta_ * s2;
            qDot4 -= beta_ * s3;
        }

        // Integrate rate of change of quaternion to yield quaternion
        q0_ += qDot1 * (1.0f / freq_);
        q1_ += qDot2 * (1.0f / freq_);
        q2_ += qDot3 * (1.0f / freq_);
        q3_ += qDot4 * (1.0f / freq_);

        // Normalize quaternion
        recipNorm = 1.0 / std::sqrt(q0_ * q0_ + q1_ * q1_ + q2_ * q2_ + q3_ * q3_);
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

   private:
    // IMU orientation w.r.t the world frame as a quaternion
    Eigen::Quaterniond q_ = Eigen::Quaterniond::Identity();
    // IMU orientation w.r.t the world frame as a rotation matrix
    Eigen::Matrix3d R_ = Eigen::Matrix3d::Identity();
    // IMU linear acceleration and angular velocity in the world frame
    Eigen::Vector3d acc_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_ = Eigen::Vector3d::Zero();
    /// Algorithm gain
    double beta_;
    /// Sampling frequency
    double freq_;
    /// Quaternion of sensor frame relative to auxiliary frame
    double q0_, q1_, q2_, q3_;
};

}  // namespace serow
