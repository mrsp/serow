/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace serow {

class State {
   public:
    State() = default;
    State(std::unordered_set<std::string> contacts_frame, bool point_feet);
    // State getters
    Eigen::Isometry3d getBasePose() const;
    const Eigen::Vector3d& getBasePosition() const;
    const Eigen::Quaterniond& getBaseOrientation() const;
    const Eigen::Vector3d& getBaseLinearVelocity() const;
    const Eigen::Vector3d& getBaseAngularVelocity() const;
    const Eigen::Vector3d& getImuLinearAccelarationBias() const;
    const Eigen::Vector3d& getImuAngularVelocityBias() const;
    const std::unordered_set<std::string>& getContactsFrame() const;
    std::optional<Eigen::Vector3d> getContactPosition(const std::string& frame_name) const;
    std::optional<Eigen::Quaterniond> getContactOrientation(const std::string& frame_name) const;
    std::optional<Eigen::Isometry3d> getContactPose(const std::string& frame_name) const;
    std::optional<bool> getContactStatus(const std::string& frame_name) const;

    // State covariance getter
    Eigen::Matrix<double, 6, 6> getBasePoseCov() const;
    const Eigen::Matrix3d& getBasePositionCov() const;
    const Eigen::Matrix3d& getBaseOrientationCov() const;
    const Eigen::Matrix3d& getBaseLinearVelocityCov() const;
    const Eigen::Matrix3d& getBaseAngularVelocityCov() const;
    const Eigen::Matrix3d& getImuLinearAccelerationBiasCov() const;
    const Eigen::Matrix3d& getImuAngularVelocityBiasCov() const;
    std::optional<Eigen::Matrix<double, 6, 6>> getContactPoseCov(
        const std::string& frame_name) const;
    std::optional<Eigen::Matrix3d> getContactPositionCov(const std::string& frame_name) const;
    std::optional<Eigen::Matrix3d> getContactOrientationCov(const std::string& frame_name) const;

    // Flag to indicate if the robot has point feet. False indicates flat feet contacts.
    bool point_feet_{};

    int num_leg_ee_{};
    // Base position in the world frame
    Eigen::Vector3d base_position_{Eigen::Vector3d::Zero()};
    // Base orientation in the world frame
    Eigen::Quaterniond base_orientation_{Eigen::Quaterniond::Identity()};
    // Base linear velocity in the world frame
    Eigen::Vector3d base_linear_velocity_{Eigen::Vector3d::Zero()};
    // Base angular velocity in the world frame
    Eigen::Vector3d base_angular_velocity_{Eigen::Vector3d::Zero()};
    // Contact state: frame_name to contact pose in the world frame
    std::unordered_map<std::string, Eigen::Vector3d> contacts_position_;
    std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientation_;
    std::unordered_map<std::string, bool> contacts_status_;
    std::unordered_map<std::string, double> contacts_probability_;
    std::unordered_set<std::string> contacts_frame_;
    // Imu acceleration bias in the local imu frame
    Eigen::Vector3d imu_linear_acceleration_bias_{Eigen::Vector3d::Zero()};
    // Imu gyro rate bias in the local imu frame
    Eigen::Vector3d imu_angular_velocity_bias_{Eigen::Vector3d::Zero()};
    // 3D CoM position
    Eigen::Vector3d com_position_{Eigen::Vector3d::Zero()};
    // 3D CoM velocity
    Eigen::Vector3d com_linear_velocity_{Eigen::Vector3d::Zero()};
    // 3D External forces at the CoM
    Eigen::Vector3d external_forces_{Eigen::Vector3d::Zero()};

    // Covariances
    // Base position covariance in the world frame
    Eigen::Matrix3d base_position_cov_{Eigen::Matrix3d::Zero()};
    // Base orientation covariance in the world frame
    Eigen::Matrix3d base_orientation_cov_{Eigen::Matrix3d::Zero()};
    // Base linear velocity covariance in the world frame
    Eigen::Matrix3d base_linear_velocity_cov_{Eigen::Matrix3d::Zero()};
    // Base angular velocity covariance in the world frame
    Eigen::Matrix3d base_angular_velocity_cov_{Eigen::Matrix3d::Zero()};
    // Imu acceleration bias covariance in the local imu frame
    Eigen::Matrix3d imu_linear_acceleration_bias_cov_{Eigen::Matrix3d::Zero()};
    // Imu gyro rate bias covariance in the local imu frame
    Eigen::Matrix3d imu_angular_velocity_bias_cov_{Eigen::Matrix3d::Zero()};
    // Feet state: frame_name to contacts pose covariance in the world frame
    std::unordered_map<std::string, Eigen::Matrix3d> contacts_position_cov_;
    std::optional<std::unordered_map<std::string, Eigen::Matrix3d>> contacts_orientation_cov_;

    std::unordered_map<std::string, Eigen::Vector3d> contact_forces;
    std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contact_torques;

    friend class Serow;
    friend class ContactEKF;
    friend class CoMEKF;
};

}  // namespace serow
