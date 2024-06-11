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
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>

namespace serow {

struct BaseState {
    /// last time the state was updated
    double timestamp{};
    /// Base position in world frame coordinates
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};
    /// Base orientation in world frame coordinates
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()};
    /// Base linear velocity in world frame coordinates
    Eigen::Vector3d base_linear_velocity{Eigen::Vector3d::Zero()};
    /// Base angular velocity in world frame coordinates
    Eigen::Vector3d base_angular_velocity{Eigen::Vector3d::Zero()};
    /// Base linear acceleration in world frame coordinates
    Eigen::Vector3d base_linear_acceleration{Eigen::Vector3d::Zero()};
    /// Base angular acceleration in world frame coordinates
    Eigen::Vector3d base_angular_acceleration{Eigen::Vector3d::Zero()};
    /// Imu acceleration bias in local base frame coordinates
    Eigen::Vector3d imu_linear_acceleration_bias{Eigen::Vector3d::Zero()};
    /// Imu gyro rate bias in local base frame coordinates
    Eigen::Vector3d imu_angular_velocity_bias{Eigen::Vector3d::Zero()};
    /// Holds contact frame name to 3D contact position in world frame coordinates
    std::map<std::string, Eigen::Vector3d> contacts_position;
    /// Holds contact frame name to 3D contact orientation in world frame coordinates, only applies
    /// if the robot has flat feet
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation;

    /// Base position covariance in world frame coordinates
    Eigen::Matrix3d base_position_cov{Eigen::Matrix3d::Zero()};
    /// Base orientation covariance in world frame coordinates
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Zero()};
    /// Base linear velocity covariance in world frame coordinates
    Eigen::Matrix3d base_linear_velocity_cov{Eigen::Matrix3d::Zero()};
    /// Base angular velocity covariance in world frame coordinates
    Eigen::Matrix3d base_angular_velocity_cov{Eigen::Matrix3d::Zero()};
    /// Imu acceleration bias covariance in local base frame coordinates
    Eigen::Matrix3d imu_linear_acceleration_bias_cov{Eigen::Matrix3d::Zero()};
    /// Imu gyro rate bias covariance in local base frame coordinates
    Eigen::Matrix3d imu_angular_velocity_bias_cov{Eigen::Matrix3d::Zero()};
    /// Holds contact frame name to 3D contact position covariance in world frame coordinates
    std::map<std::string, Eigen::Matrix3d> contacts_position_cov;
    /// Holds contact frame name to 3D contact orientation covariance in world frame coordinates,
    /// only applies if the robot has flat feet
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_cov;
};

struct CentroidalState {
    /// last time the state was updated
    double timestamp{};
    /// 3D CoM position in world frame coordinates
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};
    /// 3D CoM linear velocity in world frame coordinates
    Eigen::Vector3d com_linear_velocity{Eigen::Vector3d::Zero()};
    /// 3D External forces at the CoM in world frame coordinates
    Eigen::Vector3d external_forces{Eigen::Vector3d::Zero()};
    /// 3D COP position in world frame coordinates
    Eigen::Vector3d cop_position{Eigen::Vector3d::Zero()};
    /// 3D CoM linear acceleration in world frame coordinates
    Eigen::Vector3d com_linear_acceleration{Eigen::Vector3d::Zero()};
    /// 3D Angular momentum around the CoM in world frame coordinates
    Eigen::Vector3d angular_momentum{Eigen::Vector3d::Zero()};
    /// 3D Angular momentum derivative around the CoM in world frame coordinates
    Eigen::Vector3d angular_momentum_derivative{Eigen::Vector3d::Zero()};

    /// 3D CoM position covariance in world frame coordinates
    Eigen::Matrix3d com_position_cov{Eigen::Matrix3d::Identity()};
    /// 3D CoM linear velocity covariance in world frame coordinates
    Eigen::Matrix3d com_linear_velocity_cov{Eigen::Matrix3d::Identity()};
    /// 3D External forces at the CoM covariance in world frame coordinates
    Eigen::Matrix3d external_forces_cov{Eigen::Matrix3d::Identity()};
};

struct ContactState {
    /// last time the state was updated
    double timestamp{};
    /// Holds contact frame name to binary contact indicator
    std::map<std::string, bool> contacts_status;
    /// Holds contact frame name to continuous contact probability
    std::map<std::string, double> contacts_probability;
    /// Holds contact frame name to 3D ground reaction force in world frame coordinates
    std::map<std::string, Eigen::Vector3d> contacts_force;
    /// Holds contact frame name to 3D ground reaction torque in world frame coordinates
    std::optional<std::map<std::string, Eigen::Vector3d>> contacts_torque;
};

struct JointState {
    /// last time the state was updated
    double timestamp{};
    /// Holds joint name to joint angular position in joint coordinates
    std::map<std::string, double> joints_position;
    /// Holds joint name to joint angular velocity in joint coordinates
    std::map<std::string, double> joints_velocity;
};

class State {
   public:
    State() = default;
    State(std::set<std::string> contacts_frame, bool point_feet);
    State(const State& other);
    State(State&& other);
    State operator=(const State& other);
    State& operator=(State&& other);
    bool isPointFeet() const;

    /// State getters
    /// Returns the 3D base pose as a rigid transformation in world frame coordinates
    Eigen::Isometry3d getBasePose() const;
    /// Returns the 3D base position in world frame coordinates
    const Eigen::Vector3d& getBasePosition() const;
    /// Returns the 3D base orientation in world frame coordinates
    const Eigen::Quaterniond& getBaseOrientation() const;
    /// Returns the 3D base linear velocity in world frame coordinates
    const Eigen::Vector3d& getBaseLinearVelocity() const;
    /// Returns the 3D base angular velocity in world frame coordinates
    const Eigen::Vector3d& getBaseAngularVelocity() const;
    /// Returns the 3D IMU linear acceleration bias in the local base frame
    const Eigen::Vector3d& getImuLinearAccelerationBias() const;
    /// Returns the 3D IMU angular velocity bias in the local base frame
    const Eigen::Vector3d& getImuAngularVelocityBias() const;
    /// Returns the active (if any) contact frame names
    const std::set<std::string>& getContactsFrame() const;
    /// Returns the contact frame 3D position if the frame is in contact in world frame coordinates
    std::optional<Eigen::Vector3d> getContactPosition(const std::string& frame_name) const;
    /// Returns the contact frame 3D orientation in world frame coordinates if the frame is in
    /// contact. Only applies if the robot has flat feet
    std::optional<Eigen::Quaterniond> getContactOrientation(const std::string& frame_name) const;
    /// Returns the contact frame 3D pose in world frame coordinates as a rigid transformation if
    /// the frame is in contact. Only applies if the robot has flat feet
    std::optional<Eigen::Isometry3d> getContactPose(const std::string& frame_name) const;
    /// Returns the contact frame binary contact status if the frame is in contact. Only applies if
    /// the robot has flat feet
    std::optional<bool> getContactStatus(const std::string& frame_name) const;

    /// Returns the 3D CoM position in world frame coordinates
    const Eigen::Vector3d& getCoMPosition() const;
    /// Returns the 3D CoM linear velocity in world frame coordinates
    const Eigen::Vector3d& getCoMLinearVelocity() const;
    /// Returns the 3D CoM external forces in world frame coordinates
    const Eigen::Vector3d& getCoMExternalForces() const;

    /// State covariance getter
    /// Returns the 3D base pose as a 6 x 6 matrix in world frame coordinates
    Eigen::Matrix<double, 6, 6> getBasePoseCov() const;
    /// Returns the 3D base velocity as a 6 x 6 matrix in world frame coordinates
    Eigen::Matrix<double, 6, 6> getBaseVelocityCov() const;
    /// Returns the 3D base position covariance in world frame coordinates
    const Eigen::Matrix3d& getBasePositionCov() const;
    /// Returns the 3D base orientation covariance in world frame coordinates
    const Eigen::Matrix3d& getBaseOrientationCov() const;
    /// Returns the 3D base linear velocity covariance in world frame coordinates
    const Eigen::Matrix3d& getBaseLinearVelocityCov() const;
    /// Returns the 3D base angular velocity covariance in world frame coordinates
    const Eigen::Matrix3d& getBaseAngularVelocityCov() const;
    /// Returns the 3D IMU linear acceleration bias coviariance in the local base frame
    const Eigen::Matrix3d& getImuLinearAccelerationBiasCov() const;
    /// Returns the 3D IMU angular velocity bias covariance in the local base frame
    const Eigen::Matrix3d& getImuAngularVelocityBiasCov() const;
    /// Returns the contact frame 3D pose covariance in world frame coordinates as a 6 x 6 matrix if
    /// the frame is in contact. Only applies if the robot has flat feet
    std::optional<Eigen::Matrix<double, 6, 6>> getContactPoseCov(
        const std::string& frame_name) const;
    /// Returns the contact frame 3D position covariance if the frame is in contact in world frame
    /// coordinates
    std::optional<Eigen::Matrix3d> getContactPositionCov(const std::string& frame_name) const;
    /// Returns the contact frame 3D orientation covariance in world frame coordinates if the frame
    /// is in contact. Only applies if the robot has flat feet
    std::optional<Eigen::Matrix3d> getContactOrientationCov(const std::string& frame_name) const;

    /// Returns the 3D CoM position covariance in world frame coordinates
    const Eigen::Matrix3d& getCoMPositionCov() const;
    /// Returns the 3D CoM linear velocity covariance in world frame coordinates
    const Eigen::Matrix3d& getCoMLinearVelocityCov() const;
    /// Returns the 3D CoM external forces covariance in world frame coordinates
    const Eigen::Matrix3d& getCoMExternalForcesCov() const;

   private:
    /// Flag to indicate if the robot has point feet. False indicates flat feet contacts
    bool point_feet_{};
    /// Number of leg end-effectors
    int num_leg_ee_{};
    /// Flag to indicate if the state is valid
    bool is_valid_{};
    /// Leg contact frames
    std::set<std::string> contacts_frame_;

    /// Individual states
    JointState joint_state_;
    ContactState contact_state_;
    BaseState base_state_;
    CentroidalState centroidal_state_;

    /// Lock for safely copying or moving the State
    std::mutex mutex_;

    friend class Serow;
    friend class ContactEKF;
    friend class CoMEKF;
};

}  // namespace serow
