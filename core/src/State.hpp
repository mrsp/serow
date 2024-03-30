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
#include <mutex>
#include <optional>
#include <string>
#include <map>
#include <set>

namespace serow {



struct BaseState {
    // last time the state was updated
    double timestamp{};
    // Base position in the world frame
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};
    // Base orientation in the world frame
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()};
    // Base linear velocity in the world frame
    Eigen::Vector3d base_linear_velocity{Eigen::Vector3d::Zero()};
    // Base angular velocity in the world frame
    Eigen::Vector3d base_angular_velocity{Eigen::Vector3d::Zero()};
    // Base linear acceleration in the world frame
    Eigen::Vector3d base_linear_acceleration{Eigen::Vector3d::Zero()};
    // Base angular acceleration in the world frame
    Eigen::Vector3d base_angular_acceleration{Eigen::Vector3d::Zero()};
    // Imu acceleration bias in the local base frame
    Eigen::Vector3d imu_linear_acceleration_bias{Eigen::Vector3d::Zero()};
    // Imu gyro rate bias in the local base frame
    Eigen::Vector3d imu_angular_velocity_bias{Eigen::Vector3d::Zero()};
    // Contact state: frame_name to contact pose in the world frame
    std::map<std::string, Eigen::Vector3d> contacts_position;
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation;

    // Base position covariance in the world frame
    Eigen::Matrix3d base_position_cov{Eigen::Matrix3d::Zero()};
    // Base orientation covariance in the world frame
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Zero()};
    // Base linear velocity covariance in the world frame
    Eigen::Matrix3d base_linear_velocity_cov{Eigen::Matrix3d::Zero()};
    // Base angular velocity covariance in the world frame
    Eigen::Matrix3d base_angular_velocity_cov{Eigen::Matrix3d::Zero()};
    // Imu acceleration bias covariance in the local imu frame
    Eigen::Matrix3d imu_linear_acceleration_bias_cov{Eigen::Matrix3d::Zero()};
    // Imu gyro rate bias covariance in the local imu frame
    Eigen::Matrix3d imu_angular_velocity_bias_cov{Eigen::Matrix3d::Zero()};
    // Feet state: frame_name to contacts pose covariance in the world frame
    std::map<std::string, Eigen::Matrix3d> contacts_position_cov;
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_cov;
};

struct CentroidalState {
    // last time the state was updated
    double timestamp{};
    // 3D CoM position in the world frame
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};
    // 3D CoM linear velocity in the world frame
    Eigen::Vector3d com_linear_velocity{Eigen::Vector3d::Zero()};
    // 3D External forces at the CoM in the world frame
    Eigen::Vector3d external_forces{Eigen::Vector3d::Zero()};
    // COP position in the world frame
    Eigen::Vector3d cop_position{Eigen::Vector3d::Zero()};
    // CoM linear acceleration in the world frame
    Eigen::Vector3d com_linear_acceleration{Eigen::Vector3d::Zero()};
    // Angular momentum around the CoM in the world frame
    Eigen::Vector3d angular_momentum{Eigen::Vector3d::Zero()};
    // Angular momentum derivative around the CoM in the world frame
    Eigen::Vector3d angular_momentum_derivative{Eigen::Vector3d::Zero()};

    // 3D CoM position covariance in world frame
    Eigen::Matrix3d com_position_cov{Eigen::Matrix3d::Identity()};
    // 3D CoM linear velocity covariance in world frame
    Eigen::Matrix3d com_linear_velocity_cov{Eigen::Matrix3d::Identity()};
    // 3D External forces at the CoM covariance in world frame
    Eigen::Matrix3d external_forces_cov{Eigen::Matrix3d::Identity()};
};

struct ContactState {
    // last time the state was updated
    double timestamp{};
    std::map<std::string, bool> contacts_status;
    std::map<std::string, double> contacts_probability;
    std::map<std::string, Eigen::Vector3d> contacts_force;
    std::optional<std::map<std::string, Eigen::Vector3d>> contacts_torque;
};

struct JointState {
    // last time the state was updated
    double timestamp{};
    std::map<std::string, double> joints_position;
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

    // State getters
    Eigen::Isometry3d getBasePose() const;
    const Eigen::Vector3d& getBasePosition() const;
    const Eigen::Quaterniond& getBaseOrientation() const;
    const Eigen::Vector3d& getBaseLinearVelocity() const;
    const Eigen::Vector3d& getBaseAngularVelocity() const;
    const Eigen::Vector3d& getImuLinearAccelerationBias() const;
    const Eigen::Vector3d& getImuAngularVelocityBias() const;
    const std::set<std::string>& getContactsFrame() const;
    std::optional<Eigen::Vector3d> getContactPosition(const std::string& frame_name) const;
    std::optional<Eigen::Quaterniond> getContactOrientation(const std::string& frame_name) const;
    std::optional<Eigen::Isometry3d> getContactPose(const std::string& frame_name) const;
    std::optional<bool> getContactStatus(const std::string& frame_name) const;

    const Eigen::Vector3d& getCoMPosition() const;
    const Eigen::Vector3d& getCoMLinearVelocity() const;
    const Eigen::Vector3d& getCoMExternalForces() const;

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
    const Eigen::Matrix3d& getCoMPositionCov() const;
    const Eigen::Matrix3d& getCoMLinearVelocityCov() const;
    const Eigen::Matrix3d& getCoMExternalForcesCov() const;

   private:
    // Flag to indicate if the robot has point feet. False indicates flat feet contacts.
    bool point_feet_{};
    int num_leg_ee_{};
    bool is_valid_{};
    std::set<std::string> contacts_frame_;
    
    // Individual states
    JointState joint_state_;
    ContactState contact_state_;
    BaseState base_state_;
    CentroidalState centroidal_state_;

    std::mutex mutex_;

    friend class Serow;
    friend class ContactEKF;
    friend class CoMEKF;
};

}  // namespace serow
