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
 * @file state.h
 * @brief Defines data structures and accessors for the robot states in the SEROW state estimator.
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

/**
 * @brief Represents the base state of the robot.
 */
struct BaseState {
    /// last time the state was updated (s)
    double timestamp{};
    /// Base position in world frame coordinates (m)
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};
    /// Base orientation in world frame coordinates
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()};
    /// Base linear velocity in world frame coordinates (m/s)
    Eigen::Vector3d base_linear_velocity{Eigen::Vector3d::Zero()};
    /// Base angular velocity in world frame coordinates (rad/s)
    Eigen::Vector3d base_angular_velocity{Eigen::Vector3d::Zero()};
    /// Base linear acceleration in world frame coordinates (m/s^2)
    Eigen::Vector3d base_linear_acceleration{Eigen::Vector3d::Zero()};
    /// Base angular acceleration in world frame coordinates (rad/s^2)
    Eigen::Vector3d base_angular_acceleration{Eigen::Vector3d::Zero()};
    /// Imu acceleration bias in local imu frame coordinates (m/s^2)
    Eigen::Vector3d imu_linear_acceleration_bias{Eigen::Vector3d::Zero()};
    /// Imu gyro rate bias in local imu frame coordinates (rad/s)
    Eigen::Vector3d imu_angular_velocity_bias{Eigen::Vector3d::Zero()};
    /// Holds contact frame name to 3D contact position in world frame coordinates (m)
    std::map<std::string, Eigen::Vector3d> contacts_position;
    /// Holds contact frame name to 3D contact orientation in world frame coordinates, only applies
    /// if the robot has flat feet
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation;
    /// Holds contact frame name to 3D foot position in world frame coordinates (m)
    std::map<std::string, Eigen::Vector3d> feet_position;
    /// Holds contact frame name to 3D foot orientation in world frame coordinates
    std::map<std::string, Eigen::Quaterniond> feet_orientation;
    /// Holds contact frame name to 3D foot linear velocity in world frame coordinates (m/s)
    std::map<std::string, Eigen::Vector3d> feet_linear_velocity;
    /// Holds contact frame name to 3D foot angular velocity in world frame coordinates (rad/s)
    std::map<std::string, Eigen::Vector3d> feet_angular_velocity;

    /// Base position covariance in world frame coordinates (m^2)
    Eigen::Matrix3d base_position_cov{Eigen::Matrix3d::Identity()};
    /// Base orientation covariance in world frame coordinates (rad^2)
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Identity()};
    /// Base linear velocity covariance in world frame coordinates (m^2/s^2)
    Eigen::Matrix3d base_linear_velocity_cov{Eigen::Matrix3d::Identity()};
    /// Base angular velocity covariance in world frame coordinates (rad^2/s^2)
    Eigen::Matrix3d base_angular_velocity_cov{Eigen::Matrix3d::Identity()};
    /// Imu acceleration bias covariance in local imu frame coordinates (m^2/s^4)
    Eigen::Matrix3d imu_linear_acceleration_bias_cov{Eigen::Matrix3d::Identity()};
    /// Imu gyro rate bias covariance in local imu frame coordinates (rad^2/s^2)
    Eigen::Matrix3d imu_angular_velocity_bias_cov{Eigen::Matrix3d::Identity()};
    /// Holds contact frame name to 3D contact position covariance in world frame coordinates (m^2)
    std::map<std::string, Eigen::Matrix3d> contacts_position_cov;
    /// Holds contact frame name to 3D contact orientation covariance in world frame coordinates,
    /// (rad^2) only applies if the robot has flat feet
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_cov;
};

/**
 * @brief Represents the centroidal state of the robot.
 */
struct CentroidalState {
    /// last time the state was updated (s)
    double timestamp{};
    /// 3D CoM position in world frame coordinates (m)
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};
    /// 3D CoM linear velocity in world frame coordinates (m/s)
    Eigen::Vector3d com_linear_velocity{Eigen::Vector3d::Zero()};
    /// 3D External forces at the CoM in world frame coordinates (N)
    Eigen::Vector3d external_forces{Eigen::Vector3d::Zero()};
    /// 3D COP position in world frame coordinates (m)
    Eigen::Vector3d cop_position{Eigen::Vector3d::Zero()};
    /// 3D CoM linear acceleration in world frame coordinates (m/s^2)
    Eigen::Vector3d com_linear_acceleration{Eigen::Vector3d::Zero()};
    /// 3D Angular momentum around the CoM in world frame coordinates (kg m^2/s)
    Eigen::Vector3d angular_momentum{Eigen::Vector3d::Zero()};
    /// 3D Angular momentum derivative around the CoM in world frame coordinates (Nm)
    Eigen::Vector3d angular_momentum_derivative{Eigen::Vector3d::Zero()};

    /// 3D CoM position covariance in world frame coordinates (m^2)
    Eigen::Matrix3d com_position_cov{Eigen::Matrix3d::Identity()};
    /// 3D CoM linear velocity covariance in world frame coordinates (m^2/s^2)
    Eigen::Matrix3d com_linear_velocity_cov{Eigen::Matrix3d::Identity()};
    /// 3D External forces at the CoM covariance in world frame coordinates (N^2)
    Eigen::Matrix3d external_forces_cov{Eigen::Matrix3d::Identity()};
};

/**
 * @brief Represents the contact state of the robot.
 */
struct ContactState {
    /// last time the state was updated (s)
    double timestamp{};
    /// Holds contact frame name to binary contact indicator (0 or 1)
    std::map<std::string, bool> contacts_status;
    /// Holds contact frame name to continuous contact probability ([0, 1])
    std::map<std::string, double> contacts_probability;
    /// Holds contact frame name to 3D ground reaction force in world frame coordinates (N)
    std::map<std::string, Eigen::Vector3d> contacts_force;
    /// Holds contact frame name to 3D ground reaction torque in world frame coordinates (Nm)
    std::optional<std::map<std::string, Eigen::Vector3d>> contacts_torque;
};

/**
 * @brief Represents the joint state of the robot.
 */
struct JointState {
    /// last time the state was updated (s)
    double timestamp{};
    /// Holds joint name to joint angular position in joint coordinates (rad)
    std::map<std::string, double> joints_position;
    /// Holds joint name to joint angular velocity in joint coordinates (rad/s)
    std::map<std::string, double> joints_velocity;
};

/**
 * @brief Represents the overall state of the robot including base, centroidal, contact, and joint
 * states.
 */
class State {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
    /// Returns the contact forces in world frame.
    std::optional<Eigen::Vector3d> getContactForce(const std::string& frame_name) const;

    /// Returns the foot frame 3D position in world frame coordinates
    const Eigen::Vector3d& getFootPosition(const std::string& frame_name) const;
    /// Returns the foot frame 3D orientation in world frame coordinates
    const Eigen::Quaterniond& getFootOrientation(const std::string& frame_name) const;
    /// Returns the foot frame 3D pose in world frame coordinates as a rigid transformation
    Eigen::Isometry3d getFootPose(const std::string& frame_name) const;
    /// Returns the foot frame 3D linear velocity in world frame coordinates
    const Eigen::Vector3d& getFootLinearVelocity(const std::string& frame_name) const;
    /// Returns the foot frame 3D angular velocity in world frame coordinates
    const Eigen::Vector3d& getFootAngularVelocity(const std::string& frame_name) const;

    /// Returns the 3D CoM position in world frame coordinates
    const Eigen::Vector3d& getCoMPosition() const;
    /// Returns the 3D CoM linear velocity in world frame coordinates
    const Eigen::Vector3d& getCoMLinearVelocity() const;
    /// Returns the 3D CoM external forces in world frame coordinates
    const Eigen::Vector3d& getCoMExternalForces() const;
    /// Returns the 3D angular momentum around the CoM in world frame coordinates
    const Eigen::Vector3d& getCoMAngularMomentum() const;
    /// Returns the 3D angular momentum rate around the CoM in world frame coordinates
    const Eigen::Vector3d& getCoMAngularMomentumRate() const;
    /// Returns the 3D CoM linear acceleration in world frame coordinates approximated with the base
    /// IMU
    const Eigen::Vector3d& getCoMLinearAcceleration() const;
    /// Returns the 3D COP position in world frame coordinates
    const Eigen::Vector3d& getCOPPosition() const;

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
    /// Returns the mass of the robot
    double getMass() const;
    /// Returns the number of leg end-effectors
    int getNumLegEE() const;
    /// Returns the flag to indicate if the robot has point feet
    bool isPointFeet() const;
    /// Returns the flag to indicate if the state is valid
    bool isValid() const;
    /// Returns the flag to indicate if the state is initialized
    bool isInitialized() const;
    /// Sets the flag to indicate if the state is valid
    void setValid(bool valid);
    /// Sets the flag to indicate if the state is initialized
    void setInitialized(bool initialized);

    State() = default;
    State(std::set<std::string> contacts_frame, bool point_feet);

    void setBaseState(const BaseState& base_state);
    BaseState getBaseState() const;

    void setContactState(const ContactState& contact_state);
    ContactState getContactState() const;

    void setCentroidalState(const CentroidalState& centroidal_state);
    CentroidalState getCentroidalState() const;

    void setJointState(const JointState& joint_state);
    JointState getJointState() const;

private:
    /// Flag to indicate if the robot has point feet. False indicates flat feet contacts
    bool point_feet_{};
    /// Number of leg end-effectors
    int num_leg_ee_{};
    /// Flag to indicate if the state has converged and is valid
    bool is_valid_{};
    /// Leg contact frames
    std::set<std::string> contacts_frame_;
    /// Robot mass (kg)
    double mass_{};
    /// Flag to indicate if the state is filled with valid data
    bool is_initialized_{};

    /// Individual states
    JointState joint_state_;
    ContactState contact_state_;
    BaseState base_state_;
    CentroidalState centroidal_state_;

    friend class Serow;
    friend class ContactEKF;
    friend class CoMEKF;
};

}  // namespace serow
