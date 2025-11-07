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
#include "State.hpp"

namespace serow {

State::State(const std::set<std::string>& contacts_frame, bool point_feet,
             const std::string& base_frame) {
    contacts_frame_ = contacts_frame;
    base_frame_ = base_frame;
    num_leg_ee_ = contacts_frame_.size();
    point_feet_ = point_feet;
    is_valid_ = false;
    is_initialized_ = false;

    std::map<std::string, Eigen::Quaterniond> contacts_orientation;
    std::map<std::string, Eigen::Matrix3d> contacts_orientation_cov;
    std::map<std::string, Eigen::Vector3d> contacts_torque;
    for (const auto& cf : contacts_frame_) {
        contact_state_.contacts_status[cf] = false;
        contact_state_.contacts_probability[cf] = 0.0;
        contact_state_.contacts_force[cf] = Eigen::Vector3d::Zero();
        base_state_.contacts_position[cf] = Eigen::Vector3d::Zero();
        base_state_.contacts_position_cov[cf] = Eigen::Matrix3d::Identity();
        base_state_.feet_position[cf] = Eigen::Vector3d::Zero();
        base_state_.feet_orientation[cf] = Eigen::Quaterniond::Identity();
        base_state_.feet_linear_velocity[cf] = Eigen::Vector3d::Zero();
        base_state_.feet_angular_velocity[cf] = Eigen::Vector3d::Zero();

        if (!isPointFeet()) {
            contacts_torque[cf] = Eigen::Vector3d::Zero();
            contacts_orientation[cf] = Eigen::Quaterniond::Identity();
            contacts_orientation_cov[cf] = Eigen::Matrix3d::Identity();
        }
    }
    if (!isPointFeet()) {
        base_state_.contacts_orientation = std::move(contacts_orientation);
        base_state_.contacts_orientation_cov = std::move(contacts_orientation_cov);
        contact_state_.contacts_torque = std::move(contacts_torque);
    }
}

double State::getTimestamp(const std::string& state_type) const {
    if (state_type == "base") {
        return base_state_.timestamp;
    } else if (state_type == "joint") {
        return joint_state_.timestamp;
    } else if (state_type == "centroidal") {
        return centroidal_state_.timestamp;
    } else if (state_type == "contact") {
        return contact_state_.timestamp;
    } else {
        throw std::invalid_argument("Invalid state type");
    }
}

Eigen::Isometry3d State::getBasePose() const {
    Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
    base_pose.linear() = base_state_.base_orientation.toRotationMatrix();
    base_pose.translation() = base_state_.base_position;
    return base_pose;
}

const Eigen::Vector3d& State::getBasePosition() const {
    return base_state_.base_position;
}

const Eigen::Quaterniond& State::getBaseOrientation() const {
    return base_state_.base_orientation;
}

const Eigen::Vector3d& State::getBaseLinearVelocity() const {
    return base_state_.base_linear_velocity;
}

const Eigen::Vector3d& State::getBaseAngularVelocity() const {
    return base_state_.base_angular_velocity;
}

const Eigen::Vector3d& State::getImuLinearAccelerationBias() const {
    return base_state_.imu_linear_acceleration_bias;
}

const Eigen::Vector3d& State::getImuAngularVelocityBias() const {
    return base_state_.imu_angular_velocity_bias;
}

std::optional<Eigen::Vector3d> State::getContactPosition(const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact position
    // available
    if (contact_state_.contacts_status.count(frame_name) &&
        contact_state_.contacts_status.at(frame_name) &&
        base_state_.contacts_position.count(frame_name)) {
        return base_state_.contacts_position.at(frame_name);
    } else {
        return std::nullopt;
    }
}

std::optional<Eigen::Quaterniond> State::getContactOrientation(
    const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact orientation
    // available
    if (contact_state_.contacts_status.count(frame_name) &&
        contact_state_.contacts_status.at(frame_name) &&
        base_state_.contacts_orientation.has_value() &&
        base_state_.contacts_orientation.value().count(frame_name)) {
        return base_state_.contacts_orientation.value().at(frame_name);
    } else {
        return std::nullopt;
    }
}

std::optional<Eigen::Isometry3d> State::getContactPose(const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact orientation
    // available
    if (contact_state_.contacts_status.count(frame_name) &&
        contact_state_.contacts_status.at(frame_name) &&
        base_state_.contacts_position.count(frame_name) &&
        base_state_.contacts_orientation.has_value() &&
        base_state_.contacts_orientation.value().count(frame_name)) {
        Eigen::Isometry3d contact_pose = Eigen::Isometry3d::Identity();
        contact_pose.linear() =
            base_state_.contacts_orientation.value().at(frame_name).toRotationMatrix();
        contact_pose.translation() = base_state_.contacts_position.at(frame_name);
        return contact_pose;
    } else {
        return std::nullopt;
    }
}

const std::set<std::string>& State::getContactsFrame() const {
    return contacts_frame_;
}

std::optional<bool> State::getContactStatus(const std::string& frame_name) const {
    if (contact_state_.contacts_status.count(frame_name))
        return contact_state_.contacts_status.at(frame_name);
    else
        return std::nullopt;
}

std::optional<Eigen::Vector3d> State::getContactForce(const std::string& frame_name) const {
    if (contact_state_.contacts_force.count(frame_name) > 0) {
        return contact_state_.contacts_force.at(frame_name);
    }
    return std::nullopt;
}

std::optional<Eigen::Vector3d> State::getContactTorque(const std::string& frame_name) const {
    if (contact_state_.contacts_torque.has_value() &&
        contact_state_.contacts_torque.value().count(frame_name)) {
        return contact_state_.contacts_torque.value().at(frame_name);
    }
    return std::nullopt;
}

std::optional<double> State::getContactProbability(const std::string& frame_name) const {
    if (contact_state_.contacts_probability.count(frame_name)) {
        return contact_state_.contacts_probability.at(frame_name);
    }
    return std::nullopt;
}

const Eigen::Vector3d& State::getFootPosition(const std::string& frame_name) const {
    return base_state_.feet_position.at(frame_name);
}

const Eigen::Quaterniond& State::getFootOrientation(const std::string& frame_name) const {
    return base_state_.feet_orientation.at(frame_name);
}

Eigen::Isometry3d State::getFootPose(const std::string& frame_name) const {
    Eigen::Isometry3d foot_pose = Eigen::Isometry3d::Identity();
    foot_pose.linear() = base_state_.feet_orientation.at(frame_name).toRotationMatrix();
    foot_pose.translation() = base_state_.feet_position.at(frame_name);
    return foot_pose;
}

const Eigen::Vector3d& State::getFootLinearVelocity(const std::string& frame_name) const {
    return base_state_.feet_linear_velocity.at(frame_name);
}

const Eigen::Vector3d& State::getFootAngularVelocity(const std::string& frame_name) const {
    return base_state_.feet_angular_velocity.at(frame_name);
}

Eigen::Matrix<double, 6, 6> State::getBasePoseCov() const {
    Eigen::Matrix<double, 6, 6> base_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
    base_pose_cov.block<3, 3>(0, 0) = base_state_.base_position_cov;
    base_pose_cov.block<3, 3>(3, 3) = base_state_.base_orientation_cov;
    return base_pose_cov;
}

const Eigen::Matrix3d& State::getBasePositionCov() const {
    return base_state_.base_position_cov;
}

const Eigen::Matrix3d& State::getBaseOrientationCov() const {
    return base_state_.base_orientation_cov;
}

Eigen::Matrix<double, 6, 6> State::getBaseVelocityCov() const {
    Eigen::Matrix<double, 6, 6> base_velocity_cov = Eigen::Matrix<double, 6, 6>::Identity();
    base_velocity_cov.block<3, 3>(0, 0) = base_state_.base_linear_velocity_cov;
    base_velocity_cov.block<3, 3>(3, 3) = base_state_.base_angular_velocity_cov;
    return base_velocity_cov;
}

const Eigen::Matrix3d& State::getBaseLinearVelocityCov() const {
    return base_state_.base_linear_velocity_cov;
}

const Eigen::Matrix3d& State::getBaseAngularVelocityCov() const {
    return base_state_.base_angular_velocity_cov;
}

const Eigen::Matrix3d& State::getImuLinearAccelerationBiasCov() const {
    return base_state_.imu_linear_acceleration_bias_cov;
}

const Eigen::Matrix3d& State::getImuAngularVelocityBiasCov() const {
    return base_state_.imu_angular_velocity_bias_cov;
}

std::optional<Eigen::Matrix3d> State::getContactPositionCov(const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact position
    // covariance available
    if (contact_state_.contacts_status.count(frame_name) &&
        contact_state_.contacts_status.at(frame_name) &&
        base_state_.contacts_position_cov.count(frame_name)) {
        return base_state_.contacts_position_cov.at(frame_name);
    } else {
        return std::nullopt;
    }
}

std::optional<Eigen::Matrix3d> State::getContactOrientationCov(
    const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact orientation
    // covariance available
    if (contact_state_.contacts_status.count(frame_name) &&
        contact_state_.contacts_status.at(frame_name) &&
        base_state_.contacts_orientation_cov.has_value() &&
        base_state_.contacts_orientation.value().count(frame_name))
        return base_state_.contacts_orientation_cov.value().at(frame_name);
    else
        return std::nullopt;
}

std::optional<Eigen::Matrix<double, 6, 6>> State::getContactPoseCov(
    const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact pose
    // covariance available
    if (contact_state_.contacts_status.count(frame_name) &&
        contact_state_.contacts_status.at(frame_name) &&
        base_state_.contacts_position_cov.count(frame_name) &&
        base_state_.contacts_orientation_cov.has_value() &&
        base_state_.contacts_orientation_cov.value().count(frame_name)) {
        Eigen::Matrix<double, 6, 6> contact_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
        contact_pose_cov.block<3, 3>(0, 0) = base_state_.contacts_position_cov.at(frame_name);
        contact_pose_cov.block<3, 3>(3, 3) =
            base_state_.contacts_orientation_cov.value().at(frame_name);
        return contact_pose_cov;
    } else {
        return std::nullopt;
    }
}

const Eigen::Vector3d& State::getCoMPosition() const {
    return centroidal_state_.com_position;
}

const Eigen::Vector3d& State::getCoMLinearVelocity() const {
    return centroidal_state_.com_linear_velocity;
}

const Eigen::Vector3d& State::getCoMExternalForces() const {
    return centroidal_state_.external_forces;
}

const Eigen::Vector3d& State::getCoMAngularMomentum() const {
    return centroidal_state_.angular_momentum;
}

const Eigen::Vector3d& State::getCoMAngularMomentumRate() const {
    return centroidal_state_.angular_momentum_derivative;
}

const Eigen::Vector3d& State::getCoMLinearAcceleration() const {
    return centroidal_state_.com_linear_acceleration;
}

const Eigen::Matrix3d& State::getCoMPositionCov() const {
    return centroidal_state_.com_position_cov;
}

const Eigen::Matrix3d& State::getCoMLinearVelocityCov() const {
    return centroidal_state_.com_linear_velocity_cov;
}

const Eigen::Matrix3d& State::getCoMExternalForcesCov() const {
    return centroidal_state_.external_forces_cov;
}

const Eigen::Vector3d& State::getCOPPosition() const {
    return centroidal_state_.cop_position;
}

bool State::isPointFeet() const {
    return point_feet_;
}

double State::getMass() const {
    return mass_;
}

int State::getNumLegEE() const {
    return num_leg_ee_;
}

void State::setBaseState(const BaseState& base_state) {
    base_state_ = base_state;
}

void State::setBaseStatePose(const Eigen::Vector3d& position,
                             const Eigen::Quaterniond& orientation) {
    base_state_.base_position = position;
    base_state_.base_orientation = orientation;
}

void State::setBaseStateVelocity(const Eigen::Vector3d& linear_velocity){
    base_state_.base_linear_velocity = linear_velocity;
}

void State::setContactState(const ContactState& contact_state) {
    contact_state_ = contact_state;
}

void State::setCentroidalState(const CentroidalState& centroidal_state) {
    centroidal_state_ = centroidal_state;
}

void State::setJointState(const JointState& joint_state) {
    joint_state_ = joint_state;
}

bool State::isValid() const {
    return is_valid_;
}

bool State::isInitialized() const {
    return is_initialized_;
}

void State::setValid(bool valid) {
    is_valid_ = valid;
}

void State::setInitialized(bool initialized) {
    is_initialized_ = initialized;
}

BaseState State::getBaseState() const {
    return base_state_;
}

ContactState State::getContactState() const {
    return contact_state_;
}

CentroidalState State::getCentroidalState() const {
    return centroidal_state_;
}

JointState State::getJointState() const {
    return joint_state_;
}

const std::string& State::getBaseFrame() const {
    return base_frame_;
}

const std::map<std::string, double>& State::getJointPositions() const {
    return joint_state_.joints_position;
}

const std::map<std::string, double>& State::getJointVelocities() const {
    return joint_state_.joints_velocity;
}

}  // namespace serow
