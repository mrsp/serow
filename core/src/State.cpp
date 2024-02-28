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

State::State(std::unordered_set<std::string> contacts_frame, bool point_feet) {
    contacts_frame_ = std::move(contacts_frame);
    num_leg_ee_ = contacts_frame_.size();
    point_feet_ = point_feet;
    is_valid_ = false;

    std::unordered_map<std::string, Eigen::Quaterniond> contacts_orientation;
    std::unordered_map<std::string, Eigen::Matrix3d> contacts_orientation_cov;
    for (const auto& cf : contacts_frame_) {
        contact_state_.contacts_status[cf] = false;
        contact_state_.contacts_probability[cf] = 0.0;
        base_state_.contacts_position[cf] = Eigen::Vector3d::Zero();
        base_state_.contacts_position_cov[cf] = Eigen::Matrix3d::Identity();
        if (!isPointFeet()) {
            contacts_orientation[cf] = Eigen::Quaterniond::Identity();
            contacts_orientation_cov[cf] = Eigen::Matrix3d::Identity();
        }
    }
    if (!isPointFeet()) {
        base_state_.contacts_orientation = std::move(contacts_orientation);
        base_state_.contacts_orientation_cov = std::move(contacts_orientation_cov);
    }
}

State::State(const State& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    this->point_feet_ = other.point_feet_;
    this->num_leg_ee_ = other.num_leg_ee_;
    this->contacts_frame_ = other.contacts_frame_;
    this->is_valid_ = other.is_valid_;

    // Base state
    this->base_state_.timestamp = other.base_state_.timestamp;
    this->base_state_.base_position = other.base_state_.base_position;
    this->base_state_.base_orientation = other.base_state_.base_orientation;
    this->base_state_.base_linear_velocity = other.base_state_.base_linear_velocity;
    this->base_state_.base_angular_velocity = other.base_state_.base_angular_velocity;
    this->base_state_.base_linear_acceleration = other.base_state_.base_linear_acceleration;
    this->base_state_.base_angular_acceleration = other.base_state_.base_angular_acceleration;
    this->base_state_.imu_linear_acceleration_bias = other.base_state_.imu_linear_acceleration_bias;
    this->base_state_.imu_angular_velocity_bias = other.base_state_.imu_angular_velocity_bias;
    this->base_state_.contacts_position = other.base_state_.contacts_position;
    if (other.base_state_.contacts_orientation.has_value()) {
        this->base_state_.contacts_orientation = other.base_state_.contacts_orientation.value();
    }

    this->base_state_.base_position_cov = other.base_state_.base_position_cov;
    this->base_state_.base_orientation_cov = other.base_state_.base_orientation_cov;
    this->base_state_.base_linear_velocity_cov = other.base_state_.base_linear_velocity_cov;
    this->base_state_.base_angular_velocity_cov = other.base_state_.base_angular_velocity_cov;
    this->base_state_.imu_linear_acceleration_bias_cov =
        other.base_state_.imu_linear_acceleration_bias_cov;
    this->base_state_.imu_angular_velocity_bias_cov =
        other.base_state_.imu_angular_velocity_bias_cov;
    this->base_state_.contacts_position_cov = other.base_state_.contacts_position_cov;
    if (other.base_state_.contacts_orientation_cov.has_value()) {
        this->base_state_.contacts_orientation_cov =
            other.base_state_.contacts_orientation_cov.value();
    }

    // Contact state
    this->contact_state_.timestamp = other.contact_state_.timestamp;
    this->contact_state_.contacts_status = other.contact_state_.contacts_status;
    this->contact_state_.contacts_probability = other.contact_state_.contacts_probability;
    this->contact_state_.contacts_force = other.contact_state_.contacts_force;
    if (other.contact_state_.contacts_torque.has_value()) {
        this->contact_state_.contacts_torque = other.contact_state_.contacts_torque;
    }

    // Joint state
    this->joint_state_.timestamp = other.joint_state_.timestamp;
    this->joint_state_.joints_position = other.joint_state_.joints_position;
    this->joint_state_.joints_velocity = other.joint_state_.joints_velocity;

    // Centroidal state
    this->centroidal_state_.timestamp = other.centroidal_state_.timestamp;
    this->centroidal_state_.com_position = other.centroidal_state_.com_position;
    this->centroidal_state_.com_linear_velocity = other.centroidal_state_.com_linear_velocity;
    this->centroidal_state_.external_forces = other.centroidal_state_.external_forces;
    this->centroidal_state_.cop_position = other.centroidal_state_.cop_position;
    this->centroidal_state_.com_linear_acceleration =
        other.centroidal_state_.com_linear_acceleration;
    this->centroidal_state_.angular_momentum = other.centroidal_state_.angular_momentum;
    this->centroidal_state_.angular_momentum_derivative =
        other.centroidal_state_.angular_momentum_derivative;

    this->centroidal_state_.com_position_cov = other.centroidal_state_.com_position_cov;
    this->centroidal_state_.com_linear_velocity_cov =
        other.centroidal_state_.com_linear_velocity_cov;
    this->centroidal_state_.external_forces_cov = other.centroidal_state_.external_forces_cov;
}

State::State(State&& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    this->point_feet_ = std::move(other.point_feet_);
    this->num_leg_ee_ = std::move(other.num_leg_ee_);
    this->contacts_frame_ = std::move(other.contacts_frame_);
    this->is_valid_ = std::move(other.is_valid_);

    // Base state
    this->base_state_.timestamp = std::move(other.base_state_.timestamp);
    this->base_state_.base_position = std::move(other.base_state_.base_position);
    this->base_state_.base_orientation = std::move(other.base_state_.base_orientation);
    this->base_state_.base_linear_velocity = std::move(other.base_state_.base_linear_velocity);
    this->base_state_.base_angular_velocity = std::move(other.base_state_.base_angular_velocity);
    this->base_state_.base_linear_acceleration =
        std::move(other.base_state_.base_linear_acceleration);
    this->base_state_.base_angular_acceleration =
        std::move(other.base_state_.base_angular_acceleration);
    this->base_state_.imu_linear_acceleration_bias =
        std::move(other.base_state_.imu_linear_acceleration_bias);
    this->base_state_.imu_angular_velocity_bias =
        std::move(other.base_state_.imu_angular_velocity_bias);
    this->base_state_.contacts_position = std::move(other.base_state_.contacts_position);
    if (other.base_state_.contacts_orientation.has_value()) {
        this->base_state_.contacts_orientation =
            std::move(other.base_state_.contacts_orientation.value());
    }

    this->base_state_.base_position_cov = std::move(other.base_state_.base_position_cov);
    this->base_state_.base_orientation_cov = std::move(other.base_state_.base_orientation_cov);
    this->base_state_.base_linear_velocity_cov =
        std::move(other.base_state_.base_linear_velocity_cov);
    this->base_state_.base_angular_velocity_cov =
        std::move(other.base_state_.base_angular_velocity_cov);
    this->base_state_.imu_linear_acceleration_bias_cov =
        std::move(other.base_state_.imu_linear_acceleration_bias_cov);
    this->base_state_.imu_angular_velocity_bias_cov =
        std::move(other.base_state_.imu_angular_velocity_bias_cov);
    this->base_state_.contacts_position_cov = std::move(other.base_state_.contacts_position_cov);
    if (other.base_state_.contacts_orientation_cov.has_value()) {
        this->base_state_.contacts_orientation_cov =
            std::move(other.base_state_.contacts_orientation_cov.value());
    }

    // Contact state
    this->contact_state_.timestamp = std::move(other.contact_state_.timestamp);
    this->contact_state_.contacts_status = std::move(other.contact_state_.contacts_status);
    this->contact_state_.contacts_probability =
        std::move(other.contact_state_.contacts_probability);
    this->contact_state_.contacts_force = std::move(other.contact_state_.contacts_force);
    if (other.contact_state_.contacts_torque.has_value()) {
        this->contact_state_.contacts_torque = std::move(other.contact_state_.contacts_torque);
    }

    // Joint state
    this->joint_state_.timestamp = std::move(other.joint_state_.timestamp);
    this->joint_state_.joints_position = std::move(other.joint_state_.joints_position);
    this->joint_state_.joints_velocity = std::move(other.joint_state_.joints_velocity);

    // Centroidal state
    this->centroidal_state_.timestamp = std::move(other.centroidal_state_.timestamp);
    this->centroidal_state_.com_position = std::move(other.centroidal_state_.com_position);
    this->centroidal_state_.com_linear_velocity =
        std::move(other.centroidal_state_.com_linear_velocity);
    this->centroidal_state_.external_forces = std::move(other.centroidal_state_.external_forces);
    this->centroidal_state_.cop_position = std::move(other.centroidal_state_.cop_position);
    this->centroidal_state_.com_linear_acceleration =
        std::move(other.centroidal_state_.com_linear_acceleration);
    this->centroidal_state_.angular_momentum = std::move(other.centroidal_state_.angular_momentum);
    this->centroidal_state_.angular_momentum_derivative =
        std::move(other.centroidal_state_.angular_momentum_derivative);

    this->centroidal_state_.com_position_cov = std::move(other.centroidal_state_.com_position_cov);
    this->centroidal_state_.com_linear_velocity_cov =
        std::move(other.centroidal_state_.com_linear_velocity_cov);
    this->centroidal_state_.external_forces_cov =
        std::move(other.centroidal_state_.external_forces_cov);
}

State State::operator=(const State& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    this->point_feet_ = other.point_feet_;
    this->num_leg_ee_ = other.num_leg_ee_;
    this->contacts_frame_ = other.contacts_frame_;
    this->is_valid_ = other.is_valid_;

    // Base state
    this->base_state_.timestamp = other.base_state_.timestamp;
    this->base_state_.base_position = other.base_state_.base_position;
    this->base_state_.base_orientation = other.base_state_.base_orientation;
    this->base_state_.base_linear_velocity = other.base_state_.base_linear_velocity;
    this->base_state_.base_angular_velocity = other.base_state_.base_angular_velocity;
    this->base_state_.base_linear_acceleration = other.base_state_.base_linear_acceleration;
    this->base_state_.base_angular_acceleration = other.base_state_.base_angular_acceleration;
    this->base_state_.imu_linear_acceleration_bias = other.base_state_.imu_linear_acceleration_bias;
    this->base_state_.imu_angular_velocity_bias = other.base_state_.imu_angular_velocity_bias;
    this->base_state_.contacts_position = other.base_state_.contacts_position;
    if (other.base_state_.contacts_orientation.has_value()) {
        this->base_state_.contacts_orientation = other.base_state_.contacts_orientation.value();
    }

    this->base_state_.base_position_cov = other.base_state_.base_position_cov;
    this->base_state_.base_orientation_cov = other.base_state_.base_orientation_cov;
    this->base_state_.base_linear_velocity_cov = other.base_state_.base_linear_velocity_cov;
    this->base_state_.base_angular_velocity_cov = other.base_state_.base_angular_velocity_cov;
    this->base_state_.imu_linear_acceleration_bias_cov =
        other.base_state_.imu_linear_acceleration_bias_cov;
    this->base_state_.imu_angular_velocity_bias_cov =
        other.base_state_.imu_angular_velocity_bias_cov;
    this->base_state_.contacts_position_cov = other.base_state_.contacts_position_cov;
    if (other.base_state_.contacts_orientation_cov.has_value()) {
        this->base_state_.contacts_orientation_cov =
            other.base_state_.contacts_orientation_cov.value();
    }

    // Contact state
    this->contact_state_.timestamp = other.contact_state_.timestamp;
    this->contact_state_.contacts_status = other.contact_state_.contacts_status;
    this->contact_state_.contacts_probability = other.contact_state_.contacts_probability;
    this->contact_state_.contacts_force = other.contact_state_.contacts_force;
    if (other.contact_state_.contacts_torque.has_value()) {
        this->contact_state_.contacts_torque = other.contact_state_.contacts_torque;
    }

    // Joint state
    this->joint_state_.timestamp = other.joint_state_.timestamp;
    this->joint_state_.joints_position = other.joint_state_.joints_position;
    this->joint_state_.joints_velocity = other.joint_state_.joints_velocity;

    // Centroidal state
    this->centroidal_state_.timestamp = other.centroidal_state_.timestamp;
    this->centroidal_state_.com_position = other.centroidal_state_.com_position;
    this->centroidal_state_.com_linear_velocity = other.centroidal_state_.com_linear_velocity;
    this->centroidal_state_.external_forces = other.centroidal_state_.external_forces;
    this->centroidal_state_.cop_position = other.centroidal_state_.cop_position;
    this->centroidal_state_.com_linear_acceleration =
        other.centroidal_state_.com_linear_acceleration;
    this->centroidal_state_.angular_momentum = other.centroidal_state_.angular_momentum;
    this->centroidal_state_.angular_momentum_derivative =
        other.centroidal_state_.angular_momentum_derivative;

    this->centroidal_state_.com_position_cov = other.centroidal_state_.com_position_cov;
    this->centroidal_state_.com_linear_velocity_cov =
        other.centroidal_state_.com_linear_velocity_cov;
    this->centroidal_state_.external_forces_cov = other.centroidal_state_.external_forces_cov;

    return *this;
}

State& State::operator=(State&& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    if (this != &other) {
        this->point_feet_ = std::move(other.point_feet_);
        this->num_leg_ee_ = std::move(other.num_leg_ee_);
        this->contacts_frame_ = std::move(other.contacts_frame_);
        this->is_valid_ = std::move(other.is_valid_);
        
        // Base state
        this->base_state_.timestamp = std::move(other.base_state_.timestamp);
        this->base_state_.base_position = std::move(other.base_state_.base_position);
        this->base_state_.base_orientation = std::move(other.base_state_.base_orientation);
        this->base_state_.base_linear_velocity = std::move(other.base_state_.base_linear_velocity);
        this->base_state_.base_angular_velocity =
            std::move(other.base_state_.base_angular_velocity);
        this->base_state_.base_linear_acceleration =
            std::move(other.base_state_.base_linear_acceleration);
        this->base_state_.base_angular_acceleration =
            std::move(other.base_state_.base_angular_acceleration);
        this->base_state_.imu_linear_acceleration_bias =
            std::move(other.base_state_.imu_linear_acceleration_bias);
        this->base_state_.imu_angular_velocity_bias =
            std::move(other.base_state_.imu_angular_velocity_bias);
        this->base_state_.contacts_position = std::move(other.base_state_.contacts_position);
        if (other.base_state_.contacts_orientation.has_value()) {
            this->base_state_.contacts_orientation =
                std::move(other.base_state_.contacts_orientation.value());
        }

        this->base_state_.base_position_cov = std::move(other.base_state_.base_position_cov);
        this->base_state_.base_orientation_cov = std::move(other.base_state_.base_orientation_cov);
        this->base_state_.base_linear_velocity_cov =
            std::move(other.base_state_.base_linear_velocity_cov);
        this->base_state_.base_angular_velocity_cov =
            std::move(other.base_state_.base_angular_velocity_cov);
        this->base_state_.imu_linear_acceleration_bias_cov =
            std::move(other.base_state_.imu_linear_acceleration_bias_cov);
        this->base_state_.imu_angular_velocity_bias_cov =
            std::move(other.base_state_.imu_angular_velocity_bias_cov);
        this->base_state_.contacts_position_cov =
            std::move(other.base_state_.contacts_position_cov);
        if (other.base_state_.contacts_orientation_cov.has_value()) {
            this->base_state_.contacts_orientation_cov =
                std::move(other.base_state_.contacts_orientation_cov.value());
        }

        // Contact state
        this->contact_state_.timestamp = std::move(other.contact_state_.timestamp);
        this->contact_state_.contacts_status = std::move(other.contact_state_.contacts_status);
        this->contact_state_.contacts_probability =
            std::move(other.contact_state_.contacts_probability);
        this->contact_state_.contacts_force = std::move(other.contact_state_.contacts_force);
        if (other.contact_state_.contacts_torque.has_value()) {
            this->contact_state_.contacts_torque = std::move(other.contact_state_.contacts_torque);
        }

        // Joint state
        this->joint_state_.timestamp = std::move(other.joint_state_.timestamp);
        this->joint_state_.joints_position = std::move(other.joint_state_.joints_position);
        this->joint_state_.joints_velocity = std::move(other.joint_state_.joints_velocity);

        // Centroidal state
        this->centroidal_state_.timestamp = std::move(other.centroidal_state_.timestamp);
        this->centroidal_state_.com_position = std::move(other.centroidal_state_.com_position);
        this->centroidal_state_.com_linear_velocity =
            std::move(other.centroidal_state_.com_linear_velocity);
        this->centroidal_state_.external_forces =
            std::move(other.centroidal_state_.external_forces);
        this->centroidal_state_.cop_position = std::move(other.centroidal_state_.cop_position);
        this->centroidal_state_.com_linear_acceleration =
            std::move(other.centroidal_state_.com_linear_acceleration);
        this->centroidal_state_.angular_momentum =
            std::move(other.centroidal_state_.angular_momentum);
        this->centroidal_state_.angular_momentum_derivative =
            std::move(other.centroidal_state_.angular_momentum_derivative);

        this->centroidal_state_.com_position_cov =
            std::move(other.centroidal_state_.com_position_cov);
        this->centroidal_state_.com_linear_velocity_cov =
            std::move(other.centroidal_state_.com_linear_velocity_cov);
        this->centroidal_state_.external_forces_cov =
            std::move(other.centroidal_state_.external_forces_cov);
    }
    return *this;
}

Eigen::Isometry3d State::getBasePose() const {
    Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
    base_pose.linear() = base_state_.base_orientation.toRotationMatrix();
    base_pose.translation() = base_state_.base_position;
    return base_pose;
}

const Eigen::Vector3d& State::getBasePosition() const { return base_state_.base_position; }

const Eigen::Quaterniond& State::getBaseOrientation() const { return base_state_.base_orientation; }

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

const std::unordered_set<std::string>& State::getContactsFrame() const { return contacts_frame_; }

std::optional<bool> State::getContactStatus(const std::string& frame_name) const {
    if (contact_state_.contacts_status.count(frame_name))
        return contact_state_.contacts_status.at(frame_name);
    else
        return std::nullopt;
}

Eigen::Matrix<double, 6, 6> State::getBasePoseCov() const {
    Eigen::Matrix<double, 6, 6> base_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
    base_pose_cov.block<3, 3>(0, 0) = base_state_.base_position_cov;
    base_pose_cov.block<3, 3>(3, 3) = base_state_.base_orientation_cov;
    return base_pose_cov;
}

const Eigen::Matrix3d& State::getBasePositionCov() const { return base_state_.base_position_cov; }

const Eigen::Matrix3d& State::getBaseOrientationCov() const {
    return base_state_.base_orientation_cov;
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

const Eigen::Vector3d& State::getCoMPosition() const { return centroidal_state_.com_position; }

const Eigen::Vector3d& State::getCoMLinearVelocity() const {
    return centroidal_state_.com_linear_velocity;
}

const Eigen::Vector3d& State::getCoMExternalForces() const {
    return centroidal_state_.external_forces;
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

bool State::isPointFeet() const { return point_feet_; }

}  // namespace serow
