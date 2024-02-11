/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "State.hpp"

namespace serow {

State::State(std::unordered_set<std::string> contacts_frame, bool point_feet) {
    contacts_frame_ = std::move(contacts_frame);
    num_leg_ee_ = contacts_frame_.size();
    point_feet_ = point_feet;

    std::unordered_map<std::string, Eigen::Quaterniond> contacts_orientation;
    std::unordered_map<std::string, Eigen::Matrix3d> contacts_orientation_cov;
    for (const auto& cf : contacts_frame_) {
        contacts_status_[cf] = false;
        contacts_probability_[cf] = 0.0;
        contacts_position_[cf] = Eigen::Vector3d::Zero();
        contacts_position_cov_[cf] = Eigen::Matrix3d::Identity();
        if (!isPointFeet()) {
            contacts_orientation[cf] = Eigen::Quaterniond::Identity();
            contacts_orientation_cov[cf] = Eigen::Matrix3d::Identity();
        }
    }
    if (!isPointFeet()) {
        contacts_orientation_ = contacts_orientation;
        contacts_orientation_cov_ = contacts_orientation_cov;
    }
}

State::State(const State& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    this->point_feet_ = other.point_feet_;
    this->num_leg_ee_ = other.num_leg_ee_;
    this->base_position_ = other.base_position_;
    this->base_orientation_ = other.base_orientation_;
    this->base_linear_velocity_ = other.base_linear_velocity_;
    this->base_angular_velocity_ = other.base_angular_velocity_;
    this->contacts_position_ = other.contacts_position_;
    if (other.contacts_orientation_.has_value()) {
        this->contacts_orientation_ = other.contacts_orientation_.value();
    }
    this->contacts_status_ = other.contacts_status_;
    this->contacts_probability_ = other.contacts_probability_;
    this->contacts_frame_ = other.contacts_frame_;
    this->imu_linear_acceleration_bias_ = other.imu_linear_acceleration_bias_;
    this->imu_angular_velocity_bias_ = other.imu_angular_velocity_bias_;
    this->com_position_ = other.com_position_;
    this->com_linear_velocity_ = other.com_linear_velocity_;
    this->external_forces_ = other.external_forces_;
    this->base_position_cov_ = other.base_position_cov_;
    this->base_orientation_cov_ = other.base_orientation_cov_;
    this->base_linear_velocity_cov_ = other.base_linear_velocity_cov_;
    this->base_angular_velocity_cov_ = other.base_angular_velocity_cov_;
    this->imu_linear_acceleration_bias_cov_ = other.imu_linear_acceleration_bias_cov_;
    this->imu_angular_velocity_bias_cov_ = other.imu_angular_velocity_bias_cov_;
    this->contacts_position_cov_ = other.contacts_position_cov_;
    if (other.contacts_orientation_cov_.has_value()) {
        this->contacts_orientation_cov_ = other.contacts_orientation_cov_.value();
    }
    this->contact_forces = other.contact_forces;
    if (other.contact_torques.has_value()) {
        this->contact_torques = other.contact_torques;
    }
}

State::State(State&& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    this->point_feet_ = std::move(other.point_feet_);
    this->num_leg_ee_ = std::move(other.num_leg_ee_);
    this->base_position_ = std::move(other.base_position_);
    this->base_orientation_ = std::move(other.base_orientation_);
    this->base_linear_velocity_ = std::move(other.base_linear_velocity_);
    this->base_angular_velocity_ = std::move(other.base_angular_velocity_);
    this->contacts_position_ = std::move(other.contacts_position_);
    if (other.contacts_orientation_.has_value()) {
        this->contacts_orientation_ = std::move(other.contacts_orientation_.value());
    }
    this->contacts_status_ = std::move(other.contacts_status_);
    this->contacts_probability_ = std::move(other.contacts_probability_);
    this->contacts_frame_ = std::move(other.contacts_frame_);
    this->imu_linear_acceleration_bias_ = std::move(other.imu_linear_acceleration_bias_);
    this->imu_angular_velocity_bias_ = std::move(other.imu_angular_velocity_bias_);
    this->com_position_ = std::move(other.com_position_);
    this->com_linear_velocity_ = std::move(other.com_linear_velocity_);
    this->external_forces_ = std::move(other.external_forces_);
    this->base_position_cov_ = std::move(other.base_position_cov_);
    this->base_orientation_cov_ = std::move(other.base_orientation_cov_);
    this->base_linear_velocity_cov_ = std::move(other.base_linear_velocity_cov_);
    this->base_angular_velocity_cov_ = std::move(other.base_angular_velocity_cov_);
    this->imu_linear_acceleration_bias_cov_ = std::move(other.imu_linear_acceleration_bias_cov_);
    this->imu_angular_velocity_bias_cov_ = std::move(other.imu_angular_velocity_bias_cov_);
    this->contacts_position_cov_ = std::move(other.contacts_position_cov_);
    if (other.contacts_orientation_cov_.has_value()) {
        this->contacts_orientation_cov_ = std::move(other.contacts_orientation_cov_.value());
    }
    this->contact_forces = std::move(other.contact_forces);
    if (other.contact_torques.has_value()) {
        this->contact_torques = std::move(other.contact_torques);
    }
}

State State::operator=(const State& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    this->point_feet_ = other.point_feet_;
    this->num_leg_ee_ = other.num_leg_ee_;
    this->base_position_ = other.base_position_;
    this->base_orientation_ = other.base_orientation_;
    this->base_linear_velocity_ = other.base_linear_velocity_;
    this->base_angular_velocity_ = other.base_angular_velocity_;
    this->contacts_position_ = other.contacts_position_;
    if (other.contacts_orientation_.has_value()) {
        this->contacts_orientation_ = other.contacts_orientation_.value();
    }
    this->contacts_status_ = other.contacts_status_;
    this->contacts_probability_ = other.contacts_probability_;
    this->contacts_frame_ = other.contacts_frame_;
    this->imu_linear_acceleration_bias_ = other.imu_linear_acceleration_bias_;
    this->imu_angular_velocity_bias_ = other.imu_angular_velocity_bias_;
    this->com_position_ = other.com_position_;
    this->com_linear_velocity_ = other.com_linear_velocity_;
    this->external_forces_ = other.external_forces_;
    this->base_position_cov_ = other.base_position_cov_;
    this->base_orientation_cov_ = other.base_orientation_cov_;
    this->base_linear_velocity_cov_ = other.base_linear_velocity_cov_;
    this->base_angular_velocity_cov_ = other.base_angular_velocity_cov_;
    this->imu_linear_acceleration_bias_cov_ = other.imu_linear_acceleration_bias_cov_;
    this->imu_angular_velocity_bias_cov_ = other.imu_angular_velocity_bias_cov_;
    this->contacts_position_cov_ = other.contacts_position_cov_;
    if (other.contacts_orientation_cov_.has_value()) {
        this->contacts_orientation_cov_ = other.contacts_orientation_cov_.value();
    }
    this->contact_forces = other.contact_forces;
    if (other.contact_torques.has_value()) {
        this->contact_torques = other.contact_torques;
    }
    return *this;
}

State& State::operator=(State&& other) {
    const std::lock_guard<std::mutex> lock(this->mutex_);
    if (this != &other) {
        this->point_feet_ = std::move(other.point_feet_);
        this->num_leg_ee_ = std::move(other.num_leg_ee_);
        this->base_position_ = std::move(other.base_position_);
        this->base_orientation_ = std::move(other.base_orientation_);
        this->base_linear_velocity_ = std::move(other.base_linear_velocity_);
        this->base_angular_velocity_ = std::move(other.base_angular_velocity_);
        this->contacts_position_ = std::move(other.contacts_position_);
        if (other.contacts_orientation_.has_value()) {
            this->contacts_orientation_ = std::move(other.contacts_orientation_.value());
        }
        this->contacts_status_ = std::move(other.contacts_status_);
        this->contacts_probability_ = std::move(other.contacts_probability_);
        this->contacts_frame_ = std::move(other.contacts_frame_);
        this->imu_linear_acceleration_bias_ = std::move(other.imu_linear_acceleration_bias_);
        this->imu_angular_velocity_bias_ = std::move(other.imu_angular_velocity_bias_);
        this->com_position_ = std::move(other.com_position_);
        this->com_linear_velocity_ = std::move(other.com_linear_velocity_);
        this->external_forces_ = std::move(other.external_forces_);
        this->base_position_cov_ = std::move(other.base_position_cov_);
        this->base_orientation_cov_ = std::move(other.base_orientation_cov_);
        this->base_linear_velocity_cov_ = std::move(other.base_linear_velocity_cov_);
        this->base_angular_velocity_cov_ = std::move(other.base_angular_velocity_cov_);
        this->imu_linear_acceleration_bias_cov_ =
            std::move(other.imu_linear_acceleration_bias_cov_);
        this->imu_angular_velocity_bias_cov_ = std::move(other.imu_angular_velocity_bias_cov_);
        this->contacts_position_cov_ = std::move(other.contacts_position_cov_);
        if (other.contacts_orientation_cov_.has_value()) {
            this->contacts_orientation_cov_ = std::move(other.contacts_orientation_cov_.value());
        }
        this->contact_forces = std::move(other.contact_forces);
        if (other.contact_torques.has_value()) {
            this->contact_torques = std::move(other.contact_torques);
        }
    }
    return *this;
}

Eigen::Isometry3d State::getBasePose() const {
    Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();
    base_pose.linear() = base_orientation_.toRotationMatrix();
    base_pose.translation() = base_position_;
    return base_pose;
}

const Eigen::Vector3d& State::getBasePosition() const { return base_position_; }

const Eigen::Quaterniond& State::getBaseOrientation() const { return base_orientation_; }

const Eigen::Vector3d& State::getBaseLinearVelocity() const { return base_linear_velocity_; }

const Eigen::Vector3d& State::getBaseAngularVelocity() const { return base_angular_velocity_; }

const Eigen::Vector3d& State::getImuLinearAccelerationBias() const {
    return imu_linear_acceleration_bias_;
}

const Eigen::Vector3d& State::getImuAngularVelocityBias() const {
    return imu_angular_velocity_bias_;
}

std::optional<Eigen::Vector3d> State::getContactPosition(const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact position
    // available
    if (contacts_status_.count(frame_name) && contacts_status_.at(frame_name) &&
        contacts_position_.count(frame_name))
        return contacts_position_.at(frame_name);
    else
        return std::nullopt;
}

std::optional<Eigen::Quaterniond> State::getContactOrientation(
    const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact orientation
    // available
    if (contacts_status_.count(frame_name) && contacts_status_.at(frame_name) &&
        contacts_orientation_.has_value() && contacts_orientation_.value().count(frame_name))
        return contacts_orientation_.value().at(frame_name);
    else
        return std::nullopt;
}

std::optional<Eigen::Isometry3d> State::getContactPose(const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact orientation
    // available
    if (contacts_status_.count(frame_name) && contacts_status_.at(frame_name) &&
        contacts_position_.count(frame_name) && contacts_orientation_.has_value() &&
        contacts_orientation_.value().count(frame_name)) {
        Eigen::Isometry3d contact_pose = Eigen::Isometry3d::Identity();
        contact_pose.linear() = contacts_orientation_.value().at(frame_name).toRotationMatrix();
        contact_pose.translation() = contacts_position_.at(frame_name);
        return contact_pose;
    } else {
        return std::nullopt;
    }
}

const std::unordered_set<std::string>& State::getContactsFrame() const { return contacts_frame_; }

std::optional<bool> State::getContactStatus(const std::string& frame_name) const {
    if (contacts_status_.count(frame_name))
        return contacts_status_.at(frame_name);
    else
        return std::nullopt;
}

Eigen::Matrix<double, 6, 6> State::getBasePoseCov() const {
    Eigen::Matrix<double, 6, 6> base_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
    base_pose_cov.block<3, 3>(0, 0) = base_position_cov_;
    base_pose_cov.block<3, 3>(3, 3) = base_orientation_cov_;
    return base_pose_cov;
}

const Eigen::Matrix3d& State::getBasePositionCov() const { return base_position_cov_; }

const Eigen::Matrix3d& State::getBaseOrientationCov() const { return base_orientation_cov_; }

const Eigen::Matrix3d& State::getBaseLinearVelocityCov() const { return base_linear_velocity_cov_; }

const Eigen::Matrix3d& State::getBaseAngularVelocityCov() const {
    return base_angular_velocity_cov_;
}

const Eigen::Matrix3d& State::getImuLinearAccelerationBiasCov() const {
    return imu_linear_acceleration_bias_cov_;
}

const Eigen::Matrix3d& State::getImuAngularVelocityBiasCov() const {
    return imu_angular_velocity_bias_cov_;
}

std::optional<Eigen::Matrix3d> State::getContactPositionCov(const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact position
    // covariance available
    if (contacts_status_.count(frame_name) && contacts_status_.at(frame_name) &&
        contacts_position_cov_.count(frame_name))
        return contacts_position_cov_.at(frame_name);
    else
        return std::nullopt;
}

std::optional<Eigen::Matrix3d> State::getContactOrientationCov(
    const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact orientation
    // covariance available
    if (contacts_status_.count(frame_name) && contacts_status_.at(frame_name) &&
        contacts_orientation_cov_.has_value() && contacts_orientation_.value().count(frame_name))
        return contacts_orientation_cov_.value().at(frame_name);
    else
        return std::nullopt;
}

std::optional<Eigen::Matrix<double, 6, 6>> State::getContactPoseCov(
    const std::string& frame_name) const {
    // If the end-effector is in contact with the environment and we have a contact pose
    // covariance available
    if (contacts_status_.count(frame_name) && contacts_status_.at(frame_name) &&
        contacts_position_cov_.count(frame_name) && contacts_orientation_cov_.has_value() &&
        contacts_orientation_cov_.value().count(frame_name)) {
        Eigen::Matrix<double, 6, 6> contact_pose_cov = Eigen::Matrix<double, 6, 6>::Identity();
        contact_pose_cov.block<3, 3>(0, 0) = contacts_position_cov_.at(frame_name);
        contact_pose_cov.block<3, 3>(3, 3) = contacts_orientation_cov_.value().at(frame_name);
        return contact_pose_cov;
    } else {
        return std::nullopt;
    }
}

const Eigen::Vector3d& State::getCoMPosition() const { return com_position_; }

const Eigen::Vector3d& State::getCoMLinearVelocity() const { return com_linear_velocity_; }

const Eigen::Vector3d& State::getCoMExternalForces() const { return external_forces_; }

const Eigen::Matrix3d& State::getCoMPositionCov() const { return com_position_cov_; }

const Eigen::Matrix3d& State::getCoMLinearVelocityCov() const { return com_linear_velocity_cov_; }

const Eigen::Matrix3d& State::getCoMExternalForcesCov() const { return external_forces_cov_; }

bool State::isPointFeet() const { return point_feet_; }

}  // namespace serow
