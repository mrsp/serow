/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "State.hpp"

State::State(std::unordered_set<std::string> contacts_frame, bool point_feet) {
    contacts_frame_ = std::move(contacts_frame);
    num_leg_ee_ = contacts_frame_.size();
    point_feet_ = point_feet;
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

const Eigen::Vector3d& State::getImuLinearAccelarationBias() const {
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

std::optional<Eigen::Isometry3d> State::getContactPose(const std::string &frame_name) const {
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

std::optional<bool> State::getContactStatus(const std::string &frame_name) const {
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

const Eigen::Matrix3d& State::getBaseAngularVelocityCov() const { return base_angular_velocity_cov_; }

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

void update(State state) {
    // TODO (mrsp) fill in
}
