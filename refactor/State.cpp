/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "State.hpp"

Eigen::Isometry3d State::getBasePose() const {
    return base_pose_;
}

Eigen::Vector3d State::getBasePosition() const {
    return base_position_;
}

Eigen::Quaterniond State::getBaseOrientation() const {
    return base_orientation_;
}

Eigen::Vector3d State::getBaseLinearVelocity() const {
    return base_linear_velocity_;
}

Eigen::Vector3d State::getBaseAngularVelocity() const {
    return base_angular_velocity_;
}

Eigen::Vector3d State::getImuLinearAccelarationBias() const {
    return imu_linear_acceleration_bias_;
}

Eigen::Vector3d State::getImuAngularVelocityBias() const {
    return imu_angular_velocity_bias_;
}

std::optional<Eigen::Isometry3d> State::getFootPose(const std::string &frame_name) const {
    if (foot_pose_.count(frame_name))
        return foot_pose_.at(frame_name);
    else
        return std::nullopt;
}

std::unordered_set<std::string> State::getFootFrames() const {
    return foot_frames_;
}

std::optional<bool> State::getFootContactStatus(const std::string& frame_name) const {
    if (foot_contact_.count(frame_name))
        return foot_contact_.at(frame_name);
    else
        return std::nullopt; 
}

Eigen::Matrix<double, 6, 6> State::getBasePoseCov() const {
    return base_pose_cov_;
}

Eigen::Matrix3d State::getBasePositionCov() const {
    return base_position_cov_;
}

Eigen::Matrix3d State::getBaseOrientationCov() const {
    return base_orientation_cov_;
}

Eigen::Matrix3d State::getBaseLinearVelocityCov() const {
    return base_linear_velocity_cov_;
}

Eigen::Matrix3d State::getBaseAngularVelocityCov() const {
    return base_angular_velocity_cov_;
}

Eigen::Matrix3d State::getImuLinearAccelerationBiasCov() const {
    return imu_linear_acceleration_bias_cov_;
}

Eigen::Matrix3d State::getImuAngularVelocityBiasCov() const {
    return imu_angular_velocity_bias_cov_;
}

std::optional<Eigen::Matrix<double, 6, 6>>
State::getFootPoseCov(const std::string &frame_name) const {
    if (foot_pose_cov_.count(frame_name))
        return foot_pose_cov_.at(frame_name);
    else
        return std::nullopt;
}

void update(State state) {
    // TODO (mrsp) fill in
}
