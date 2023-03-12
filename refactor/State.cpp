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

Eigen::Vector3d State::getImuAccelarationBias() const {
    return imu_accelaration_bias_;
}

Eigen::Vector3d State::getImuGyroRateBias() const {
    return imu_gyro_rate_bias_;
}

std::optional<Eigen::Isometry3d> State::getFootPose(const std::string &frame_name) const {
    if (foot_pose_.has_value() && foot_pose_.value().count(frame_name))
        return foot_pose_.value().at(frame_name);
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

Eigen::Matrix3d State::getImuAccelarationBiasCov() const {
    return imu_accelaration_bias_cov_;
}

Eigen::Matrix3d State::getImuGyroRateBiasCov() const {
    return imu_gyro_rate_bias_cov_;
}

std::optional<Eigen::Matrix<double, 6, 6>>
State::getFootPoseCov(const std::string &frame_name) const {
    if (foot_pose_cov_.has_value() && foot_pose_cov_.value().count(frame_name))
        return foot_pose_cov_.value().at(frame_name);
    else
        return std::nullopt;
}

void update(State state) {
    // TODO (mrsp) fill in
}
