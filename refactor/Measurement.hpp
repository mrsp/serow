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

struct JointMeasurement {
    double timestamp{};
    double position{};
    std::optional<double> velocity{};
};

struct ImuMeasurement {
    double timestamp{};
    Eigen::Vector3d linear_acceleration{};
    Eigen::Vector3d angular_velocity{};
    Eigen::Vector3d angular_acceleration{};
    Eigen::Matrix3d angular_velocity_cov{};
    Eigen::Matrix3d linear_acceleration_cov{};
    Eigen::Matrix3d angular_velocity_bias_cov{};
    Eigen::Matrix3d linear_acceleration_bias_cov{};
};

struct ForceTorqueMeasurement {
    double timestamp{};
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};
    std::optional<Eigen::Vector3d> torque;
};

struct GroundReactionForceMeasurement {
    double timestamp{};
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};
};

struct KinematicMeasurement {
    double timestamp{};
    std::unordered_map<std::string, bool> contacts_status;
    std::unordered_map<std::string, double> contacts_probability;
    std::unordered_map<std::string, Eigen::Vector3d> contacts_position;
    std::unordered_map<std::string, Eigen::Matrix3d> contacts_position_noise;
    std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientation;
    std::optional<std::unordered_map<std::string, Eigen::Matrix3d>> contacts_orientation_noise;
    std::optional<Eigen::Vector3d> com_angular_momentum;
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d position_slip_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d orientation_slip_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d position_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d orientation_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d com_position_process_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d com_linear_velocity_process_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d external_forces_process_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d com_position_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d com_linear_acceleration_cov{Eigen::Matrix3d::Identity()};
};

struct OdometryMeasurement {
    double timestamp{};
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()};
    Eigen::Matrix3d base_position_cov{Eigen::Matrix3d::Identity()};
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Identity()};
};

}  // namespace serow
