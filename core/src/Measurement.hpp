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
#include <optional>
#include <string>

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
    std::map<std::string, bool> contacts_status;
    std::map<std::string, double> contacts_probability;
    std::map<std::string, Eigen::Vector3d> contacts_position;
    std::map<std::string, Eigen::Matrix3d> contacts_position_noise;
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation;
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_noise;
    Eigen::Vector3d com_angular_momentum_derivative{Eigen::Vector3d::Zero()};
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};
    Eigen::Vector3d com_linear_acceleration{Eigen::Vector3d::Zero()};
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

struct TerrainMeasurement {
    double timestamp{};
    double height{};
    double height_cov{1.0};
    TerrainMeasurement(double timestamp, double height, double height_cov)
        :timestamp(timestamp), height(height), height_cov(height_cov){}
};

using ContactMeasurement = double;

}  // namespace serow
