/**
 * Copyright (C) 2025 Stylianos Piperakis, Ownage Dynamics L.P.
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
 * @file DebugLogger.hpp
 * @brief Defines the DebugLogger class for logging robot state data in MCAP format.
 *
 * The DebugLogger provides functionality to log various robot states and measurements
 * in a binary MCAP format, including:
 * - Base state (position, orientation, velocities, etc.)
 * - Joint measurements
 * - Contact states
 * - Force-torque measurements
 * - IMU measurements
 * - Centroidal state
 *
 * The logger uses the PIMPL pattern to hide implementation details and minimize
 * compilation dependencies.
 **/
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <fstream>
#include <map>
#include <mcap/mcap.hpp>
#include <mcap/writer.hpp>
#include <memory>
#include <string>

#include "Measurement.hpp"
#include "State.hpp"

namespace serow {

class DebugLogger {
public:
    DebugLogger(const std::string& log_file_path = "/tmp/serow_log.mcap");
    ~DebugLogger();

    // Delete copy operations
    DebugLogger(const DebugLogger&) = delete;
    DebugLogger& operator=(const DebugLogger&) = delete;

    // Allow move operations
    DebugLogger(DebugLogger&&) noexcept = default;
    DebugLogger& operator=(DebugLogger&&) noexcept = default;

    // void log(const BaseState& base_state);
    // void log(const CentroidalState& centroidal_state);
    // void log(const ContactState& contact_state);
    void log(const ImuMeasurement& imu_measurement);
    // void log(const std::map<std::string, JointMeasurement>& joints_measurement);
    // void log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement);

private:
    class Impl;  // Forward declaration of the implementation class
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace serow
