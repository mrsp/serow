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
 * @file ProprioceptionLogger.hpp
 * @brief Defines the ProprioceptionLogger class for logging robot state data in MCAP format.
 *
 * The ProprioceptionLogger provides functionality to log various robot states and measurements
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
#include <mcap/mcap.hpp>
#include <mcap/writer.hpp>
#include <memory>
#include <string>

#include "Measurement.hpp"
#include "State.hpp"
#include "TerrainElevation.hpp"

namespace serow {

class ProprioceptionLogger {
public:
    ProprioceptionLogger(const std::string& log_file_path = "/tmp/serow_proprioception.mcap");
    ~ProprioceptionLogger();

    // Delete copy operations
    ProprioceptionLogger(const ProprioceptionLogger&) = delete;
    ProprioceptionLogger& operator=(const ProprioceptionLogger&) = delete;

    // Allow move operations
    ProprioceptionLogger(ProprioceptionLogger&&) noexcept = default;
    ProprioceptionLogger& operator=(ProprioceptionLogger&&) noexcept = default;

    void log(const BaseState& base_state);
    void log(const CentroidalState& centroidal_state);
    void log(const ContactState& contact_state);
    void log(const ImuMeasurement& imu_measurement);
    void log(const std::map<std::string, JointMeasurement>& joints_measurement);
    void log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement);

private:
    class Impl;  // Forward declaration of the implementation class
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace serow
