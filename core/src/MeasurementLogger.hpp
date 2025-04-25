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
 * @file MeasurementLogger.hpp
 * @brief Defines the MeasurementLogger class for logging robot measurement data in MCAP format.
 *
 * The MeasurementLogger provides functionality to log various robot measurements
 * in a binary MCAP format, including:
 * - IMU measurements
 * - Kinematic measurements
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
#include "Schemas.hpp"

namespace serow {

class MeasurementLogger {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MeasurementLogger(const std::string& log_file_path = "/tmp/serow_measurements.mcap");
    ~MeasurementLogger();

    // Delete copy operations
    MeasurementLogger(const MeasurementLogger&) = delete;
    MeasurementLogger& operator=(const MeasurementLogger&) = delete;

    // Allow move operations
    MeasurementLogger(MeasurementLogger&&) noexcept = default;
    MeasurementLogger& operator=(MeasurementLogger&&) noexcept = default;

    void log(const ImuMeasurement& imu_measurement);
    void log(const KinematicMeasurement& kinematic_measurement);
    void log(const BasePoseGroundTruth& base_pose_ground_truth);
    void setStartTime(double timestamp);
    bool isInitialized() const;

private:
    class Impl;  // Forward declaration of the implementation class
    std::unique_ptr<Impl> pimpl_;
};

}  // namespace serow
