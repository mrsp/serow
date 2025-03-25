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
/**
 * @file DebugLog.hpp
 * @brief Defines data structures and accessors for the robot states in the SEROW state estimator.
 **/
#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <fstream>
#include <map>
#include <string>
#include <mcap/mcap.hpp>
#include <mcap/writer.hpp>

#include "Measurement.hpp"
#include "State.hpp"

namespace serow {

class DebugLogger {
   public:
    DebugLogger(const std::string& log_file_path = "/tmp/serow_log.mcap");
    ~DebugLogger();
    void log(const BaseState& base_state);
    void log(const CentroidalState& centroidal_state);
    void log(const ContactState& contact_state);
    void log(const ImuMeasurement& imu_measurement);
    void log(const std::map<std::string, JointMeasurement>& joints_measurement);
    void log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement);

   private:
    std::unique_ptr<mcap::McapWriter> writer_;
    std::unique_ptr<mcap::FileWriter> file_writer_;
};

}  // namespace serow
