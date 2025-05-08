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
 * @file ExteroceptionLogger.hpp
 * @brief Defines the ExteroceptionLogger class for logging robot state data in MCAP format.
 *
 * The ExteroceptionLogger provides functionality to log various exteroceptive measurements and
 * estimate in a binary MCAP format, including:
 * - Local map
 *
 * The logger uses the PIMPL pattern to hide implementation details and minimize
 * compilation dependencies.
 **/
#pragma once

#include <fstream>
#include <mcap/mcap.hpp>
#include <mcap/writer.hpp>
#include <memory>
#include <string>

#include "Schemas.hpp"
#include "common.hpp"

namespace serow {

class ExteroceptionLogger {
public:
    ExteroceptionLogger(const std::string& log_file_path = "/tmp/serow_exteroception.mcap");
    ~ExteroceptionLogger();

    // Delete copy operations
    ExteroceptionLogger(const ExteroceptionLogger&) = delete;
    ExteroceptionLogger& operator=(const ExteroceptionLogger&) = delete;

    // Allow move operations
    ExteroceptionLogger(ExteroceptionLogger&&) noexcept = default;
    ExteroceptionLogger& operator=(ExteroceptionLogger&&) noexcept = default;

    void log(const LocalMapState& local_map_state);
    double getLastTimestamp() const;
    bool isInitialized() const;
    void setStartTime(double timestamp);

private:
    class Impl;  // Forward declaration of the implementation class
    std::unique_ptr<Impl> pimpl_;
    double last_timestamp_{-1.0};
};
}  // namespace serow
