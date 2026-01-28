/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
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

#include <cmath>
#include <deque>
#include <optional>
#include <set>
#include <string>
#include <vector>
#include <map>

#include "Measurement.hpp"

namespace serow {

class ForceTorqueMeasurementBuffer {
public:
    ForceTorqueMeasurementBuffer(const size_t max_size = 1000);

    /**
     * @brief Add a synchronized vector of force-torque measurements to the buffer
     * @param measurement ForceTorqueMeasurement to add
     */
    void add(const ForceTorqueMeasurement& measurement);

    /**
     * @brief Clear the buffer
     */
    void clear();

    /**
     * @brief Get the size of the buffer
     * @return Size of the buffer
     */
    size_t size() const;

    /**
     * @brief Get the time range of the buffer
     * @return Pair of (earliest_timestamp, latest_timestamp) or empty pair if buffer is empty
     */
    std::optional<std::pair<double, double>> getTimeRange() const;

    /**
     * @brief Check if a timestamp falls within the buffer's time range
     * @param timestamp Timestamp to check
     * @param tolerance Additional tolerance beyond the buffer's time range
     * @return true if timestamp is within range, false otherwise
     */
    bool isTimestampInRange(const double timestamp, const double tolerance = 0.005) const;

    /**
     * @brief Get a ForceTorqueMeasurement at the exact timestamp, or interpolate between measurements
     * Assumes measurements come in monotonically increasing time order
     * @param timestamp Target timestamp for the measurement
     * @param max_time_diff Maximum allowed time difference
     * @return Interpolated ForceTorqueMeasurement, or nullopt if timestamp is outside buffer range
     */
    std::optional<ForceTorqueMeasurement> get(const double timestamp, 
                                              const double max_time_diff = 0.015) const;

private:
    size_t max_size_{1000};
    std::deque<ForceTorqueMeasurement> measurements_{};

    /**
     * @brief Interpolate between two vectors of ForceTorqueMeasurements at a given timestamp
     * @param m1 First measurement (earlier timestamp)
     * @param m2 Second measurement (later timestamp)
     * @param target_timestamp Target timestamp for interpolation
     * @param t1 Timestamp of first measurement
     * @return Interpolated ForceTorqueMeasurement
     */
    ForceTorqueMeasurement interpolate(const ForceTorqueMeasurement& m1, 
                                       const ForceTorqueMeasurement& m2, 
                                       const double target_timestamp, 
                                       const double t1) const;
};

}  // namespace serow
