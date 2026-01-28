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
#include <algorithm>
#include <iostream>

#include "ForceTorqueMeasurementBuffer.hpp"

namespace serow {

ForceTorqueMeasurementBuffer::ForceTorqueMeasurementBuffer(const size_t max_size) : max_size_(max_size) {}

void ForceTorqueMeasurementBuffer::add(const ForceTorqueMeasurement& measurement) {
    // Check if the buffer is full
    while (measurements_.size() >= max_size_) {
        measurements_.pop_front();
    }

    // Add the measurement to the buffer 
    measurements_.push_back(measurement);
}

void ForceTorqueMeasurementBuffer::clear() {
    measurements_.clear();
}

size_t ForceTorqueMeasurementBuffer::size() const {
    return measurements_.size();
}

std::optional<std::pair<double, double>> ForceTorqueMeasurementBuffer::getTimeRange() const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    return std::make_pair(measurements_.front().timestamp, measurements_.back().timestamp);
}

bool ForceTorqueMeasurementBuffer::isTimestampInRange(const double timestamp,
                                                      const double tolerance) const {
    auto time_range = getTimeRange();
    if (!time_range.has_value()) {
        return false;
    }

    return timestamp >= (time_range.value().first - tolerance) &&
        timestamp <= (time_range.value().second + tolerance);
}

std::optional<ForceTorqueMeasurement> ForceTorqueMeasurementBuffer::get(const double timestamp,
                                                                        const double max_time_diff) const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    // Find the two measurements to interpolate between
    auto it = std::lower_bound(measurements_.begin(), measurements_.end(), timestamp,
                               [](const ForceTorqueMeasurement& m, double t) { return m.timestamp < t; });

    // Handle edge cases
    if (it == measurements_.begin()) {
        // Check if interpolation is within tolerance
        if (std::abs(it->timestamp - timestamp) <= max_time_diff) {
            return interpolate(*it, *it, timestamp, it->timestamp);
        }
        return std::nullopt;
    }

    if (it == measurements_.end()) {
        // Check if interpolation is within tolerance
        auto last_it = std::prev(it);
        if (std::abs(last_it->timestamp - timestamp) <= max_time_diff) {
            return interpolate(*last_it, *last_it, timestamp, last_it->timestamp);
        }
        return std::nullopt;
    }

    // Check for exact match
    if (std::abs(it->timestamp - timestamp) < 1e-6) {
        return *it;
    }

    // Check if interpolation is within tolerance
    auto prev_it = std::prev(it);

    // Check if timestamp is between the two measurements
    if (prev_it->timestamp <= timestamp && timestamp <= it->timestamp) {
        double time_diff =
            std::max(std::abs(timestamp - prev_it->timestamp), std::abs(timestamp - it->timestamp));

        if (time_diff <= max_time_diff) {
            return interpolate(*prev_it, *it, timestamp, prev_it->timestamp);
        }
    }

    // Check if timestamp is close to either measurement
    if (std::abs(prev_it->timestamp - timestamp) <= max_time_diff) {
        return interpolate(*prev_it, *prev_it, timestamp, prev_it->timestamp);
    }

    if (std::abs(it->timestamp - timestamp) <= max_time_diff) {
        return interpolate(*it, *it, timestamp, it->timestamp);
    }

    return std::nullopt;
}

ForceTorqueMeasurement ForceTorqueMeasurementBuffer::interpolate(const ForceTorqueMeasurement& m1, 
                                                                 const ForceTorqueMeasurement& m2, 
                                                                 const double target_timestamp, 
                                                                 const double t1) const {
    ForceTorqueMeasurement result;
    result.timestamp = target_timestamp;

    // Calculate interpolation weight factor
    const double t2 = m2.timestamp;
    double w = 0.0;

    if (std::abs(t2 - t1) > 1e-9) {
        w = (target_timestamp - t1) / (t2 - t1);
    }

    // Clamp w to [0, 1] for safety
    w = std::max(0.0, std::min(1.0, w));

    result.timestamp = target_timestamp;
    // Interpolate force (linear interpolation)
    result.force =
        m1.force + w * (m2.force - m1.force);

    // Interpolate center of pressure (linear interpolation)
    result.cop = m1.cop + w * (m2.cop - m1.cop);

    // Interpolate torque (linear interpolation)
    result.torque = std::nullopt;
    if (m1.torque.has_value() && m2.torque.has_value()) {
        result.torque =
            m1.torque.value() + w * (m2.torque.value() - m1.torque.value());
    }
    return result;
}

}  // namespace serow
