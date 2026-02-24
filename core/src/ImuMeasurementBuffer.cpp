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

#ifdef __linux__
#include <eigen3/Eigen/Eigenvalues>
#else
#include <Eigen/Eigenvalues>
#endif

#include "ImuMeasurementBuffer.hpp"

namespace serow {

ImuMeasurementBuffer::ImuMeasurementBuffer(const size_t max_size, bool interpolate_covariance) : max_size_(max_size), interpolate_covariance_(interpolate_covariance) {}

void ImuMeasurementBuffer::add(const ImuMeasurement& measurement) {
    // Check if the buffer is full
    while (measurements_.size() >= max_size_) {
        measurements_.pop_front();
    }

    // Add the measurement to the buffer 
    measurements_.push_back(measurement);
}

void ImuMeasurementBuffer::clear() {
    measurements_.clear();
}

size_t ImuMeasurementBuffer::size() const {
    return measurements_.size();
}

std::optional<std::pair<double, double>> ImuMeasurementBuffer::getTimeRange() const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    return std::make_pair(measurements_.front().timestamp, measurements_.back().timestamp);
}

bool ImuMeasurementBuffer::isTimestampInRange(const double timestamp,
                                              const double tolerance) const {
    auto time_range = getTimeRange();
    if (!time_range.has_value()) {
        return false;
    }

    return timestamp >= (time_range.value().first - tolerance) &&
        timestamp <= (time_range.value().second + tolerance);
}

std::optional<ImuMeasurement> ImuMeasurementBuffer::get(const double timestamp,
                                                        const double max_time_diff) const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    // Find the two measurements to interpolate between
    auto it = std::lower_bound(measurements_.begin(), measurements_.end(), timestamp,
                               [](const ImuMeasurement& m, double t) { return m.timestamp < t; });

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

ImuMeasurement ImuMeasurementBuffer::interpolate(const ImuMeasurement& m1, 
                                                           const ImuMeasurement& m2, 
                                                           const double target_timestamp, 
                                                           const double t1) const {
    ImuMeasurement result;
    result.timestamp = target_timestamp;

    // Calculate interpolation weight factor
    const double t2 = m2.timestamp;
    double w = 0.0;
    if (std::abs(t2 - t1) > 1e-9) {
        w = (target_timestamp - t1) / (t2 - t1);
    }
    w = std::max(0.0, std::min(1.0, w));

    result.timestamp = target_timestamp;

    // Interpolate linear acceleration (linear interpolation)
    result.linear_acceleration =
        m1.linear_acceleration + w * (m2.linear_acceleration - m1.linear_acceleration);

    // Interpolate angular velocity (linear interpolation)
    result.angular_velocity =
        m1.angular_velocity + w * (m2.angular_velocity - m1.angular_velocity);

    // Interpolate orientation (slerp interpolation)
    result.orientation =
        m1.orientation.slerp(w, m2.orientation);

    if (interpolate_covariance_) {
        // Interpolate linear acceleration covariance (convex combination preserves PSD; then enforce PD)
        result.linear_acceleration_cov = makePositiveDefinite(
            m1.linear_acceleration_cov + w * (m2.linear_acceleration_cov - m1.linear_acceleration_cov));

        // Interpolate angular velocity covariance (convex combination preserves PSD; then enforce PD)
        result.angular_velocity_cov = makePositiveDefinite(
            m1.angular_velocity_cov + w * (m2.angular_velocity_cov - m1.angular_velocity_cov));

        // Interpolate base orientation covariance (convex combination preserves PSD; then enforce PD)
        result.orientation_cov = makePositiveDefinite(
            m1.orientation_cov + w * (m2.orientation_cov - m1.orientation_cov));
    }

    return result;
}

Eigen::Matrix3d ImuMeasurementBuffer::makePositiveDefinite(const Eigen::Matrix3d& sym) const {
    // Symmetrize (in case of numerical drift)
    const Eigen::Matrix3d A = 0.5 * (sym + sym.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(A);
    if (solver.info() != Eigen::Success) {
        return Eigen::Matrix3d::Identity();  // fallback
    }

    const Eigen::Vector3d& ev = solver.eigenvalues();
    constexpr double min_eigenvalue = 1e-8;
    Eigen::Vector3d ev_clamped = ev;
    for (int i = 0; i < 3; ++i) {
        if (ev_clamped(i) < min_eigenvalue) {
            ev_clamped(i) = min_eigenvalue;
        }
    }

    return solver.eigenvectors() * ev_clamped.asDiagonal() * solver.eigenvectors().transpose();
}

}  // namespace serow
