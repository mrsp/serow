#include <algorithm>

#include "Buffer.hpp"

namespace serow {

ImuMeasurementBuffer::ImuMeasurementBuffer(const size_t max_size) : max_size_(max_size) {}

void ImuMeasurementBuffer::add(const ImuMeasurement& measurement) {
    if (measurements_.size() >= max_size_) {
        measurements_.pop_front();
    }
    measurements_.push_back(measurement);
}

void ImuMeasurementBuffer::clear() {
    measurements_.clear();
}

size_t ImuMeasurementBuffer::size() const {
    return measurements_.size();
}

bool ImuMeasurementBuffer::isSorted() const {
    if (measurements_.size() <= 1) {
        return true;
    }

    for (auto it = measurements_.begin(); it != std::prev(measurements_.end()); ++it) {
        if (it->timestamp > std::next(it)->timestamp) {
            return false;
        }
    }
    return true;
}

std::optional<std::pair<double, double>> ImuMeasurementBuffer::getTimeRange() const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    if (isSorted()) {
        return std::make_pair(measurements_.front().timestamp, measurements_.back().timestamp);
    } else {
        // Find min and max timestamps
        auto [min_it, max_it] =
            std::minmax_element(measurements_.begin(), measurements_.end(),
                                [](const ImuMeasurement& a, const ImuMeasurement& b) {
                                    return a.timestamp < b.timestamp;
                                });
        return std::make_pair(min_it->timestamp, max_it->timestamp);
    }
}

bool ImuMeasurementBuffer::isTimestampInRange(const double timestamp,
                                              const double tolerance) const {
    auto time_range = getTimeRange();
    if (!time_range) {
        return false;
    }

    return timestamp >= (time_range->first - tolerance) &&
        timestamp <= (time_range->second + tolerance);
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
    if (std::abs(it->timestamp - timestamp) < 1e-9) {
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

std::optional<ImuMeasurement> ImuMeasurementBuffer::getClosest(const double timestamp,
                                                               const double max_time_diff) const {
    // For small buffers, linear search is more efficient due to cache locality
    // For large buffers, binary search is more efficient
    constexpr size_t LINEAR_SEARCH_THRESHOLD = 32;

    if (measurements_.size() <= LINEAR_SEARCH_THRESHOLD) {
        return getClosestLinear(timestamp, max_time_diff);
    } else {
        // Ensure buffer is sorted for binary search
        if (!isSorted()) {
            // If not sorted, fall back to linear search for this call
            return getClosestLinear(timestamp, max_time_diff);
        }
        return getClosestBinary(timestamp, max_time_diff);
    }
}

std::optional<ImuMeasurement> ImuMeasurementBuffer::getClosestBinary(
    const double timestamp, const double max_time_diff) const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    // Binary search for the closest timestamp
    auto it = std::lower_bound(measurements_.begin(), measurements_.end(), timestamp,
                               [](const ImuMeasurement& m, double t) { return m.timestamp < t; });

    // Handle edge cases
    if (it == measurements_.begin()) {
        // Target timestamp is before all measurements
        if (std::abs(it->timestamp - timestamp) <= max_time_diff) {
            return *it;
        }
        return std::nullopt;
    }

    if (it == measurements_.end()) {
        // Target timestamp is after all measurements
        --it;
        if (std::abs(it->timestamp - timestamp) <= max_time_diff) {
            return *it;
        }
        return std::nullopt;
    }

    // Find the closest measurement between it and it-1
    auto prev_it = std::prev(it);
    double diff_curr = std::abs(it->timestamp - timestamp);
    double diff_prev = std::abs(prev_it->timestamp - timestamp);

    if (diff_curr <= diff_prev && diff_curr <= max_time_diff) {
        return *it;
    } else if (diff_prev <= max_time_diff) {
        return *prev_it;
    }

    return std::nullopt;
}

std::optional<ImuMeasurement> ImuMeasurementBuffer::getClosestLinear(
    const double timestamp, const double max_time_diff) const {
    if (measurements_.empty()) {
        return std::nullopt;
    }

    auto closest_it = measurements_.begin();
    double min_diff = std::abs(closest_it->timestamp - timestamp);

    for (auto it = std::next(measurements_.begin()); it != measurements_.end(); ++it) {
        double diff = std::abs(it->timestamp - timestamp);
        if (diff < min_diff) {
            min_diff = diff;
            closest_it = it;
        }
    }

    if (min_diff <= max_time_diff) {
        return *closest_it;
    }

    return std::nullopt;
}

ImuMeasurement ImuMeasurementBuffer::interpolate(const ImuMeasurement& m1, const ImuMeasurement& m2,
                                                 double target_timestamp, double t1) const {
    ImuMeasurement result;
    result.timestamp = target_timestamp;

    // Calculate interpolation factor
    double t2 = m2.timestamp;
    double alpha = 0.0;

    if (std::abs(t2 - t1) > 1e-9) {
        alpha = (target_timestamp - t1) / (t2 - t1);
    }

    // Clamp alpha to [0, 1] for safety
    alpha = std::max(0.0, std::min(1.0, alpha));

    result.timestamp = target_timestamp;
    // Interpolate linear acceleration (linear interpolation)
    result.linear_acceleration =
        m1.linear_acceleration + alpha * (m2.linear_acceleration - m1.linear_acceleration);

    // Interpolate angular velocity (linear interpolation)
    result.angular_velocity =
        m1.angular_velocity + alpha * (m2.angular_velocity - m1.angular_velocity);

    // Interpolate angular acceleration (linear interpolation)
    result.angular_acceleration =
        m1.angular_acceleration + alpha * (m2.angular_acceleration - m1.angular_acceleration);

    // Interpolate orientation (spherical linear interpolation for quaternions)
    if (m1.orientation.dot(m2.orientation) >= 0) {
        // Same hemisphere, use slerp
        result.orientation = m1.orientation.slerp(alpha, m2.orientation);
    } else {
        // Different hemispheres, use slerp with conjugate quaternion
        result.orientation = m1.orientation.slerp(alpha, m2.orientation.conjugate());
    }

    // Skip covariance interpolation
    result.orientation_cov = m1.orientation_cov;
    result.angular_velocity_cov = m1.angular_velocity_cov;
    result.linear_acceleration_cov = m1.linear_acceleration_cov;
    result.angular_velocity_bias_cov = m1.angular_velocity_bias_cov;
    result.linear_acceleration_bias_cov = m1.linear_acceleration_bias_cov;

    return result;
}

}  // namespace serow
