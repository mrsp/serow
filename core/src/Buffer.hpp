#pragma once

#include <cmath>
#include <deque>
#include <optional>

#include "Measurement.hpp"

namespace serow {

class ImuMeasurementBuffer {
public:
    ImuMeasurementBuffer(const size_t max_size);

    /**
     * @brief Add an IMU measurement to the buffer
     * @param measurement IMU measurement to add
     */
    void add(const ImuMeasurement& measurement);

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
     * @brief Check if the buffer is sorted by timestamp (ascending order)
     * @return true if sorted, false otherwise
     */
    bool isSorted() const;

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
     * @brief Get IMU measurement at the exact timestamp, or interpolate between measurements
     * Assumes measurements come in monotonically increasing time order
     * @param timestamp Target timestamp for the measurement
     * @return Interpolated IMU measurement, or nullopt if timestamp is outside buffer range
     */
    std::optional<ImuMeasurement> get(const double timestamp,
                                      const double max_time_diff = 0.015) const;

    /**
     * @brief Get the IMU measurement with the closest timestamp using the most efficient method
     * Automatically chooses between linear search (for small buffers) and binary search (for large
     * buffers)
     * @param timestamp Target timestamp to search for
     * @param max_time_diff Maximum allowed time difference (default: 50ms)
     * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within
     * max_time_diff
     */
    std::optional<ImuMeasurement> getClosest(const double timestamp,
                                             const double max_time_diff = 0.015) const;

private:
    size_t max_size_{100};
    std::deque<ImuMeasurement> measurements_{};

    /**
     * @brief Interpolate between two IMU measurements at a given timestamp
     * @param m1 First measurement (earlier timestamp)
     * @param m2 Second measurement (later timestamp)
     * @param target_timestamp Target timestamp for interpolation
     * @param t1 Timestamp of first measurement
     * @return Interpolated IMU measurement
     */
    ImuMeasurement interpolate(const ImuMeasurement& m1, const ImuMeasurement& m2,
                               double target_timestamp, double t1) const;

    /**
     * @brief Get the IMU measurement with the closest timestamp to the given timestamp
     * @param timestamp Target timestamp to search for
     * @param max_time_diff Maximum allowed time difference (default: 50ms)
     * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within
     * max_time_diff
     */
    std::optional<ImuMeasurement> getClosestBinary(const double timestamp,
                                                   const double max_time_diff = 0.015) const;

    /**
     * @brief Get the IMU measurement with the closest timestamp using linear search (for small
     * buffers)
     * @param timestamp Target timestamp to search for
     * @param max_time_diff Maximum allowed time difference (default: 50ms)
     * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within
     * max_time_diff
     */
    std::optional<ImuMeasurement> getClosestLinear(const double timestamp,
                                                   const double max_time_diff = 0.015) const;
};

}  // namespace serow
