/**
 * @file Measurements.hpp
 * @brief Defines various measurement structs used in the Serow library
 * @author Stylianos Piperakis
 */

#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <map>
#include <optional>
#include <string>
#include <algorithm>
#include <deque>

namespace serow {

/**
 * @struct JointMeasurement
 * @brief Represents a joint measurement consisting of timestamp, position, and optional velocity
 */
struct JointMeasurement {
    double timestamp{};              ///< Timestamp of the measurement (s)
    double position{};               ///< Joint position measurement (rad)
    std::optional<double> velocity;  ///< Optional joint velocity measurement (rad/s)
};

/**
 * @struct ImuMeasurement
 * @brief Represents IMU (Inertial Measurement Unit) measurements including linear acceleration,
 * angular velocity, and their covariances
 */
struct ImuMeasurement {
    double timestamp{};                      ///< Timestamp of the measurement (s)
    Eigen::Vector3d linear_acceleration{};   ///< Linear acceleration measured by IMU (m/s^2)
    Eigen::Vector3d angular_velocity{};      ///< Angular velocity measured by IMU (rad/s)
    Eigen::Quaterniond orientation{};        ///< Orientation measured by IMU (quaternion)
    Eigen::Vector3d angular_acceleration{};  ///< Angular acceleration measured by IMU (rad/s^2)

    Eigen::Matrix3d orientation_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of orientation (rad^2)
    Eigen::Matrix3d angular_velocity_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of angular velocity (rad^2/s^2)
    Eigen::Matrix3d linear_acceleration_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of linear acceleration (m^2/s^4)
    Eigen::Matrix3d angular_velocity_bias_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of angular velocity bias (rad^2/s^2)
    Eigen::Matrix3d linear_acceleration_bias_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of linear acceleration bias (m^2/s^4)
};

/**
 * @struct ForceTorqueMeasurement
 * @brief Represents force-torque sensor measurements including force, center of pressure (COP),
 * and optional torque
 */
struct ForceTorqueMeasurement {
    double timestamp{};                              ///< Timestamp of the measurement (s)
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};  ///< Force measured by force-torque sensor (N)
    Eigen::Vector3d cop{
        Eigen::Vector3d::Zero()};  ///< Center of pressure (COP) measured by force-torque sensor (m)
    std::optional<Eigen::Vector3d>
        torque;  ///< Optional torque measured by force-torque sensor (Nm)
};

/**
 * @struct GroundReactionForceMeasurement
 * @brief Represents ground reaction force measurements including force and center of pressure (COP)
 */
struct GroundReactionForceMeasurement {
    double timestamp{};                              ///< Timestamp of the measurement (s)
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};  ///< Ground reaction force (N)
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};    ///< Center of pressure (COP) (m)
};

/**
 * @struct KinematicMeasurement
 * @brief Represents kinematic measurements including contact status, position, orientation,
 * and other dynamics-related quantities
 */
struct KinematicMeasurement {
    double timestamp{};  ///< Timestamp of the measurement (s)
    Eigen::Vector3d base_linear_velocity{Eigen::Vector3d::Zero()};  ///< Base linear velocity (m/s)
    std::map<std::string, bool>
        contacts_status;  ///< Map of contact status for different parts (0 or 1)
    std::map<std::string, double> contacts_probability;  ///< Map of contact probabilities ([0, 1])
    std::map<std::string, bool> is_new_contact;  ///< Holds contact frame name to flag to indicate
                                                 ///< if a new contact has been detected

    std::map<std::string, Eigen::Vector3d>
        contacts_position;  ///< Map of contact positions relative to base frame (m)
    std::map<std::string, Eigen::Vector3d> base_to_foot_positions;
    std::map<std::string, Eigen::Quaterniond>
        base_to_foot_orientations;  ///< Map of foot orientations relative to base frame
                                    ///< (quaternion)
    std::map<std::string, Eigen::Vector3d>
        base_to_foot_linear_velocities;  ///< Map of foot linear velocities relative to base frame
                                         ///< (m/s)
    std::map<std::string, Eigen::Vector3d>
        base_to_foot_angular_velocities;  ///< Map of foot angular velocities relative to base frame
                                          ///< (rad/s)
    std::map<std::string, Eigen::Matrix3d>
        contacts_position_noise;  ///< Map of contact position noise covariances relative to base
                                  ///< frame (m^2)
    std::optional<std::map<std::string, Eigen::Quaterniond>>
        contacts_orientation;  ///< Optional map of contact orientations relative to base frame
    std::optional<std::map<std::string, Eigen::Matrix3d>>
        contacts_orientation_noise;  ///< Optional map of contact orientation noise covariances
                                     ///< relative to base frame (rad^2)
    Eigen::Vector3d com_angular_momentum_derivative{
        Eigen::Vector3d::Zero()};  ///< Derivative of center of mass (COM) angular momentum (Nm)
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};  ///< Center of mass (COM) position (m)
    Eigen::Vector3d com_linear_acceleration{
        Eigen::Vector3d::Zero()};  ///< Center of mass (COM) linear acceleration (m/s^2)
    Eigen::Matrix3d base_linear_velocity_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of base linear velocity (m^2/s^2)
    Eigen::Matrix3d position_slip_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of position slip (m^2)
    Eigen::Matrix3d orientation_slip_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of orientation slip (rad^2)
    Eigen::Matrix3d position_cov{Eigen::Matrix3d::Identity()};  ///< Covariance of position (m^2)
    Eigen::Matrix3d orientation_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of orientation (rad^2)
    Eigen::Matrix3d com_position_process_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of COM position process noise (m^2)
    Eigen::Matrix3d com_linear_velocity_process_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of COM linear velocity process noise
                                       ///< (m^2/s^2)
    Eigen::Matrix3d external_forces_process_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of external forces process noise (N^2)
    Eigen::Matrix3d com_position_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of COM position (m^2)
    Eigen::Matrix3d com_linear_acceleration_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance of COM linear acceleration (m^2/s^4)
};

/**
 * @struct OdometryMeasurement
 * @brief Represents odometry measurements including base position, orientation, and their
 * covariances
 */
struct OdometryMeasurement {
    double timestamp{};                                      ///< Timestamp of the measurement (s)
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};  ///< Base position from odometry (m)
    Eigen::Quaterniond base_orientation{
        Eigen::Quaterniond::Identity()};  ///< Base orientation from odometry
    Eigen::Matrix3d base_position_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of base position (m^2)
    Eigen::Matrix3d base_orientation_cov{
        Eigen::Matrix3d::Identity()};  ///< Covariance matrix of base orientation (rad^2)
};

/**
 * @struct TerrainMeasurement
 * @brief Represents terrain height measurements at a specific timestamp
 */
struct TerrainMeasurement {
    double timestamp{};      ///< Timestamp of the measurement (s)
    double height{};         ///< Terrain height measurement (m)
    double height_cov{1.0};  ///< Covariance of terrain height measurement (m^2)
    TerrainMeasurement(double timestamp, double height, double height_cov = 1.0)
        : timestamp(timestamp), height(height), height_cov(height_cov) {}
};

/**
 * @typedef ContactMeasurement
 * @brief Alias for a contact measurement, typically representing contact forces or pressure
 */
using ContactMeasurement = double;

/**
 * @struct BasePoseGroundTruth
 * @brief Represents base pose measurements including position and orientation
 */
struct BasePoseGroundTruth {
    double timestamp{};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
};

class ImuMeasurementBuffer {
    public:
        ImuMeasurementBuffer(const size_t max_size) : max_size_(max_size) {}
    
        void add(const ImuMeasurement& measurement) {
            if (measurements_.size() > max_size_) {
                measurements_.pop_front();
            }
            measurements_.push_back(measurement);
        }
    
        void clear() {
            measurements_.clear();
        }
    
        size_t size() const {
            return measurements_.size();
        }
    
        /**
         * @brief Check if the buffer is sorted by timestamp (ascending order)
         * @return true if sorted, false otherwise
         */
        bool isSorted() const {
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
    
    
        /**
         * @brief Get the time range of the buffer
         * @return Pair of (earliest_timestamp, latest_timestamp) or empty pair if buffer is empty
         */
        std::optional<std::pair<double, double>> getTimeRange() const {
            if (measurements_.empty()) {
                return std::nullopt;
            }
            
            if (isSorted()) {
                return std::make_pair(measurements_.front().timestamp, measurements_.back().timestamp);
            } else {
                // Find min and max timestamps
                auto [min_it, max_it] = std::minmax_element(measurements_.begin(), measurements_.end(),
                    [](const ImuMeasurement& a, const ImuMeasurement& b) {
                        return a.timestamp < b.timestamp;
                    });
                return std::make_pair(min_it->timestamp, max_it->timestamp);
            }
        }
    
        /**
         * @brief Check if a timestamp falls within the buffer's time range
         * @param timestamp Timestamp to check
         * @param tolerance Additional tolerance beyond the buffer's time range
         * @return true if timestamp is within range, false otherwise
         */
        bool isTimestampInRange(const double timestamp, const double tolerance = 0.0) const {
            auto time_range = getTimeRange();
            if (!time_range) {
                return false;
            }
            
            return timestamp >= (time_range->first - tolerance) && 
                   timestamp <= (time_range->second + tolerance);
        }
    
        
        /**
         * @brief Interpolate between two IMU measurements at a given timestamp
         * @param m1 First measurement (earlier timestamp)
         * @param m2 Second measurement (later timestamp)
         * @param target_timestamp Target timestamp for interpolation
         * @param t1 Timestamp of first measurement
         * @return Interpolated IMU measurement
         */
        ImuMeasurement interpolate(const ImuMeasurement& m1, const ImuMeasurement& m2, 
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
            
            // Interpolate linear acceleration (linear interpolation)
            result.linear_acceleration = m1.linear_acceleration + 
                                       alpha * (m2.linear_acceleration - m1.linear_acceleration);
            
            // Interpolate angular velocity (linear interpolation)
            result.angular_velocity = m1.angular_velocity + 
                                    alpha * (m2.angular_velocity - m1.angular_velocity);
            
            // Interpolate angular acceleration (linear interpolation)
            result.angular_acceleration = m1.angular_acceleration + 
                                        alpha * (m2.angular_acceleration - m1.angular_acceleration);
            
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
    
        /**
         * @brief Get IMU measurement at the exact timestamp, or interpolate between measurements
         * Assumes measurements come in monotonically increasing time order
         * @param timestamp Target timestamp for the measurement
         * @param max_interpolation_time Maximum time difference for interpolation (default: 0.1s)
         * @return Interpolated IMU measurement, or nullopt if timestamp is outside buffer range or interpolation range
         */
        std::optional<ImuMeasurement> getInterpolated(const double timestamp, const double max_interpolation_time = 0.1) const {
            if (measurements_.empty()) {
                return std::nullopt;
            }
    
            // Check if timestamp is within buffer range
            if (timestamp < measurements_.front().timestamp || timestamp > measurements_.back().timestamp) {
                return std::nullopt;
            }
    
            // Find the two measurements to interpolate between
            auto it = std::lower_bound(measurements_.begin(), measurements_.end(), timestamp,
                [](const ImuMeasurement& m, double t) { return m.timestamp < t; });
    
            // Handle edge cases
            if (it == measurements_.begin()) {
                // Check if interpolation is within tolerance
                if (std::abs(it->timestamp - timestamp) <= max_interpolation_time) {
                    return interpolate(*it, *it, timestamp, it->timestamp);
                }
                return std::nullopt;
            }
    
            if (it == measurements_.end()) {
                // Check if interpolation is within tolerance
                auto last_it = std::prev(it);
                if (std::abs(last_it->timestamp - timestamp) <= max_interpolation_time) {
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
            double time_diff = std::max(std::abs(timestamp - prev_it->timestamp), 
                                       std::abs(timestamp - it->timestamp));
            
            if (time_diff <= max_interpolation_time) {
                return interpolate(*prev_it, *it, timestamp, prev_it->timestamp);
            }
    
            return std::nullopt;
        }
    
        /**
         * @brief Get IMU measurement at the exact timestamp, or interpolate between measurements
         * Assumes measurements come in monotonically increasing time order
         * @param timestamp Target timestamp for the measurement
         * @return Interpolated IMU measurement, or nullopt if timestamp is outside buffer range
         */
        std::optional<ImuMeasurement> get(const double timestamp) const {
            if (measurements_.empty()) {
                return std::nullopt;
            }
    
            // Check if timestamp is within buffer range
            if (timestamp < measurements_.front().timestamp || timestamp > measurements_.back().timestamp) {
                return std::nullopt;
            }
    
            // Find the two measurements to interpolate between
            auto it = std::lower_bound(measurements_.begin(), measurements_.end(), timestamp,
                [](const ImuMeasurement& m, double t) { return m.timestamp < t; });
    
            // Handle edge cases
            if (it == measurements_.begin()) {
                // Exact match at the beginning
                if (std::abs(it->timestamp - timestamp) < 1e-9) {
                    return *it;
                }
                // Interpolate between first measurement and a virtual "zero" measurement
                return interpolate(*it, *it, timestamp, it->timestamp);
            }
    
            if (it == measurements_.end()) {
                // Exact match at the end
                auto last_it = std::prev(it);
                if (std::abs(last_it->timestamp - timestamp) < 1e-9) {
                    return *last_it;
                }
                // Interpolate between last measurement and a virtual "future" measurement
                return interpolate(*last_it, *last_it, timestamp, last_it->timestamp);
            }
    
            // Check for exact match
            if (std::abs(it->timestamp - timestamp) < 1e-9) {
                return *it;
            }
    
            // Interpolate between the measurement before and after the target timestamp
            auto prev_it = std::prev(it);
            return interpolate(*prev_it, *it, timestamp, prev_it->timestamp);
        }
    
        /**
         * @brief Get the IMU measurement with the closest timestamp to the given timestamp
         * @param timestamp Target timestamp to search for
         * @param max_time_diff Maximum allowed time difference (default: 50ms)
         * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within max_time_diff
         */
        std::optional<ImuMeasurement> getClosest(const double timestamp, const double max_time_diff = 0.05) const {
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
    
        /**
         * @brief Get the IMU measurement with the closest timestamp using linear search (for small buffers)
         * @param timestamp Target timestamp to search for
         * @param max_time_diff Maximum allowed time difference (default: 50ms)
         * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within max_time_diff
         */
        std::optional<ImuMeasurement> getClosestLinear(const double timestamp, const double max_time_diff = 0.05) const {
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
    
        /**
         * @brief Get the IMU measurement with the closest timestamp using interpolation-aware search
         * @param timestamp Target timestamp to search for
         * @param max_time_diff Maximum allowed time difference (default: 50ms)
         * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within max_time_diff
         */
        std::optional<ImuMeasurement> getClosestInterpolated(const double timestamp, const double max_time_diff = 0.05) const {
            if (measurements_.size() < 2) {
                return getClosest(timestamp, max_time_diff);
            }
    
            // Binary search for the closest timestamp
            auto it = std::lower_bound(measurements_.begin(), measurements_.end(), timestamp,
                [](const ImuMeasurement& m, double t) { return m.timestamp < t; });
    
            // Handle edge cases
            if (it == measurements_.begin()) {
                if (std::abs(it->timestamp - timestamp) <= max_time_diff) {
                    return *it;
                }
                return std::nullopt;
            }
    
            if (it == measurements_.end()) {
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
    
            // If both measurements are within max_time_diff, return the closer one
            if (diff_curr <= max_time_diff && diff_prev <= max_time_diff) {
                return (diff_curr <= diff_prev) ? *it : *prev_it;
            } else if (diff_curr <= max_time_diff) {
                return *it;
            } else if (diff_prev <= max_time_diff) {
                return *prev_it;
            }
            return std::nullopt;
        }
    
        /**
         * @brief Get the IMU measurement with the closest timestamp using the most efficient method
         * Automatically chooses between linear search (for small buffers) and binary search (for large buffers)
         * @param timestamp Target timestamp to search for
         * @param max_time_diff Maximum allowed time difference (default: 50ms)
         * @return Optional IMU measurement with closest timestamp, or nullopt if no measurement within max_time_diff
         */
        std::optional<ImuMeasurement> getClosestEfficient(const double timestamp, const double max_time_diff = 0.05) const {
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
                return getClosest(timestamp, max_time_diff);
            }
        }
    
    private:
        size_t max_size_{100};
        double max_time_diff_{0.01};  // 10ms
        std::deque<ImuMeasurement> measurements_{};
};


}  // namespace serow
