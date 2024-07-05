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

namespace serow {

/**
 * @struct JointMeasurement
 * @brief Represents a joint measurement consisting of timestamp, position, and optional velocity
 */
struct JointMeasurement {
    double timestamp{};                     ///< Timestamp of the measurement
    double position{};                      ///< Joint position measurement
    std::optional<double> velocity{};       ///< Optional joint velocity measurement
};

/**
 * @struct ImuMeasurement
 * @brief Represents IMU (Inertial Measurement Unit) measurements including linear acceleration,
 * angular velocity, and their covariances
 */
struct ImuMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement
    Eigen::Vector3d linear_acceleration{};               ///< Linear acceleration measured by IMU
    Eigen::Vector3d angular_velocity{};                  ///< Angular velocity measured by IMU
    Eigen::Vector3d angular_acceleration{};              ///< Angular acceleration measured by IMU
    Eigen::Matrix3d angular_velocity_cov{};              ///< Covariance matrix of angular velocity
    Eigen::Matrix3d linear_acceleration_cov{};           ///< Covariance matrix of linear acceleration
    Eigen::Matrix3d angular_velocity_bias_cov{};         ///< Covariance matrix of angular velocity bias
    Eigen::Matrix3d linear_acceleration_bias_cov{};      ///< Covariance matrix of linear acceleration bias
};

/**
 * @struct ForceTorqueMeasurement
 * @brief Represents force-torque sensor measurements including force, center of pressure (COP),
 * and optional torque
 */
struct ForceTorqueMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};      ///< Force measured by force-torque sensor
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};        ///< Center of pressure (COP) measured by force-torque sensor
    std::optional<Eigen::Vector3d> torque;              ///< Optional torque measured by force-torque sensor
};

/**
 * @struct GroundReactionForceMeasurement
 * @brief Represents ground reaction force measurements including force and center of pressure (COP)
 */
struct GroundReactionForceMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};      ///< Ground reaction force
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};        ///< Center of pressure (COP)
};

/**
 * @struct KinematicMeasurement
 * @brief Represents kinematic measurements including contact status, position, orientation,
 * and other dynamics-related quantities
 */
struct KinematicMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement
    std::map<std::string, bool> contacts_status;        ///< Map of contact status for different parts
    std::map<std::string, double> contacts_probability; ///< Map of contact probabilities
    std::map<std::string, Eigen::Vector3d> contacts_position;                    ///< Map of contact positions
    std::map<std::string, Eigen::Matrix3d> contacts_position_noise;              ///< Map of contact position noise covariances
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation;        ///< Optional map of contact orientations
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_noise;      ///< Optional map of contact orientation noise covariances
    Eigen::Vector3d com_angular_momentum_derivative{Eigen::Vector3d::Zero()};     ///< Derivative of center of mass (COM) angular momentum
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};                        ///< Center of mass (COM) position
    Eigen::Vector3d com_linear_acceleration{Eigen::Vector3d::Zero()};              ///< Center of mass (COM) linear acceleration
    Eigen::Matrix3d position_slip_cov{Eigen::Matrix3d::Identity()};                ///< Covariance of position slip
    Eigen::Matrix3d orientation_slip_cov{Eigen::Matrix3d::Identity()};             ///< Covariance of orientation slip
    Eigen::Matrix3d position_cov{Eigen::Matrix3d::Identity()};                     ///< Covariance of position
    Eigen::Matrix3d orientation_cov{Eigen::Matrix3d::Identity()};                  ///< Covariance of orientation
    Eigen::Matrix3d com_position_process_cov{Eigen::Matrix3d::Identity()};         ///< Covariance of COM position process noise
    Eigen::Matrix3d com_linear_velocity_process_cov{Eigen::Matrix3d::Identity()};  ///< Covariance of COM linear velocity process noise
    Eigen::Matrix3d external_forces_process_cov{Eigen::Matrix3d::Identity()};       ///< Covariance of external forces process noise
    Eigen::Matrix3d com_position_cov{Eigen::Matrix3d::Identity()};                 ///< Covariance of COM position
    Eigen::Matrix3d com_linear_acceleration_cov{Eigen::Matrix3d::Identity()};       ///< Covariance of COM linear acceleration
};

/**
 * @struct OdometryMeasurement
 * @brief Represents odometry measurements including base position, orientation, and their covariances
 */
struct OdometryMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};      ///< Base position from odometry
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()}; ///< Base orientation from odometry
    Eigen::Matrix3d base_position_cov{Eigen::Matrix3d::Identity()};      ///< Covariance matrix of base position
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Identity()};   ///< Covariance matrix of base orientation
};

/**
 * @struct TerrainMeasurement
 * @brief Represents terrain height measurements at a specific timestamp
 */
struct TerrainMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement
    double height{};                                    ///< Terrain height measurement
    double height_cov{1.0};                             ///< Covariance of terrain height measurement (default 1.0)
    TerrainMeasurement(double timestamp, double height, double height_cov = 1.0)
        :timestamp(timestamp), height(height), height_cov(height_cov) {} ///< Constructor with initialization
};

/**
 * @typedef ContactMeasurement
 * @brief Alias for a contact measurement, typically representing contact forces or pressure
 */
using ContactMeasurement = double;

}  // namespace serow
