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
    double timestamp{};                     ///< Timestamp of the measurement (s)
    double position{};                      ///< Joint position measurement (rad)
    std::optional<double> velocity;       ///< Optional joint velocity measurement (rad/s)
};

/**
 * @struct ImuMeasurement
 * @brief Represents IMU (Inertial Measurement Unit) measurements including linear acceleration,
 * angular velocity, and their covariances
 */
struct ImuMeasurement {
    double timestamp{};                                  ///< Timestamp of the measurement (s)
    Eigen::Vector3d linear_acceleration{};               ///< Linear acceleration measured by IMU (m/s^2)
    Eigen::Vector3d angular_velocity{};                  ///< Angular velocity measured by IMU (rad/s)
    Eigen::Vector3d angular_acceleration{};              ///< Angular acceleration measured by IMU (rad/s^2)
    Eigen::Matrix3d angular_velocity_cov{};              ///< Covariance matrix of angular velocity (rad^2/s^2)
    Eigen::Matrix3d linear_acceleration_cov{};           ///< Covariance matrix of linear acceleration (m^2/s^4)
    Eigen::Matrix3d angular_velocity_bias_cov{};         ///< Covariance matrix of angular velocity bias (rad^2/s^2)
    Eigen::Matrix3d linear_acceleration_bias_cov{};      ///< Covariance matrix of linear acceleration bias (m^2/s^4)
};

/**
 * @struct ForceTorqueMeasurement
 * @brief Represents force-torque sensor measurements including force, center of pressure (COP),
 * and optional torque
 */
struct ForceTorqueMeasurement {
    double timestamp{};                                  ///< Timestamp of the measurement (s)
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};      ///< Force measured by force-torque sensor (N)
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};        ///< Center of pressure (COP) measured by force-torque sensor (m)
    std::optional<Eigen::Vector3d> torque;               ///< Optional torque measured by force-torque sensor (Nm)
};

/**
 * @struct GroundReactionForceMeasurement
 * @brief Represents ground reaction force measurements including force and center of pressure (COP)
 */
struct GroundReactionForceMeasurement {
    double timestamp{};                                  ///< Timestamp of the measurement (s)
    Eigen::Vector3d force{Eigen::Vector3d::Zero()};      ///< Ground reaction force (N)
    Eigen::Vector3d cop{Eigen::Vector3d::Zero()};        ///< Center of pressure (COP) (m)
};

/**
 * @struct KinematicMeasurement
 * @brief Represents kinematic measurements including contact status, position, orientation,
 * and other dynamics-related quantities
 */
struct KinematicMeasurement {
    double timestamp{};                                                                     ///< Timestamp of the measurement (s)
    Eigen::Vector3d base_linear_velocity{Eigen::Vector3d::Zero()};                          ///< Base linear velocity (m/s)
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()};                    ///< Base orientation (quaternion)
    std::map<std::string, bool> contacts_status;                                            ///< Map of contact status for different parts (0 or 1)
    std::map<std::string, double> contacts_probability;                                     ///< Map of contact probabilities ([0, 1])
    std::map<std::string, Eigen::Vector3d> contacts_position;                               ///< Map of contact positions relative to base frame (m) 
    std::map<std::string, Eigen::Matrix3d> contacts_position_noise;                         ///< Map of contact position noise covariances relative to base frame (m^2)
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation;          ///< Optional map of contact orientations relative to base frame 
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_noise;       ///< Optional map of contact orientation noise covariances relative to base frame (rad^2)
    Eigen::Vector3d com_angular_momentum_derivative{Eigen::Vector3d::Zero()};               ///< Derivative of center of mass (COM) angular momentum (Nm)
    Eigen::Vector3d com_position{Eigen::Vector3d::Zero()};                                  ///< Center of mass (COM) position (m)
    Eigen::Vector3d com_linear_acceleration{Eigen::Vector3d::Zero()};                       ///< Center of mass (COM) linear acceleration (m/s^2)
    Eigen::Matrix3d base_linear_velocity_cov{Eigen::Matrix3d::Identity()};                  ///< Covariance of base linear velocity (m^2/s^2)
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Identity()};                      ///< Covariance of base orientation (rad^2)
    Eigen::Matrix3d position_slip_cov{Eigen::Matrix3d::Identity()};                         ///< Covariance of position slip (m^2)
    Eigen::Matrix3d orientation_slip_cov{Eigen::Matrix3d::Identity()};                      ///< Covariance of orientation slip (rad^2)
    Eigen::Matrix3d position_cov{Eigen::Matrix3d::Identity()};                              ///< Covariance of position (m^2)
    Eigen::Matrix3d orientation_cov{Eigen::Matrix3d::Identity()};                           ///< Covariance of orientation (rad^2)
    Eigen::Matrix3d com_position_process_cov{Eigen::Matrix3d::Identity()};                  ///< Covariance of COM position process noise (m^2)
    Eigen::Matrix3d com_linear_velocity_process_cov{Eigen::Matrix3d::Identity()};           ///< Covariance of COM linear velocity process noise (m^2/s^2)
    Eigen::Matrix3d external_forces_process_cov{Eigen::Matrix3d::Identity()};               ///< Covariance of external forces process noise (N^2)
    Eigen::Matrix3d com_position_cov{Eigen::Matrix3d::Identity()};                          ///< Covariance of COM position (m^2)
    Eigen::Matrix3d com_linear_acceleration_cov{Eigen::Matrix3d::Identity()};               ///< Covariance of COM linear acceleration (m^2/s^4)
};

/**
 * @struct OdometryMeasurement
 * @brief Represents odometry measurements including base position, orientation, and their covariances
 */
struct OdometryMeasurement {
    double timestamp{};                                                  ///< Timestamp of the measurement (s)
    Eigen::Vector3d base_position{Eigen::Vector3d::Zero()};              ///< Base position from odometry (m)
    Eigen::Quaterniond base_orientation{Eigen::Quaterniond::Identity()}; ///< Base orientation from odometry
    Eigen::Matrix3d base_position_cov{Eigen::Matrix3d::Identity()};      ///< Covariance matrix of base position (m^2)
    Eigen::Matrix3d base_orientation_cov{Eigen::Matrix3d::Identity()};   ///< Covariance matrix of base orientation (rad^2)
};

/**
 * @struct TerrainMeasurement
 * @brief Represents terrain height measurements at a specific timestamp
 */
struct TerrainMeasurement {
    double timestamp{};                                 ///< Timestamp of the measurement (s)
    double height{};                                    ///< Terrain height measurement (m)
    double height_cov{1.0};                             ///< Covariance of terrain height measurement (m^2)
    TerrainMeasurement(double timestamp, double height, double height_cov = 1.0)
        :timestamp(timestamp), height(height), height_cov(height_cov) {} 
};

/**
 * @typedef ContactMeasurement
 * @brief Alias for a contact measurement, typically representing contact forces or pressure
 */
using ContactMeasurement = double;

}  // namespace serow
