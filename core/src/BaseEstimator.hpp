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

/**
 * @file BaseEstimator.hpp
 * @brief Abstract interface for base state estimators (EKF variants).
 *        Concrete implementations include ContactEKF and RightInvariantEKF.
 * @author Stylianos Piperakis
 */

#pragma once

#include <memory>
#include <set>
#include <string>

#include "Measurement.hpp"
#include "State.hpp"

namespace serow {

class TerrainElevation;

/**
 * @class BaseEstimator
 * @brief Pure virtual interface for base state estimators that fuse IMU, leg-kinematic,
 *        and optionally external odometry measurements to estimate the robot's base state.
 */
class BaseEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual ~BaseEstimator() = default;

    /**
     * @brief Initializes the estimator with the initial robot state and configuration.
     * @param state Initial state of the robot.
     * @param contacts_frame Set of contact frame names.
     * @param g Acceleration due to gravity.
     * @param imu_rate IMU update rate.
     * @param kin_rate Kinematic update rate.
     * @param eps Minimum contact probability to update the state with kinematics.
     * @param point_feet Flag indicating if the feet are point contacts.
     * @param use_imu_orientation Flag indicating if IMU orientation is used during the update step.
     * @param verbose Flag indicating if verbose output should be enabled.
     */
    virtual void init(const BaseState& state, std::set<std::string> contacts_frame, double g,
                      double imu_rate, double kin_rate, double eps = 0.05, bool point_feet = true,
                      bool use_imu_orientation = false, bool verbose = false) = 0;

    /**
     * @brief Predicts the robot's state forward based on IMU measurements.
     * @param state Current state of the robot.
     * @param imu IMU measurements.
     */
    virtual void predict(BaseState& state, const ImuMeasurement& imu) = 0;

    /**
     * @brief Updates the robot's state based on kinematic measurements (and optionally odometry
     *        and terrain measurements).
     * @param state Current state of the robot.
     * @param imu IMU measurements.
     * @param kin Kinematic measurements.
     * @param odom Optional odometry measurements.
     * @param terrain_estimator Optional terrain elevation mapper.
     */
    virtual void update(BaseState& state, const ImuMeasurement& imu,
                        const KinematicMeasurement& kin,
                        std::optional<OdometryMeasurement> odom = std::nullopt,
                        std::shared_ptr<TerrainElevation> terrain_estimator = nullptr) = 0;

    /**
     * @brief Sets the state of the estimator.
     * @param state The state to set.
     */
    virtual void setState(const BaseState& state) = 0;

    /**
     * @brief Updates the robot's state based on base linear velocity measurements.
     * @param state Current state of the robot.
     * @param base_linear_velocity Base linear velocity in world coordinates.
     * @param base_linear_velocity_cov Spectral density of base linear velocity measurement.
     * @param timestamp Timestamp of the measurement.
     */
    virtual void updateWithBaseLinearVelocity(BaseState& state,
                                              const Eigen::Vector3d& base_linear_velocity,
                                              const Eigen::Matrix3d& base_linear_velocity_cov,
                                              const double timestamp) = 0;

    /**
     * @brief Updates the robot's state based on IMU orientation measurements.
     * @param state Current state of the robot.
     * @param imu_orientation Orientation of the IMU.
     * @param imu_orientation_cov Spectral density of the IMU orientation measurements.
     * @param timestamp Timestamp of the measurement.
     */
    virtual void updateWithIMUOrientation(BaseState& state,
                                          const Eigen::Quaterniond& imu_orientation,
                                          const Eigen::Matrix3d& imu_orientation_cov,
                                          const double timestamp) = 0;
};

}  // namespace serow
