/**
 * Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
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
 * @file BaseEKF.hpp
 * @brief Implementation of Extended Kalman Filter (EKF) for state estimation in humanoid robots.
 *        This EKF fuses data from an Inertial Measurement Unit (IMU), base linear velocity
 *        measurement, and optionally external odometry (e.g., Visual Odometry or Lidar
 *        Odometry). It models the state of the robot including base position, velocity,
 * orientation  in world frame coordinates as well as the gyro and accelerometer biases.
 * @author Stylianos Piperakis
 * @details The state estimation involves predicting and updating the robot's state based on sensor
 *          measurements and the robot's dynamics. The EKF integrates outlier detection mechanisms
 *          to improve state estimation robustness, incorporating beta distribution parameters and
 *          digamma function approximations. More information on the nonlinear state estimation for
 *          humanoid robot walking can be found in:
 *          https://www.researchgate.net/publication/326194869_Nonlinear_State_Estimation_for_Humanoid_Robot_Walking
 */

#pragma once

#include <iostream>

#include "Measurement.hpp"  // Includes various sensor measurements
#include "OutlierDetector.hpp"
#include "State.hpp"  // Includes definitions of robot state variables

namespace serow {

/**
 * @class BaseEKF
 * @brief Implements an Extended Kalman Filter (EKF) for state estimation in humanoid robots,
 *        specifically for fusing IMU data, base linear velocity, and external odometry.
 */
class BaseEKF {
public:
    /**
     * @brief Initializes the EKF with the initial robot state and other parameters.
     * @param state Initial state of the robot.
     * @param g Acceleration due to gravity.
     * @param imu_rate IMU update rate.
     * @param outlier_detection Flag indicating if outlier detection mechanisms should be enabled.
     */
    void init(const BaseState& state, double g, double imu_rate, bool outlier_detection = false);

    /**
     * @brief Predicts the robot's state forward based on IMU.
     * @param state Current state of the robot.
     * @param imu IMU measurements.
     */
    void predict(BaseState& state, const ImuMeasurement& imu);

    /**
     * @brief Updates the robot's state based on kinematic measurements and optionally odometry
     *        measurements.
     * @param state Current state of the robot.
     * @param kin Kinematic measurements.
     * @param odom Optional odometry measurements.
     */
    void update(BaseState& state, const KinematicMeasurement& kin,
                std::optional<OdometryMeasurement> odom = std::nullopt);

private:
    int num_states_{};          ///< Number of state variables.
    int num_inputs_{};          ///< Number of input variables.
    bool outlier_detection_{};  ///< Flag indicating if outlier detection is enabled.
    double nominal_dt_{};       ///< Nominal sampling time for prediction step.
    Eigen::Vector3d g_;         ///< Gravity vector.
    // State indices
    Eigen::Array3i v_idx_;   ///< Indices for velocity state variables.
    Eigen::Array3i r_idx_;   ///< Indices for orientation state variables.
    Eigen::Array3i p_idx_;   ///< Indices for position state variables.
    Eigen::Array3i bg_idx_;  ///< Indices for gyro bias state variables.
    Eigen::Array3i ba_idx_;  ///< Indices for accelerometer bias state variables.
    // Input indices
    Eigen::Array3i ng_idx_;                     ///< Indices for IMU input variables.
    Eigen::Array3i na_idx_;                     ///< Indices for kinematic input variables.
    Eigen::Array3i nbg_idx_;                    ///< Indices for gyro bias input variables.
    Eigen::Array3i nba_idx_;                    ///< Indices for accelerometer bias input variables.
    std::optional<double> last_imu_timestamp_;  ///< Timestamp of the last IMU measurement.

    /// Error Covariance, Linearized state transition model, Identity matrix, state uncertainty
    /// matrix 15 x 15
    Eigen::Matrix<double, 15, 15> I_, P_;
    /// Linearized state-input model 15 x 12
    Eigen::Matrix<double, 15, 12> Lc_;

    OutlierDetector contact_outlier_detector;  ///< Outlier detector instance.

    /**
     * @brief Computes discrete dynamics for the prediction step of the EKF.
     * @param state Current state of the robot.
     * @param dt Time step for prediction.
     * @param angular_velocity Angular velocity measurements.
     * @param linear_acceleration Linear acceleration measurements.
     */
    void computeDiscreteDynamics(BaseState& state, double dt,
                                 Eigen::Vector3d angular_velocity,
                                 Eigen::Vector3d linear_acceleration);

    /**
     * @brief Computes Jacobians for the prediction step of the EKF.
     * @param state Current state of the robot.
     * @param angular_velocity Angular velocity measurements.
     * @return Tuple containing prediction Jacobians (state transition and input models).
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computePredictionJacobians(
        const BaseState& state, Eigen::Vector3d angular_velocity);

    void updateWithTwist(BaseState& state, const Eigen::Vector3d& base_linear_velocity,
                         const Eigen::Matrix3d& base_linear_velocity_cov,
                         const Eigen::Quaterniond& base_orientation,
                         const Eigen::Matrix3d& base_orientation_cov);

    /**
     * @brief Updates the robot's state based on odometry measurements.
     * @param state Current state of the robot.
     * @param base_position Position of the robot's base.
     * @param base_orientation Orientation of the robot's base.
     * @param base_position_cov Covariance of base position measurements.
     * @param base_orientation_cov Covariance of base orientation measurements.
     */
     void updateWithOdometry(BaseState& state, const Eigen::Vector3d& base_position,
                             const Eigen::Quaterniond& base_orientation,
                             const Eigen::Matrix3d& base_position_cov,
                             const Eigen::Matrix3d& base_orientation_cov);

    /**
     * @brief Updates the state of the robot with the provided state change and covariance matrix.
     * @param state Current state of the robot (will be updated in-place).
     * @param dx State change vector.
     * @param P Covariance matrix of the state change.
     */
    void updateState(BaseState& state, const Eigen::VectorXd& dx, const Eigen::MatrixXd& P) const;

    /**
     * @brief Updates a copy of the robot's state with the provided state change and covariance
     * matrix.
     * @param state Current state of the robot (will not be modified).
     * @param dx State change vector.
     * @param P Covariance matrix of the state change.
     * @return Updated state after applying the state change.
     */
    BaseState updateStateCopy(const BaseState& state, const Eigen::VectorXd& dx,
                              const Eigen::MatrixXd& P) const;
};

}  // namespace serow
