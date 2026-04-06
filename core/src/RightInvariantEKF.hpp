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
 * @file RightInvariantEKF.hpp
 * @brief Right-Invariant Extended Kalman Filter for state estimation in legged robots.
 *        The filter state lives on SE_2(3) x R^6 and exploits the group-affine property of
 *        IMU-driven dynamics to obtain a state-independent linearized error propagation.
 *        Fuses IMU, leg-kinematic velocity, optional external odometry and terrain height.
 * @author Stylianos Piperakis
 */

#pragma once

#include <iostream>
#include <memory>

#include "BaseEstimator.hpp"
#include "LocalTerrainMapper.hpp"
#include "Measurement.hpp"
#include "OutlierDetector.hpp"
#include "State.hpp"

namespace serow {

/**
 * @class RightInvariantEKF
 * @brief Right-Invariant Extended Kalman Filter on SE_2(3) for legged-robot state estimation,
 *        fusing IMU data, leg-kinematic velocity, and optionally external odometry.
 */
class RightInvariantEKF : public BaseEstimator {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void init(const BaseState& state, std::set<std::string> contacts_frame, const double g,
              const double imu_rate, const double kin_rate, const double eps = 0.05,
              const bool point_feet = true, const bool use_imu_orientation = false,
              const bool verbose = false) override;

    void predict(BaseState& state, const ImuMeasurement& imu) override;

    void update(BaseState& state, const ImuMeasurement& imu, const KinematicMeasurement& kin,
                std::optional<OdometryMeasurement> odom = std::nullopt,
                std::shared_ptr<TerrainElevation> terrain_estimator = nullptr) override;

    void setState(const BaseState& state) override;

    void updateWithBaseLinearVelocity(BaseState& state, const Eigen::Vector3d& base_linear_velocity,
                                      const Eigen::Matrix3d& base_linear_velocity_cov,
                                      const double timestamp) override;

    void updateWithIMUOrientation(BaseState& state, const Eigen::Quaterniond& imu_orientation,
                                  const Eigen::Matrix3d& imu_orientation_cov,
                                  const double timestamp) override;

private:
    int num_states_{};                      ///< Number of state variables.
    int num_inputs_{};                      ///< Number of input variables.
    int num_leg_end_effectors_{};           ///< Number of leg end-effectors.
    std::set<std::string> contacts_frame_;  ///< Set of contact frame names.
    double nominal_imu_dt_{};               ///< Nominal sampling time for prediction step.
    double nominal_kin_dt_{};               ///< Nominal sampling time for kinematic update step.
    Eigen::Vector3d g_;                     ///< Gravity vector.
    double eps_{0.05};  ///< Minimum contact probability to update the state with kinematics.
    // State indices
    Eigen::Array3i v_idx_;   ///< Indices for velocity state variables.
    Eigen::Array3i r_idx_;   ///< Indices for orientation state variables.
    Eigen::Array3i p_idx_;   ///< Indices for position state variables.
    Eigen::Array3i bg_idx_;  ///< Indices for gyro bias state variables.
    Eigen::Array3i ba_idx_;  ///< Indices for accelerometer bias state variables.

    // Input indices
    Eigen::Array3i ng_idx_;   ///< Indices for gyro input variables.
    Eigen::Array3i na_idx_;   ///< Indices for acceleration input variables.
    Eigen::Array3i nbg_idx_;  ///< Indices for gyro bias input variables.
    Eigen::Array3i nba_idx_;  ///< Indices for accelerometer bias input variables.
    std::optional<double> last_imu_predict_timestamp_;  ///< Timestamp of the last IMU measurement
                                                        ///< used in the predict step.
    std::optional<double> last_kin_update_timestamp_;   ///< Timestamp of the last kinematic
                                                        ///< measurement used in the update step.
    std::optional<double> last_imu_update_timestamp_;   ///< Timestamp of the last IMU measurement
                                                        ///< used in the update step.
    std::optional<double> last_terrain_update_timestamp_;  ///< Timestamp of the last terrain
                                                           ///< measurement used in the update step.

    /// Error Covariance, Linearized state transition model, Identity matrix, state uncertainty
    /// matrix 15 x 15
    Eigen::Matrix<double, 15, 15> I_, P_;
    /// Constant (state-independent) part of the continuous-time A matrix
    Eigen::Matrix<double, 15, 15> Ac_;
    /// Constant part of the noise-input Jacobian L (bias-drift rows only)
    Eigen::Matrix<double, 15, 12> Lc_;

    OutlierDetector base_position_outlier_detector;  ///< Outlier detector instance.

    bool point_feet_{};           ///< Flag indicating if the robot has point feet.
    bool verbose_{};              ///< Flag indicating if verbose output is enabled.
    bool use_imu_orientation_{};  ///< Flag indicating if IMU orientation is used during the update
                                  ///< step.

    std::optional<Eigen::Vector3d>
        first_odometry_position_;  ///< Initial odometry measurement position (world coordinates).
    std::optional<Eigen::Quaterniond>
        first_odometry_orientation_;  ///< Initial odometry measurement orientation (world
                                      ///< coordinates).

    /**
     * @brief Computes discrete dynamics for the prediction step of the EKF.
     * @param state Current state of the robot.
     * @param dt Time step for prediction.
     * @param angular_velocity Angular velocity measurements.
     * @param linear_acceleration Linear acceleration measurements.
     */
    void computeDiscreteDynamics(BaseState& state, double dt, Eigen::Vector3d angular_velocity,
                                 Eigen::Vector3d linear_acceleration);

    /**
     * @brief Computes Jacobians for the prediction step of the EKF.
     * @param state Current state of the robot.
     * @param angular_velocity Angular velocity measurements.
     * @return Tuple containing prediction Jacobians (state transition and input models).
     */
    std::tuple<Eigen::Matrix<double, 15, 15>, Eigen::Matrix<double, 15, 12>>
    computePredictionJacobians(const BaseState& state);

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
     * @brief Updates the robot's state based on terrain measurements.
     * @param state Current state of the robot.
     * @param contacts_position Positions of leg contacts.
     * @param contacts_position_noise Covariances of leg contact positions.
     * @param contacts_probability Probabilities of leg contacts.
     * @param timestamp Timestamp of the terrain measurement.
     * @param terrain_estimator Terrain elevation mapper.
     */
    void updateWithTerrain(BaseState& state,
                           const std::map<std::string, Eigen::Vector3d>& contacts_position,
                           const std::map<std::string, Eigen::Matrix3d>& contacts_position_noise,
                           const std::map<std::string, double>& contacts_probability,
                           const double timestamp,
                           std::shared_ptr<TerrainElevation> terrain_estimator);

    /**
     * @brief Updates the state of the robot with the provided state change and covariance
     * matrix.
     * @param state Current state of the robot (will be updated in-place).
     * @param dx State change vector.
     * @param P Covariance matrix of the state change.
     */
    void updateState(BaseState& state, const Eigen::Matrix<double, 15, 1>& dx,
                     const Eigen::Matrix<double, 15, 15>& P) const;

    /**
     * @brief Updates a copy of the robot's state with the provided state change and covariance
     * matrix.
     * @param state Current state of the robot (will not be modified).
     * @param dx State change vector.
     * @param P Covariance matrix of the state change.
     * @return Updated state after applying the state change.
     */
    BaseState updateStateCopy(const BaseState& state, const Eigen::Matrix<double, 15, 1>& dx,
                              const Eigen::Matrix<double, 15, 15>& P) const;
};

}  // namespace serow
