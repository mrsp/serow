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
 * @file ContactEKF.hpp
 * @brief Implementation of Extended Kalman Filter (EKF) for state estimation in humanoid robots.
 *        This EKF fuses data from an Inertial Measurement Unit (IMU), relative to the base leg
 *        contact measurements, and optionally external odometry (e.g., Visual Odometry or Lidar
 *        Odometry). It models the state of the robot including base position, velocity,
 * orientation, gyro and accelerometer biases, leg contact positions, and contact orientations in
 * world frame coordinates.
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
#include <memory>

#include "LocalTerrainMapper.hpp"
#include "Measurement.hpp"  // Includes various sensor measurements
#include "OutlierDetector.hpp"
#include "State.hpp"  // Includes definitions of robot state variables

namespace serow {

/**
 * @class ContactEKF
 * @brief Implements an Extended Kalman Filter (EKF) for state estimation in humanoid robots,
 *        specifically for fusing IMU data, base leg contact measurements, and external odometry.
 */
class ContactEKF {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /**
     * @brief Initializes the EKF with the initial robot state, contact frames, and other
     * parameters.
     * @param state Initial state of the robot.
     * @param contacts_frame Set of contact frame names.
     * @param point_feet Flag indicating if the robot has point feet.
     * @param g Acceleration due to gravity.
     * @param imu_rate IMU update rate.
     * @param outlier_detection Flag indicating if outlier detection mechanisms should be enabled.
     */
    void init(const BaseState& state, std::set<std::string> contacts_frame, bool point_feet,
              double g, double imu_rate, bool outlier_detection = false);
    /**
     * @brief Predicts the robot's state forward based on IMU and kinematic measurements.
     * @param state Current state of the robot.
     * @param imu IMU measurements.
     * @param kin Kinematic measurements.
     */
    void predict(BaseState& state, const ImuMeasurement& imu, const KinematicMeasurement& kin);

    /**
     * @brief Updates the robot's state based on kinematic measurements (and optionally odometry
     *        and terrain measurements).
     * @param state Current state of the robot.
     * @param kin Kinematic measurements.
     * @param odom Optional odometry measurements.
     * @param terrain Optional terrain measurements.
     */
    void update(BaseState& state, const ImuMeasurement& imu, const KinematicMeasurement& kin,
                std::optional<OdometryMeasurement> odom = std::nullopt,
                std::shared_ptr<TerrainElevation> terrain_estimator = nullptr);

    /**
     * @brief Sets the state of the EKF.
     * @param state The state to set.
     */
    void setState(const BaseState& state);

    /**
     * @brief Updates the robot's state based on contact position measurements.
     * @param state Current state of the robot.
     * @param cf Contact frame name.
     * @param cs Leg contact status.
     * @param cp Leg contact position.
     * @param cp_noise Covariance of leg contact position measurement.
     * @param position_cov Covariance of position measurements.
     * @param terrain_estimator Terrain elevation estimator.
     */
    void updateWithContactPosition(BaseState& state, const std::string& cf, const bool cs,
                                   const Eigen::Vector3d& cp, Eigen::Matrix3d cp_noise,
                                   const Eigen::Matrix3d& position_cov,
                                   std::shared_ptr<TerrainElevation> terrain_estimator);

    /**
     * @brief Updates the robot's state based on IMU orientation measurements.
     * @param state Current state of the robot.
     * @param imu_orientation Orientation of the IMU.
     * @param imu_orientation_cov Covariance of the IMU orientation measurements.
     */
    void updateWithIMUOrientation(BaseState& state, const Eigen::Quaterniond& imu_orientation,
                                  const Eigen::Matrix3d& imu_orientation_cov);

    /**
     * @brief Sets the action for the contact estimator
     * @param cf Contact frame name
     * @param action Action
     */
    void setAction(const std::string& cf, const Eigen::VectorXd& action);

    /**
     * @brief Clears the action covariance gain matrix
     */
    void clearAction();

    /**
     * @brief Gets the contact position innovation
     * @param contact_frame Contact frame name
     * @param innovation Contact position innovation
     * @param covariance Contact position covariance
     */
    bool getContactPositionInnovation(const std::string& contact_frame, Eigen::Vector3d& innovation,
                                      Eigen::Matrix3d& covariance) const;

    /**
     * @brief Gets the contact orientation innovation
     * @param contact_frame Contact frame name
     * @param innovation Contact orientation innovation
     * @param covariance Contact orientation covariance
     */
    bool getContactOrientationInnovation(const std::string& contact_frame,
                                         Eigen::Vector3d& innovation,
                                         Eigen::Matrix3d& covariance) const;

private:
    int num_states_{};                      ///< Number of state variables.
    int num_inputs_{};                      ///< Number of input variables.
    int contact_dim_{};                     ///< Dimension of contact-related variables.
    int num_leg_end_effectors_{};           ///< Number of leg end-effectors.
    std::set<std::string> contacts_frame_;  ///< Set of contact frame names.
    bool point_feet_{};                     ///< Flag indicating if the robot has point feet.
    bool outlier_detection_{};              ///< Flag indicating if outlier detection is enabled.
    double nominal_dt_{};                   ///< Nominal sampling time for prediction step.
    Eigen::Vector3d g_;                     ///< Gravity vector.
    // State indices
    Eigen::Array3i v_idx_;   ///< Indices for velocity state variables.
    Eigen::Array3i r_idx_;   ///< Indices for orientation state variables.
    Eigen::Array3i p_idx_;   ///< Indices for position state variables.
    Eigen::Array3i bg_idx_;  ///< Indices for gyro bias state variables.
    Eigen::Array3i ba_idx_;  ///< Indices for accelerometer bias state variables.
    std::map<std::string, Eigen::Array3i> pl_idx_;  ///< Indices for leg contact position variables.
    std::map<std::string, Eigen::Array3i>
        rl_idx_;  ///< Indices for leg contact orientation variables.
    // Input indices
    Eigen::Array3i ng_idx_;   ///< Indices for IMU input variables.
    Eigen::Array3i na_idx_;   ///< Indices for kinematic input variables.
    Eigen::Array3i nbg_idx_;  ///< Indices for gyro bias input variables.
    Eigen::Array3i nba_idx_;  ///< Indices for accelerometer bias input variables.
    std::map<std::string, Eigen::Array3i>
        npl_idx_;  ///< Indices for updated leg contact position variables.
    std::map<std::string, Eigen::Array3i>
        nrl_idx_;  ///< Indices for updated leg contact orientation variables.
    std::optional<double> last_imu_timestamp_;  ///< Timestamp of the last IMU measurement.

    /// Error Covariance, Linearized state transition model, Identity matrix, state uncertainty
    /// matrix 15 + 6N x 15 + 6N
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> I_, P_;
    /// Linearized state-input model 15 + 6N x 12 + 6N
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Lc_;

    OutlierDetector contact_outlier_detector;  ///< Outlier detector instance.

    std::map<std::string, double> contact_position_action_cov_gain_;
    std::map<std::string, double> contact_orientation_action_cov_gain_;

    std::map<std::string, std::pair<Eigen::Vector3d, Eigen::Matrix3d>> contact_position_innovation_;
    std::map<std::string, std::pair<Eigen::Vector3d, Eigen::Matrix3d>>
        contact_orientation_innovation_;

    /**
     * @brief Computes discrete dynamics for the prediction step of the EKF.
     * @param state Current state of the robot.
     * @param dt Time step for prediction.
     * @param angular_velocity Angular velocity measurements.
     * @param linear_acceleration Linear acceleration measurements.
     * @param contacts_status Status of leg contacts.
     * @param contacts_position Position of leg contacts.
     * @param contacts_orientations Orientations of leg contacts.
     */
    void computeDiscreteDynamics(
        BaseState& state, double dt, Eigen::Vector3d angular_velocity,
        Eigen::Vector3d linear_acceleration,
        std::optional<std::map<std::string, bool>> contacts_status,
        std::optional<std::map<std::string, Eigen::Vector3d>> contacts_position,
        std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientations =
            std::nullopt);

    /**
     * @brief Computes Jacobians for the prediction step of the EKF.
     * @param state Current state of the robot.
     * @param angular_velocity Angular velocity measurements.
     * @return Tuple containing prediction Jacobians (state transition and input models).
     */
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computePredictionJacobians(
        const BaseState& state, Eigen::Vector3d angular_velocity);

    /**
     * @brief Updates the robot's state based on contact-related measurements.
     * @param state Current state of the robot.
     * @param contacts_position Positions of leg contacts.
     * @param contacts_position_noise Noise in position measurements.
     * @param contacts_status Status of leg contacts.
     * @param position_cov Covariance of position measurements.
     * @param contacts_orientation Orientations of leg contacts (optional).
     * @param contacts_orientation_noise Noise in orientation measurements (optional).
     * @param orientation_cov Covariance of orientation measurements (optional).
     * @param terrain_estimator Terrain elevation estimator (optional).
     */
    void updateWithContacts(
        BaseState& state, const std::map<std::string, Eigen::Vector3d>& contacts_position,
        std::map<std::string, Eigen::Matrix3d> contacts_position_noise,
        const std::map<std::string, bool>& contacts_status, const Eigen::Matrix3d& position_cov,
        std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation,
        std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_noise,
        std::optional<Eigen::Matrix3d> orientation_cov,
        std::shared_ptr<TerrainElevation> terrain_estimator);

    /**
     * @brief Updates the robot's state based on contact orientation measurements.
     * @param state Current state of the robot.
     * @param cf Contact frame name.
     * @param cs Leg contact status.
     * @param co Leg contact orientation.
     * @param co_noise Covariance of leg contact orientation measurement.
     * @param orientation_cov Covariance of orientation measurements.
     */
    void updateWithContactOrientation(BaseState& state, const std::string& cf, const bool cs,
                                      const Eigen::Quaterniond& co, Eigen::Matrix3d co_noise,
                                      const Eigen::Matrix3d& orientation_cov);

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
     * @param contacts_status Status of leg contacts.
     * @param terrain_estimator Terrain elevation mapper.
     */
    void updateWithTerrain(BaseState& state, const std::map<std::string, bool>& contacts_status,
                           std::shared_ptr<TerrainElevation> terrain_estimator);

    /**
     * @brief Updates the state of the robot with the provided state change and covariance
     * matrix.
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
