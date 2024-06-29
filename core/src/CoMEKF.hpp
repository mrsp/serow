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
 * @brief Nonlinear CoM Estimation based on encoder, force/torque or pressure, and IMU measurements
 * @author Stylianos Piperakis
 * @details Estimates the 3D CoM position, velocity and external forces. More info in Nonlinear
 State Estimation for Humanoid Robot Walking
 https://www.researchgate.net/publication/326194869_Nonlinear_State_Estimation_for_Humanoid_Robot_Walking

 */
#pragma once

#include "Measurement.hpp"
#include "State.hpp"

namespace serow {

class CoMEKF {
   private:
    /// Previous ground reaction force timestamp, used in predict step
    std::optional<double> last_grf_timestamp_;

    /// State covariance, Identity matrices
    Eigen::Matrix<double, 9, 9> P_;
    Eigen::Matrix<double, 9, 9> I_ = Eigen::Matrix<double, 9, 9>::Identity();

    /// nominal F/T sampling period, robot's mass and gravity constant
    double nominal_dt_{};
    double mass_{};
    double g_{};

    /// state indices: com position, com linear velocity, external force
    Eigen::Array3i c_idx_;
    Eigen::Array3i v_idx_;
    Eigen::Array3i f_idx_;

    /// @brief Computes the nonlinear CoM dynamics
    /// @param state the EKF state to be used in the computation
    /// @param cop_position the position of the COP in world coordinates
    /// @param ground_reaction_force the total ground reaction force in world coordinates
    /// @param com_angular_momentum_derivative the angular momentum rate around the CoM in world coordinates
    /// @return the EKF state derivative
    Eigen::Matrix<double, 9, 1> computeContinuousDynamics(
        const CentroidalState& state, const Eigen::Vector3d& cop_position,
        const Eigen::Vector3d& ground_reaction_force,
        const Eigen::Vector3d& com_angular_momentum_derivative);
        
    /// @brief Computes the linearized state dynamics
    /// @param state the EKF state to be used for the linearization
    /// @param cop_position the position of the COP in world coordinates
    /// @param ground_reaction_force the total ground reaction force in world coordinates
    /// @param com_angular_momentum_derivative  the angular momentum rate around the CoM in world coordinates
    /// @return The linearized state transition matrix and the linearized state-input noise matrix
    std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>> computePredictionJacobians(
        const CentroidalState& state, const Eigen::Vector3d& cop_position,
        const Eigen::Vector3d& ground_reaction_force,
        const Eigen::Vector3d& com_angular_momentum_derivative);
    
    /// @brief Performs the EKF update step with a CoM linear acceleration measurement
    /// @param state the EKF state used in the computation
    /// @param com_linear_acceleration the CoM linear acceleration in world coordinates
    /// @param cop_position the COP position in world coordinates
    /// @param ground_reaction_force the total ground reaction force in world coordinates
    /// @param com_linear_acceleration_cov the CoM linear acceleration covariance in world coordinates
    /// @param com_angular_momentum_derivative the angular momentum rate around the CoM in world coordinates
    /// @return The updated EKF state
    CentroidalState updateWithCoMAcceleration(
        const CentroidalState& state, const Eigen::Vector3d& com_linear_acceleration,
        const Eigen::Vector3d& cop_position, const Eigen::Vector3d& ground_reaction_force,
        const Eigen::Matrix3d& com_linear_acceleration_cov,
        const Eigen::Vector3d& com_angular_momentum_derivative);

    /// @brief Performs the EKF update step with a CoM position measurement
    /// @param state the EKF state used in the computation
    /// @param com_position the CoM position in world coordinates
    /// @param com_position_cov the CoM position covariance in world coordinates
    /// @return The updated EKF state
    CentroidalState updateWithCoMPosition(const CentroidalState& state,
                                          const Eigen::Vector3d& com_position,
                                          const Eigen::Matrix3d& com_position_cov);

    /// @brief Updates the EKF state 
    /// @param state the EKF state to be updated
    /// @param dx the state correction vector
    /// @param P the EKF state covariance
    void updateState(CentroidalState& state, const Eigen::Matrix<double, 9, 1>& dx,
                     const Eigen::Matrix<double, 9, 9>& P) const;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /// @brief Initializes the 3D CoM estimator
    /// @param state the initial state of the EKF
    /// @param mass the robot's mass
    /// @param g the gravity constant
    /// @param rate the nominal rate, corresponds to the F/T rate and the rate the predict step is realized
    void init(const CentroidalState& state, double mass, double g, double rate);

    /// @brief Realizes the EKF predict step
    /// @param state the EKF state used in prediction
    /// @param kin the KinematicMeasurement used in the computation
    /// @param grf the GroundReactionForceMeasurement used in the computation
    /// @return The predicted EKF state
    CentroidalState predict(const CentroidalState& state, const KinematicMeasurement& kin,
                            const GroundReactionForceMeasurement& grf);

    /// @brief Realizes the EKF update step with a CoM position measurement
    /// @param state the EKF state used for the update
    /// @param kin the KinematicMeasurement used in the computation
    /// @return The updated EKF state
    CentroidalState updateWithKinematics(const CentroidalState& state,
                                         const KinematicMeasurement& kin);

    /// @brief Realizes the EKF update step with a CoM linear acceleration measurement
    /// @param state the EKF state used for the update
    /// @param kin the KinematicMeasurement used in the computation
    /// @param grf the GroundReactionForceMeasurement used in the computation
    /// @return The updated EKF state
    CentroidalState updateWithImu(const CentroidalState& state, const KinematicMeasurement& kin,
                                  const GroundReactionForceMeasurement& grf);
};

}  // namespace serow
