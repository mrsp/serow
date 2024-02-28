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
    // Previous grf timestamp
    std::optional<double> last_grf_timestamp_;

    // Error-Covariance, Identity matrices
    Eigen::Matrix<double, 9, 9> P_, I_;

    // nominal F/T time, robot's mass and gravity constant
    double nominal_dt_{}, mass_{}, g_{};

    // state indices, com position, com linear velocity, external force
    Eigen::Array3i c_idx_, v_idx_, f_idx_;

    // Compute the nonlinear dynamics
    Eigen::Matrix<double, 9, 1> computeContinuousDynamics(
        const CentroidalState& state, const Eigen::Vector3d& cop_position,
        const Eigen::Vector3d& ground_reaction_force,
        const Eigen::Vector3d& com_angular_momentum_derivative);
    // Compute Linearized matrices
    std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>> computePredictionJacobians(
        const CentroidalState& state, const Eigen::Vector3d& cop_position,
        const Eigen::Vector3d& ground_reaction_force,
        const Eigen::Vector3d& com_angular_momentum_derivative);

    CentroidalState updateWithCoMAcceleration(
        const CentroidalState& state, const Eigen::Vector3d& com_linear_acceleration,
        const Eigen::Vector3d& cop_position, const Eigen::Vector3d& ground_reaction_force,
        const Eigen::Matrix3d& com_linear_acceleration_cov,
        const Eigen::Vector3d& com_angular_momentum_derivative);

    CentroidalState updateWithCoMPosition(const CentroidalState& state,
                                          const Eigen::Vector3d& com_position,
                                          const Eigen::Matrix3d& com_position_cov);

    void updateState(CentroidalState& state, const Eigen::Matrix<double, 9, 1>& dx,
                     const Eigen::Matrix<double, 9, 9>& P) const;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void init(const CentroidalState& state, double mass, double g, double rate);

    CentroidalState predict(const CentroidalState& state, const KinematicMeasurement& kin,
                            const GroundReactionForceMeasurement& grf);

    CentroidalState updateWithKinematics(const CentroidalState& state,
                                         const KinematicMeasurement& kin);

    CentroidalState updateWithImu(const CentroidalState& state, const KinematicMeasurement& kin,
                                  const GroundReactionForceMeasurement& grf);
};

}  // namespace serow
