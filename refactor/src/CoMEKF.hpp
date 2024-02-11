/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
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

    // Compute the nonlinear dynamics
    Eigen::Matrix<double, 9, 1> computeContinuousDynamics(
        const State& state, const Eigen::Vector3d& cop_position,
        const Eigen::Vector3d& ground_reaction_force,
        std::optional<Eigen::Vector3d> com_angular_momentum_derivative);
    // Compute Linearized matrices
    std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>> computePredictionJacobians(
        const State& state, const Eigen::Vector3d& cop_position,
        const Eigen::Vector3d& ground_reaction_force,
        std::optional<Eigen::Vector3d> com_angular_momentum_derivative);

    State updateWithCoMAcceleration(const State& state,
                                    const Eigen::Vector3d& com_linear_acceleration,
                                    const Eigen::Vector3d& cop_position,
                                    const Eigen::Vector3d& ground_reaction_force,
                                    const Eigen::Matrix3d& com_linear_acceleration_cov,
                                    std::optional<Eigen::Vector3d> com_angular_momentum_derivative);

    State updateWithCoMPosition(const State& state, const Eigen::Vector3d& com_position,
                                const Eigen::Matrix3d& com_position_cov);

    void updateState(State& state, const Eigen::Matrix<double, 9, 1>& dx,
                     const Eigen::Matrix<double, 9, 9>& P) const;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void init(const State& state, double mass, double rate);

    State predict(const State& state, const KinematicMeasurement& kin,
                  const GroundReactionForceMeasurement& grf);

    State updateWithKinematics(const State& state, const KinematicMeasurement& kin);

    State updateWithImu(const State& state, const KinematicMeasurement& kin,
                        const GroundReactionForceMeasurement& grf);
};

}  // namespace serow
