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
#include "State.hpp"

class CoMEKF {
   private:
    // Error-Covariance, Identity matrices
    Eigen::Matrix<double, 9, 9> P_, I_;
    
    // nominal sampling time, robot's mass, gravity constant and inertia around the x and y axes
    double nominal_dt_{}, mass_{}, g_{}, I_xx_{}, I_yy_{};
    
    // Compute the nonlinear dynamics
    Eigen::Matrix<double, 9, 1> computeContinuousDynamics(State state, Eigen::Vector3d COP,
                                                          Eigen::Vector3d fN,
                                                          std::optional<Eigen::Vector3d> Ldot);
    // Compute Linearized matrices
    std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>> computePredictionJacobians(
        State state, Eigen::Vector3d COP, Eigen::Vector3d fN, std::optional<Eigen::Vector3d> Ldot);

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    /** @fn void init()
     *   @brief initializes the nonlinear 3D CoM estimator
     */
    void init(double mass, double I_xx, double I_yy, double rate);

    State predict(State state, KinematicMeasurement kin, GroundReactionForceMeasurement grf);
    /** @fn update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot);
     *   @brief realizes the update step of the EKF
     *   @param Acc  3D Body acceleration as measured with an IMU
     *   @param Pos  3D CoM position as measured with encoders
     *   @param Gyro 3D Base angular velocity as measured with an IMU
     *   @param Gyrodot 3D Base angular acceleration as derived from an IMU
     */
    // void update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot);
};
