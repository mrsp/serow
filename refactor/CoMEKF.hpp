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
    
    State updateWithImu(State state, Eigen::Vector3d Acc, Eigen::Vector3d Pos, Eigen::Vector3d Gyro,
                        Eigen::Vector3d Gyrodot, Eigen::Vector3d COP, Eigen::Vector3d fN,
                        Eigen::Matrix3d com_linear_acceleration_cov,
                        std::optional<Eigen::Vector3d> Ldot);
    State updateWithKinematics(State state, Eigen::Vector3d com_position,
                               Eigen::Matrix3d com_position_cov);
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    void init(double mass, double I_xx, double I_yy, double rate);

    State predict(State state, KinematicMeasurement kin, GroundReactionForceMeasurement grf);

    State update(State state, KinematicMeasurement kin, GroundReactionForceMeasurement grf,
                 ImuMeasurement imu);
};
