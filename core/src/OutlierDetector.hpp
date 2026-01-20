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

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif
#include <iostream>

namespace serow {

/**
 * @struct OutlierDetector
 * @brief Implements outlier detection mechanisms using beta distribution parameters for state
 *        estimation in humanoid robots.
 */
struct OutlierDetector {
    double zeta =
        1.0;  ///< Parameter for outlier detection:
              ///< https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double f_0 =
        0.1;  ///< Parameter for outlier detection:
              ///< https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double e_0 =
        0.9;  ///< Parameter for outlier detection:
              ///< https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double f_t =
        0.1;  ///< Parameter for outlier detection:
              ///< https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double e_t =
        0.9;  ///< Parameter for outlier detection:
              ///< https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double threshold = 1e-5;
    size_t iters = 4;  ///< Number of iterations for outlier detection

    /**
     * @brief Computes the digamma function approximation.
     * @param x Argument for digamma function.
     * @return Computed value of digamma function.
     */
    double computePsi(double x);

    /**
     * @brief Initializes the outlier detection process.
     */
    void init();

    /**
     * @brief Estimates the outlier indicator zeta based on provided matrices.
     * @param BetaT Matrix used in outlier estimation.
     * @param R Matrix used in outlier estimation.
     */
    void estimate(const Eigen::Matrix3d& BetaT, const Eigen::Matrix3d& R);
};

}  // namespace serow
