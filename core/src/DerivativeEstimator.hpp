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
 * @file DerivativeEstimator.hpp
 * @brief Header file for the DerivativeEstimator class.
 * @details Provides a mechanism to estimate the derivative of a signal using numerical
 * differentiation and a 2nd order Low Pass Butterworth Filter.
 */

#pragma once

#include "ButterworthLPF.hpp"
#include "Differentiator.hpp"

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

namespace serow {

/**
 * @class DerivativeEstimator
 * @brief Class for estimating the derivative of a signal using numerical differentiation and a 2nd
 * order Low Pass Butterworth Filter.
 */
class DerivativeEstimator {
   private:
    /// 2nd order butterworth filter to smooth the signal
    std::vector<ButterworthLPF> bw_;
    /// linear differentiator filter to compute the derivative
    std::vector<Differentiator> df_;

   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// name of the estimator e.g. "com-angular-momentum"
    std::string name_{};

    /// dimensions of the signal
    size_t dim_{};

    /// Filtered signal
    Eigen::VectorXd x_;
    /// Signal derivative
    Eigen::VectorXd x_dot_;

    /**
     * @brief Estimates the derivative of a measurement.
     * @param measurement The signal to estimate the derivative of.
     * @return The signal's derivative.
     */
    Eigen::VectorXd filter(const Eigen::VectorXd& measurement);

    /**
     * @brief Resets the estimator.
     * @param verbose Whether or not to print debug messages.
     */
    void reset(bool verbose = false);

    /**
     * @brief Initializes the derivative estimator.
     * @param name Name of the estimator e.g. "com-angular-momentum".
     * @param f_sampling The sampling frequency of the signal e.g. 100Hz.
     * @param f_cutoff The cut-off frequency of the low pass filter e.g. 10Hz.
     * @param dim Dimensions of the signal e.g. 3.
     * @param verbose Whether or not to print debug messages.
     */
    DerivativeEstimator(const std::string& name, double f_sampling, double f_cutoff, size_t dim,
                        bool verbose = false);

    /**
     * @brief Default constructor.
     */
    DerivativeEstimator() = default;

    /**
     * @brief Sets the state of the estimator.
     * @param x The signal.
     * @param x_dot The signal's derivative.
     */
    void setState(const Eigen::VectorXd& x, const Eigen::VectorXd& x_dot);
};

}  // namespace serow
