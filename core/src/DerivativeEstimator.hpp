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
 * @brief Signal derivative estimation with numerical differentiation and 2nd order Low Pass
 * Butterworth Filter
 * @author Stylianos Piperakis
 * @details estimates the measurement's derivative
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

class DerivativeEstimator {
   private:
    /// 2nd order butterworth filter to smooth the signal
    std::vector<ButterworthLPF> bw_;
    /// linear differentiator filter to compute the derivative
    std::vector<Differentiator> df_;

   public:
    /// name of the estimator e.g. "com-angular-momentum"
    std::string name_{};

    /// dimensions of the signal
    size_t dim_{};

    /// Filtered signal
    Eigen::VectorXd x_;
    /// Signal derivative
    Eigen::VectorXd x_dot_;

    /// @brief Estimates the derivative of a measurement
    /// @param measurement the signal to estimate the derivative of
    /// @return The signal's derivative
    Eigen::VectorXd filter(const Eigen::VectorXd& measurement);

    /// @brief resets the estimator
    /// @param verbose whether or not to print debug messages
    void reset(bool verbose = false);

    /// @brief Initializes the derivative estimator
    /// @param name name of the estimator e.g. "com-angular-momentum"
    /// @param f_sampling the sampling frequency of the signal e.g. 100hz
    /// @param f_cutoff the cut-off frequency of the low pass filter e.g. 10hz
    /// @param dim dimensions of the signal e.g. 3D
    /// @param verbose whether or not to print debug messages
    DerivativeEstimator(const std::string& name, double f_sampling, double f_cutoff, size_t dim,
                        bool verbose = false);
    
    DerivativeEstimator() = default;
};

}  // namespace serow
