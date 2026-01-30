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
 * @file DerivativeEstimator.hpp
 * @brief Header file for the DerivativeEstimator class.
 * @details Provides a mechanism to estimate the derivative of a signal using numerical
 * differentiation and a 2nd order Low Pass Butterworth Filter.
 */

#pragma once

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

#include <optional>
#include <vector>
#include <deque>
#include <string>

namespace serow {

/**
 * @brief Computes the coefficients of the Savitzky-Golay numerical differentiation.
 * @param M The number of coefficients to compute.
 * @return The coefficients of the Savitzky-Golay numerical differentiation.
*/
static inline std::vector<double> computeSGCoefficients(const int M) {
    std::vector<double> coeffs;
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(M, 3);
    for (int i = 0; i < M; ++i) {
        double idx = -static_cast<double>(i); // 0, -1, -2...
        J(i, 0) = 1.0;          // Constant term
        J(i, 1) = idx;          // Linear term (t)
        J(i, 2) = idx * idx;    // Quadratic term (t^2)
    }
    
    // Pseudoinverse: (J^T * J)^-1 * J^T
    const Eigen::MatrixXd J_pinv = (J.transpose() * J).ldlt().solve(J.transpose());
    
    // The second row corresponds to the 1st derivative 
    coeffs.resize(M);
    for (int i = 0; i < M; ++i) {
        coeffs[i] = J_pinv(1, i);
    }
    return coeffs;
}

/**
 * @class DerivativeEstimator
 * @brief Class for estimating the derivative of a signal using Savitzky-Golay numerical differentiation.
 */
class DerivativeEstimator {
private:
    /// Coefficients for the 1st derivative
    std::vector<double> coefficients_;
    
    /// Number of coefficients
    size_t M_{};

    // Circular buffers for each dimension: [dimension][samples]
    std::vector<std::deque<double>> buffers_;
    
    /// Flag to check if the estimator is initialized
    bool initialized_{false};
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// name of the estimator e.g. "com-angular-momentum"
    std::string name_{};

    /// dimensions of the signal
    size_t dim_{};

    /// Signal derivative
    Eigen::VectorXd x_dot_;

    /// Signal derivative covariance
    Eigen::VectorXd x_dot_cov_;

    /// Nominal sample time
    double nominal_dt_{};

    /// Timestamp of the last measurement
    std::optional<double> timestamp_ = std::nullopt;

    /// Flag to check if verbose is enabled
    bool verbose_{false};

    /// Noise gain
    double noise_gain_{0.0};

    /**
     * @brief Sets the state of the estimator.
     * @param x_dot The signal's derivative to set.
     */
    void setState(const Eigen::VectorXd& x_dot);

    /**
     * @brief Gets the covariance of the derivative.
     * @return The covariance of the derivative.
     */
    Eigen::MatrixXd getCovariance() const;

    /**
     * @brief Estimates the derivative of a measurement.
     * @param measurement The signal to estimate the derivative of.
     * @param measurement_variance The variance of the measurement.
     * @param timestamp The timestamp of the measurement.
     * @return The signal's derivative.
     */
    Eigen::VectorXd filter(const Eigen::VectorXd& measurement, const Eigen::VectorXd& measurement_variance, double timestamp);

    /**
     * @brief Resets the estimator.
     */
    void reset();

    /**
     * @brief Initializes the derivative estimator.
     * @param name Name of the estimator e.g. "com-angular-momentum".
     * @param coefficients The coefficients of the Savitzky-Golay numerical differentiation.
     * @param f_sampling The sampling frequency of the signal e.g. 100Hz.
     * @param dim Dimensions of the signal e.g. 3.
     * @param time_horizon The time horizon of the estimator e.g. 0.02s.
     * @param verbose Whether or not to print debug messages.
     */
    DerivativeEstimator(const std::string& name, const std::vector<double>& coefficients, double f_sampling, 
                        size_t dim = 1, double time_horizon = 0.02, bool verbose = false);

    /**
     * @brief Default constructor.
     */
    DerivativeEstimator() = default;
};

}  // namespace serow
