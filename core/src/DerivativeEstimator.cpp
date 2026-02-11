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
#include "DerivativeEstimator.hpp"

#include <iostream>

namespace serow {

DerivativeEstimator::DerivativeEstimator(const std::string& name, const std::vector<double>& coefficients, 
                                         double f_sampling, size_t dim, double time_horizon, bool verbose) {
    name_ = name;
    dim_ = dim;
    nominal_dt_ = 1.0 / f_sampling;
    verbose_ = verbose;

    if (coefficients.empty()) {
        // Compute the coefficients automatically
        M_ = std::max(3, static_cast<int>(std::round(time_horizon * f_sampling)));
        coefficients_ = computeSGCoefficients(M_); 
    } else {
        M_ = coefficients.size();
        coefficients_ = coefficients;
    }

    buffers_.resize(dim_, std::deque<double>(M_, 0.0));
    x_dot_ = Eigen::VectorXd::Zero(dim_);
    
    // Noise power gain: sum of squared coefficients
    noise_gain_ = 0.0;
    for (const double& c : coefficients_) {
        noise_gain_ += c * c;
    }
    // Final variance scale factor: (sum c_i^2) / dt^2
    noise_gain_ /= (nominal_dt_ * nominal_dt_);
    x_dot_cov_ = Eigen::VectorXd::Ones(dim_);

    initialized_ = false;
}

void DerivativeEstimator::reset() {
    for (size_t i = 0; i < dim_; i++) {
        buffers_[i].clear();
        buffers_[i].resize(M_, 0.0);
        x_dot_(i) = 0.0;
        x_dot_cov_(i) = 1.0;
    }

    timestamp_.reset();
    initialized_ = false;

    if (verbose_) {
        std::cout << name_ << " estimator reset" << std::endl;
    }
}

void DerivativeEstimator::setState(const Eigen::VectorXd& x_dot) {
    x_dot_ = x_dot;
}

Eigen::VectorXd DerivativeEstimator::filter(const Eigen::VectorXd& measurement, 
                                            const Eigen::VectorXd& measurement_variance, 
                                            double timestamp) {
    if (measurement.size() != static_cast<int>(dim_) || measurement_variance.size() != static_cast<int>(dim_)) {
        throw std::runtime_error("[DerivativeEstimator] Wrong signal dimensions to filter signal " + name_);
    }

    // --- Warm Start Logic ---
    if (!initialized_) {
        for (size_t d = 0; d < dim_; ++d) {
            std::fill(buffers_[d].begin(), buffers_[d].end(), measurement(d));
        }
        timestamp_ = timestamp;
        initialized_ = true;
        return x_dot_; 
    }

    // --- Time Delta Handling ---
    if (timestamp_) {
        const double dt = timestamp - timestamp_.value();
        if (dt <= 0.0) return x_dot_; // Ignore duplicate/older timestamps
    }
    timestamp_ = timestamp;

    // --- Convolution ---
    for (size_t d = 0; d < dim_; ++d) {
        buffers_[d].push_front(measurement(d));
        buffers_[d].pop_back();

        double derivative_sum = 0.0;
        for (size_t i = 0; i < M_; ++i) {
            derivative_sum += coefficients_[i] * buffers_[d][i];
        }
        
        // Division by nominal_dt converts "change per sample-index" to "change per second"
        x_dot_(d) = derivative_sum / nominal_dt_;

        // Update the covariance of the derivative
        x_dot_cov_(d) = measurement_variance(d) * noise_gain_;
    }
    return x_dot_;
}

Eigen::MatrixXd DerivativeEstimator::getCovariance() const {
    return x_dot_cov_.asDiagonal();
}

}  // namespace serow
