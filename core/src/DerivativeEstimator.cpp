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
#include "DerivativeEstimator.hpp"

#include <iostream>

namespace serow {

DerivativeEstimator::DerivativeEstimator(const std::string& name, double f_sampling,
                                         double f_cutoff, size_t dim, bool verbose) {
    name_ = name;
    dim_ = dim;
    bw_.resize(dim);
    df_.resize(dim);
    for (size_t i = 0; i < dim; i++) {
        bw_[i] = std::move(ButterworthLPF(name_, f_sampling, f_cutoff, verbose));
        df_[i] = std::move(Differentiator(name_, 1.0 / f_sampling, verbose));
    }

    x_ = Eigen::VectorXd::Zero(dim);
    x_dot_ = Eigen::VectorXd::Zero(dim);
    nominal_dt_ = 1.0 / f_sampling;
    verbose_ = verbose;

    if (verbose_) {
        std::cout << name_ << " estimator initialized successfully" << std::endl;
    }
}

void DerivativeEstimator::setState(const Eigen::VectorXd& x, const Eigen::VectorXd& x_dot) {
    x_ = x;
    x_dot_ = x_dot;
}

void DerivativeEstimator::reset() {
    for (size_t i = 0; i < dim_; i++) {
        bw_[i].reset();
        df_[i].reset();
    }
    timestamp_.reset();

    if (verbose_) {
        std::cout << name_ << " estimator reset" << std::endl;
    }
}

Eigen::VectorXd DerivativeEstimator::filter(const Eigen::VectorXd& measurement, double timestamp) {
    if (measurement.size() != static_cast<int>(dim_)) {
        throw std::runtime_error(
            "Derivative estimator created with wrong signal dimensions, returning the measurement");
        return measurement;
    }

    double dt = nominal_dt_;
    if (timestamp_) {
        dt = timestamp - timestamp_.value();
        if (dt < nominal_dt_ / 2.0) {
            std::cout << "[SEROW/DerivativeEstimator] " << name_ << ": Sample time is abnormal "
                      << dt << " while the nominal sample time is " << nominal_dt_
                      << " setting to nominal" << std::endl;
            dt = nominal_dt_;
        }
    }
    timestamp_ = timestamp;

    for (size_t i = 0; i < dim_; i++) {
        x_(i) = bw_[i].filter(measurement(i));
        x_dot_(i) = df_[i].filter(x_(i), dt);
    }
    return x_dot_;
}

}  // namespace serow
