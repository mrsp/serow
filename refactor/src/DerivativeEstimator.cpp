/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "DerivativeEstimator.hpp"

#include <iostream>

namespace serow {

void DerivativeEstimator::init(std::string joint_name, double f_sampling, double f_cutoff,
                               bool verbose) {
    name_ = joint_name;
    for (size_t i = 0; i < 3; i++) {
        bw_[i].init(name_, f_sampling, f_cutoff, verbose);
        df_[i].init(name_, 1.0 / f_sampling, verbose);
    }

    std::cout << name_ << " estimator initialized successfully" << std::endl;
}

void DerivativeEstimator::reset(bool verbose) {
    for (size_t i = 0; i < 3; i++) {
        bw_[i].reset();
        df_[i].reset();
    }

    std::cout << name_ << " estimator reset" << std::endl;
}

Eigen::Vector3d DerivativeEstimator::filter(const Eigen::Vector3d& measurement) {
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    for (size_t i = 0; i < 3; i++) {
        out(i) = bw_[i].filter(df_[i].filter(measurement(i)));
    }
    return out;
}

}  // namespace serow
