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
