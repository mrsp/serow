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
#include "JointEstimator.hpp"

#include <iostream>

namespace serow {

void JointEstimator::init(std::string joint_name, double f_sampling, double f_cutoff,
                          bool verbose) {
    joint_name_ = joint_name;
    joint_position_ = 0.0;
    joint_velocity_ = 0.0;

    bw_.init(joint_name_, f_sampling, f_cutoff, verbose);
    df_.init(joint_name_, 1.0 / f_sampling, verbose);

    if (verbose) {
        std::cout << joint_name_ << " Joint estimator initialized successfully" << std::endl;
    }
}

void JointEstimator::reset(bool verbose) {
    joint_position_ = 0.0;
    joint_velocity_ = 0.0;

    bw_.reset(verbose);
    df_.reset(verbose);

    if (verbose) {
        std::cout << joint_name_ << " Joint estimator reset" << std::endl;
    }
}

double JointEstimator::filter(double joint_position_measurement) {
    joint_position_ = joint_position_measurement;
    joint_velocity_ = bw_.filter(df_.filter(joint_position_measurement));
    return joint_velocity_;
}

}  // namespace serow
