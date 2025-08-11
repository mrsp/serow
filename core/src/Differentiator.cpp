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
#include "Differentiator.hpp"

#include <iostream>

namespace serow {

Differentiator::Differentiator(const std::string& name, double dt, bool verbose) {
    dt_ = dt;
    name_ = name;
    verbose_ = verbose;

    if (verbose) {
        std::cout << name_ << " Finite Differentiator Initialized Successfully" << std::endl;
    }
}

double Differentiator::filter(double x, double dt) {
    if (dt < dt_ / 2.0) {
        if (verbose_) {
            std::cout << "[SEROW/Differentiator] " << name_ << ": Sample time is abnormal " << dt
                      << " while the nominal sample time is " << dt_ << " setting to nominal"
                      << std::endl;
        }
        dt = dt_;
    }

    if (firstrun_) {
        firstrun_ = false;
        xdot_ = 0;
    } else {
        xdot_ = (x - x_prev_) / dt;
    }

    x_prev_ = x;
    return xdot_;
}

void Differentiator::reset() {
    x_prev_ = 0.0;
    xdot_ = 0.0;
    firstrun_ = true;

    if (verbose_) {
        std::cout << name_ << "Finite Differentiator Reseted Successfully" << std::endl;
    }
}

}  // namespace serow
