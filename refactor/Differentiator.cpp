/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "Differentiator.hpp"

#include <iostream>

void Differentiator::init(std::string name, double dt) {
    dt_ = dt;
    name_ = name;
    std::cout << name_ << " Finite Differentiator Initialized Successfully" << std::endl;
}

double Differentiator::filter(double x) {
    if (firstrun_) {
        firstrun_ = false;
        xdot_ = 0;
    } else
        xdot_ = (x - x_prev_) / dt_;

    x_prev_ = x;
    return xdot_;
}

void Differentiator::reset() {
    x_prev_ = 0.0;
    xdot_ = 0.0;
    firstrun_ = true;
    std::cout << name_ << "Finite Differentiator Reseted Successfully" << std::endl;
}
