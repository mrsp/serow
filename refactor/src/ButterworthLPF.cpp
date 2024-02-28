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
#include "ButterworthLPF.hpp"

#include <math.h>
#include <iostream>

namespace serow {

void ButterworthLPF::reset(bool verbose) {
    a1_ = 0;
    a2_ = 0;
    b0_ = 0;
    b1_ = 0;
    b2_ = 0;
    i_ = 0;
    y_p_ = 0;
    y_pp_ = 0;
    x_p_ = 0;
    x_pp_ = 0;

    if (verbose) {
        std::cout << name_ << " Low-pass Butterworth filter reset" << std::endl;
    }
}

void ButterworthLPF::init(std::string name, double f_sampling, double f_cutoff, bool verbose) {
    double ff = f_cutoff / f_sampling;
    double ita = 1.0 / tan(3.14159265359 * ff);
    double q = sqrt(2.0);
    b0_ = 1.0 / (1.0 + q * ita + ita * ita);
    b1_ = 2 * b0_;
    b2_ = b0_;
    a1_ = 2.0 * (ita * ita - 1.0) * b0_;
    a2_ = -(1.0 - q * ita + ita * ita) * b0_;
    name_ = name;
    a_ = (2.0 * 3.14159265359 * ff) / (2.0 * 3.14159265359 * ff + 1.0);
    
    if (verbose) {
    std::cout << name << " Low-pass Butterworth filter initialized" << std::endl;
    }
}

double ButterworthLPF::filter(double y) {
    double out{};
    if (i_ > 2) {
        out = b0_ * y + b1_ * y_p_ + b2_ * y_pp_ + a1_ * x_p_ + a2_ * x_pp_;
    } else {
        out = x_p_ + a_ * (y - x_p_);
        i_++;
    }
    y_pp_ = y_p_;
    y_p_ = y;
    x_pp_ = x_p_;
    x_p_ = out;

    return out;
}

} // namespace serow
