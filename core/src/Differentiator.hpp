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
 * @brief Numerical differentiation with the first order Euler method
 * @author Stylianos Piperakis
 * @details to numerical compute the derivative of a signal
 */
#pragma once

#include <string>

namespace serow {

class Differentiator {
   public:
    /// @brief sets the filter's sampling time
    /// @param dt sampling time (s)
    void setParams(double dt) { dt_ = dt; }

    /// @brief Differentiates the measurement with finite differences
    /// @param x measurement to be differentiated
    /// @return The measurement's derivative
    double filter(double x);

    /// @brief Initializes the numerical differentiator
    /// @param name name of the filter e.g "LHipYawPitch"
    /// @param dt sampling time (s)
    /// @param verbose whether or not to print debug messages
    void init(const std::string& name, double dt, bool verbose = true);

    /// @brief Resets the filter
    /// @param verbose whether or not to print debug messages
    void reset(bool verbose = true);

   private:
    /// previous measurement
    double x_prev_{};
    /// estimated derivative
    double xdot_{};
    /// sampling time (s)
    double dt_{};
    /// name of the filter e.g "LHipYawPitch"
    std::string name_{};
    bool firstrun_{true};
};

}  // namespace serow
