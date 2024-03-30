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
    /** @fn void setParams(double dt)
     *  @brief differentiates the measurement with finite differences
     *  @param dt Sampling time in seconds e.g. 0.01s
     */
    void setParams(double dt) { dt_ = dt; }
    /** @fn void filter(double x)
     *  @brief differentiates the measurement with finite differences
     *  @param x signal to be differentiatied
     */
    double filter(double x);

    /** @fn void init(std::string name, double dt);
     *  @brief initializes the numerical differentiator
     *  @param name name of the signal e.g LHipYawPitch
     *  @param dt Sampling time in seconds e.g. 0.01s
     */
    void init(std::string name, double dt, bool verbose = true);

    /** @fn void reset();
     *  @brief  resets the the numerical differentiator's state
     */
    void reset(bool verbose = true);

   private:
    double x_prev_{};
    double xdot_{};
    double dt_{};
    bool firstrun_{true};
    std::string name_{};
};

}  // namespace serow
