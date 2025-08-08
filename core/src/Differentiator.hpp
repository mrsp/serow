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
 * @file Differentiator.hpp
 * @brief Header file for the Differentiator class.
 * @details Provides a mechanism to compute the numerical derivative of a signal using the
 * first-order Euler method.
 */

#pragma once

#include <string>

namespace serow {

/**
 * @class Differentiator
 * @brief Class for numerical differentiation using the first-order Euler method.
 */
class Differentiator {
public:
    /**
     * @brief Initializes the numerical differentiator.
     * @param name Name of the filter, e.g., "LHipYawPitch".
     * @param dt Sampling time in seconds.
     * @param verbose Whether or not to print debug messages.
     */
    Differentiator(const std::string& name, double dt, bool verbose = true);

    /**
     * @brief Default constructor.
     */
    Differentiator() = default;

    /**
     * @brief Sets the filter's sampling time.
     * @param dt Sampling time in seconds.
     */
    void setParams(double dt) {
        dt_ = dt;
    }

    /**
     * @brief Differentiates the measurement with finite differences.
     * @param x Measurement to be differentiated.
     * @param dt Sampling time in seconds.
     * @return The measurement's derivative.
     */
    double filter(double x, double dt);

    /**
     * @brief Resets the filter.
     * @param verbose Whether or not to print debug messages.
     */
    void reset();

private:
    /// Previous measurement
    double x_prev_{};
    /// Estimated derivative
    double xdot_{};
    /// Sampling time in seconds
    double dt_{};
    /// Name of the filter, e.g., "LHipYawPitch"
    std::string name_{};
    /// Flag to check if it's the first run
    bool firstrun_{true};
    /// Flag to check if verbose is enabled
    bool verbose_{};
};

}  // namespace serow
