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
 * @file ButterworthLPF.hpp
 * @brief A 2nd order Low Pass Butterworth Filter
 * @author Stylianos Piperakis
 * @details This class is used to filter out high frequency noise in certain sensor measurements.
 */
#pragma once

#include <string>

namespace serow {

/**
 * @class ButterworthLPF
 * @brief Class implementing a 2nd order Low Pass Butterworth Filter.
 */
class ButterworthLPF {
private:
    double x_p_{};  /**< State at t-1 */
    double x_pp_{}; /**< State at t-2 */
    double y_p_{};  /**< Measurement at t-1 */
    double y_pp_{}; /**< Measurement at t-2 */

    double a1_{}; /**< 2nd order coefficient a1 */
    double a2_{}; /**< 2nd order coefficient a2 */
    double b0_{}; /**< 2nd order coefficient b0 */
    double b1_{}; /**< 2nd order coefficient b1 */
    double b2_{}; /**< 2nd order coefficient b2 */
    double a_{};  /**< 2nd order coefficient a */

    int i_{};

    std::string name_{}; /**< The name of the filter, e.g., "LHipPitch" */

public:
    /**
     * @brief Resets the 2nd order Low Pass Butterworth filter state.
     * @param verbose If true, prints a reset message to the console.
     */
    void reset(bool verbose = true);

    /**
     * @brief Recursively filters a measurement with a 2nd order Low Pass Butterworth filter.
     * @param y The measurement to be filtered.
     * @return The filtered measurement.
     */
    double filter(double y);

    /**
     * @brief Default constructor of 2nd order Low Pass Butterworth filter.
     */
    ButterworthLPF() = default;

    /**
     * @brief Initializes the 2nd order Low Pass Butterworth filter.
     * @param name The name of the filter, e.g., "LHipPitch".
     * @param f_sampling The sampling frequency of the sensor, e.g., 100 Hz.
     * @param f_cutoff The cut-off frequency of the filter, e.g., 10 Hz.
     * @param verbose If true, prints initialization details to the console.
     */
    ButterworthLPF(std::string name_, double f_sampling, double f_cutoff, bool verbose = true);
};

}  // namespace serow
