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
 * @brief A 2nd order Low Pass Butterworth Filter
 * @author Stylianos Piperakis
 * @details to filter out high frequency noise in certain sensor measurements
 */
#pragma once

#include <string>

namespace serow {

class ButterworthLPF {
   private:
    /// state at t-1, t-2, measurement at t-1, t-2
    double x_p_{}, x_pp_{}, y_p_{}, y_pp_{};
    /// 2nd order coefficients
    double a1_{}, a2_{}, b0_{}, b1_{}, b2_{}, a_{};
    int i_{};
    /// the name of the filter e.g. "LHipPitch"
    std::string name_{};

   public:
    /** @fn void reset(bool verbose = true)
     *  @brief resets the 2nd order Low Pass Butterworth filter state
     */
    void reset(bool verbose = true);

    /** @fn double filter(double y)
     *  @brief recursively filters a measurement with a 2nd order Low Pass Butterworth filter
     *  @param y  measurement to be filtered
     */
    double filter(double y);

    /** @fn ButterworthLPF()
     *  @brief constructor of 2nd order Low Pass Butterworth filter
     */
    ButterworthLPF() = default;

    /** @fn ButterworthLPF(string name_ ,double fsampling, double fcutoff)
     *  @brief initializes the 2nd order Low Pass Butterworth filter
     *  @param name_ the name of the filter e.g. "LHipPitch"
     *  @param f_sampling the sampling frequency of the sensor e.g. 100hz
     *  @param f_cutoff the cut-off frequency of the filter e.g. 10hz
     */
    ButterworthLPF(std::string name_, double f_sampling, double f_cutoff, bool verbose = true);
};

}  // namespace serow
