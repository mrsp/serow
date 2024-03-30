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
 * @brief Signal derivative estimation with numerical differentiation and
 * 2nd order Low Pass Butterworth Filter
 * @author Stylianos Piperakis
 * @details estimates the measurement's 3D derivative
 */
#pragma once

#include "ButterworthLPF.hpp"
#include "Differentiator.hpp"

#ifdef __linux__
#include <eigen3/Eigen/Dense>
#else
#include <Eigen/Dense>
#endif

namespace serow {

class DerivativeEstimator {
   private:
    /// 2nd order butterworth filter to smooth the 3D derivative
    ButterworthLPF bw_[3];
    /// linear differentiator filter to compute the 3D derivative
    Differentiator df_[3];

   public:
    std::string name_{};

    /** @fn Eigen::Vector3d filter(Eigen::Vector3d measurement);
     *  @brief estimates the derivative of a 3D measurement
     */
    Eigen::Vector3d filter(const Eigen::Vector3d& measurement);

    /** @fn void reset();
     *  @brief resets the the estimator
     */
    void reset(bool verbose = false);

    /** @fn void init(std::string name, double f_sampling, double f_cutoff);
     *  @brief initializes the differentiator filter
     *  @param name the name of the filter e.g. "AngularMomentum"
     *  @param f_sampling the sampling frequency of the sensor e.g. 100hz
     *  @param f_cutoff the cut-off frequency of the  2nd order Low Pass
     *  Butterworth Filter filter e.g. 10hz
     */
    void init(std::string name, double f_sampling, double f_cutoff, bool verbose = false);
};

}  // namespace serow
