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
 * @brief Joint angular velocity estimation with numerical differentiation and
 * 2nd order Low Pass Butterworth Filter
 * @author Stylianos Piperakis
 * @details estimates the joint angular velocity from joint encoder measurements
 */
#pragma once

#include "ButterworthLPF.hpp"
#include "Differentiator.hpp"

namespace serow {

class JointEstimator {
   private:
    /// 2nd order butterworth filter to smooth the angular velocity
    ButterworthLPF bw_;
    /// linear differentiator filter to compute the angular velocity
    Differentiator df_;

   public:
    /// Joint angular position
    double joint_position_{};
    /// Joint angular velocity
    double joint_velocity_{};
    /// Joint name
    std::string joint_name_{};

    /** @fn double filter(double joint_position_measurement);
     *  @brief estimates the Joint Velocity using the Joint Position measurement
     *  by the encoders
     */
    double filter(double joint_position_measurement);

    /** @fn void reset();
     *  @brief  resets the the joint estimator
     */
    void reset(bool verbose = true);

    /** @fn void init(std::string joint_name,double f_sampling, double f_cutoff);
     *  @brief initializes the differentiator filter
     *  @param joint_name the name of the filter e.g. "LHipPitch"
     *  @param f_sampling the sampling frequency of the sensor e.g. 100hz
     *  @param f_cutoff the cut-off frequency of the  2nd order Low Pass
     *  Butterworth Filter filter e.g. 10hz
     */
    void init(std::string joint_name, double f_sampling, double f_cutoff, bool verbose = true);
};

}  // namespace serow