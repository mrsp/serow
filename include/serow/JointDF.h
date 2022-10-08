/*
 * Copyright 2017-2023 Stylianos Piperakis,
 * Foundation for Research and Technology Hellas (FORTH)
 * License: BSD
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Foundation for Research and Technology Hellas
 *       (FORTH) nor the names of its contributors may be used to endorse or
 *       promote products derived from this software without specific prior
 *       written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
 /**
 * @brief Joint angular velocity estimation with numerical differentiation and
 * 2nd order Low Pass Butterworth Filter
 * @author Stylianos Piperakis
 * @details estimates the joint angular velocity from joint encoder measurements
 */

#pragma once
#include <serow/differentiator.h>
#include <serow/butterworthLPF.h>
#include <iostream>

using namespace std;
class JointDF
{

private:
    /// 2nd order butterworth filter to smooth the angular velocity
    butterworthLPF bw;
    /// linear differentiator filter to compute the angular velocity
    Differentiator df;

public:
    /// Joint angular position
    double JointPosition;
    /// Joint angular velocity
    double JointVelocity;
    /// Joint name
    string JointName;

    /** @fn double filter(double JointPosMeasurement);
     *  @brief estimates the Joint Velocity using the Joint Position measurement
     *  by the encoders
     */
    double filter(double JointPosMeasurement);
    void reset();
    /** @fn void init(string JointName_,double fsampling, double fcutoff);
     *  @brief initializes the differentiator filter
     *  @param JointName_ the name of the filter e.g. "LHipPitch"
     *  @param fsampling the sampling frequency of the sensor e.g. 100hz
     *  @param fcutoff the cut-off frequency of the  2nd order Low Pass
     *  Butterworth Filter filter e.g. 10hz
     */
    void init(string JointName_, double fsampling, double fcutoff);
};
