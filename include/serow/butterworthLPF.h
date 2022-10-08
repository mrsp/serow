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
 * @brief A 2nd order Low Pass Butterworth Filter
 * @author Stylianos Piperakis
 * @details to filter out high frequency noise in certain sensor measurements
 */

#pragma once
#include <string.h>
#include <math.h>
#include <iostream>

class butterworthLPF
{

private:
    /// state at t-1, t-2, measurement at t-1, t-2
    double x_p, x_pp, y_p, y_pp;
    /// 2nd order coefficients
    double a1, a2, b0, b1, b2, ff, ita, q, a;
    /// frequency ration and sampling frequency
    double fx, fs;
    int i;

public:
    /// the name of the filter e.g. "LHipPitch"
    std::string name;
    /** @fn void reset()
     *  @brief resets the 2nd order Low Pass Butterworth filter state
     */
    void reset();

    /** @fn double filter(double y)
     *  @brief recursively filters a measurement with a 2nd order Low Pass
     *  Butterworth filter
     *  @param y  measurement to be filtered
     */
    double filter(double y);
    /** @fn butterworthLPF()
     *  @brief constructor of 2nd order Low Pass Butterworth filter
     */
    butterworthLPF();
    /** @fn void init(string name_ ,double fsampling, double fcutoff)
     *  @brief initializes the 2nd order Low Pass Butterworth filter
     *  @param name_ the name of the filter e.g. "LHipPitch"
     *  @param fsampling the sampling frequency of the sensor e.g. 100hz
     *  @param fcutoff the cut-off frequency of the filter e.g. 10hz
     */
    void init(std::string name_, double fsampling, double fcutoff);
};
