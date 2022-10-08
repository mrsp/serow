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
 * @brief An 1-D moving average filter
 * @author Stylianos Piperakis
 * @details to smooth out/filter a sensor measurement signal
 */

#pragma once
#include <queue>
#include <iostream>

class MovingAverageFilter
{

private:
    /// Moving average window
    int windowSize;
    unsigned currentstep;
    /// Moving average buffer
    std::queue<float> windowBuffer;

public:
    /// Moving average cuurent state
    float x;

    /** @fn void setParams(int windowSize_)
     *  @brief sets the buffer size of the moving average filter
     */
    void setParams(int windowSize_)
    {
        windowSize = windowSize_;
    }

    /** @fn void filter(float y)
     *  @brief filters the measurement with a cummulative moving average filter
     */
    void filter(float y);
    /** fn MovingAverageFilter()
     *  @brief initializes the moving average filter
     */
    MovingAverageFilter();
    /** @fn void reset()
     *  @brief resets the state of the moving average filter
     */
    void reset();
};
