/*
 * humanoid_state_estimation - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2016-2017 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 *     * Neither the name of the University of Freiburg nor the names of its
 *       contributors may be used to endorse or promote products derived from
 *       this software without specific prior written permission.
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
 
#include <humanoid_state_estimation/MovingAverageFilter.h>


MovingAverageFilter::MovingAverageFilter()
{
    x = 0.000;
    //Window Size
    windowSize = 10;
    currentstep = 0;
    std::cout<<"Moving Average Filter Initialized Successfully"<<std::endl;
}



void MovingAverageFilter::reset()
{
    x = 0.000;
    currentstep = 0;
    while(windowBuffer.size()>0)
        windowBuffer.pop();
    std::cout<<"Moving Average Filter Reseted"<<std::endl;
}


/** MovingAverageFilter filter to  deal with the Noise **/
void MovingAverageFilter::filter(float  y)
{

    if(currentstep<windowSize)
    {
        //Moving Window
        x = (x*currentstep + y) /(currentstep + 1);
        currentstep++;
    }
    else
    {
        x+=(y-windowBuffer.front())/windowSize;
        if(windowBuffer.size()>=windowSize)
            windowBuffer.pop();
    }
    windowBuffer.push(y);
    /** ------------------------------------------------------------- **/
}
