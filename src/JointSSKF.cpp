/*
 * humanoid_state_estimation - a complete state estimation scheme for humanoid robots
 *
 * Copyright 2017-2018 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 *     * Neither the name of the Foundation for Research and Technology Hellas (FORTH) 
 *	 nor the names of its contributors may be used to endorse or promote products derived from
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

#include "humanoid_state_estimation/JointSSKF.h"



void JointSSKF::init(string JointName_)
{


    F = Matrix2d::Zero();
    F << 1.000, 0.00 , 0.00, 1.000;
    x = Vector2d::Zero();
    K = Vector2d::Zero();
    K << 1.7, 25.000;

    JointName = JointName_;

    JointPosition=0.000;
    JointVelocity=0.000;

    firstrun=true;

    std::cout<<JointName<<" Steady-State Kalman Filter Initialized Successfully"<<std::endl;


}


void JointSSKF::reset()
{

    F = Matrix2d::Zero();
    F << 1.000, dt , 0.00, 1.000;
    x = Vector2d::Zero();
    K = Vector2d::Zero();
    K << 1.7, 25.000;

    JointPosition=0.000;
    JointVelocity=0.000;

    firstrun=true;

    std::cout<<JointName<<" Steady-State Kalman Filter Reseted"<<std::endl;


}

void JointSSKF::setdt(double dtt)
{
	dt=dtt;
	F(0,1) = dt;
}


/** JointSSKF filter to  deal with Delay, and  Noise **/
void JointSSKF::filter(double JointPosMeasurement)
{

    if(firstrun)
    {
        x(0)=JointPosMeasurement;
        firstrun=false;
    }

    /** Predict **/

    x=F*x;

    /** Update **/

    x +=K*(JointPosMeasurement-x(0));

    JointPosition=x(0);
    JointVelocity=x(1);
    /** ------------------------------------------------------------- **/
}
