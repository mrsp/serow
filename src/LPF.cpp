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
 
#include "humanoid_state_estimation/LPF.h"

LPF::LPF()
{
    s = Vector3d::Zero();
    s_ = Vector3d::Zero();
    a_lp = Matrix3d::Zero();

    x = 0;
    y = 0;
    z = 0;

}
void LPF::reset()
{
    s = Vector3d::Zero();
    s_ = Vector3d::Zero();
    a_lp = Matrix3d::Zero();

    x = 0;
    y = 0;
    z = 0;

    std::cout<<"LPF reseted! "<<std::endl;
}

void LPF::setdt(double dtt)
{
    dt=dtt;
}

void LPF::setCutOffFreq(double fx_, double fy_, double fz_)
{
    fx = fx_;
    fy = fy_;
    fz = fz_;
}


/** LPF filter to  deal with the Noise **/
void LPF::filter(Vector3d  y_m)
{
    //Cut-off Frequencies
    a_lp(0,0)=(2.0*3.14159265359*dt*fx)/(2*3.14159265359*dt*fx+1);
    a_lp(1,1)=(2.0*3.14159265359*dt*fy)/(2*3.14159265359*dt*fy+1);
    a_lp(2,2)=(2.0*3.14159265359*dt*fz)/(2*3.14159265359*dt*fz+1);

    //Low Pass Filtering
    s= a_lp*y_m + s_ - a_lp * s_;


    x=s(0);
    y=s(1);
    z=s(2);

    s_=s;
    /** ------------------------------------------------------------- **/
}
