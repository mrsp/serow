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
 
#ifndef __COMEKF_H__
#define __COMEKF_H__

#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

class CoMEKF {

private:

	Matrix<double, 9, 9> F, P, I, Q, Fd;

	Vector3d COP, fN, L;

	Matrix<double, 6, 9> H;

	Matrix<double, 9, 6> K;

	Matrix<double, 6, 6> R, S;


	Matrix<double, 6, 1> z;
	
	double tmp;
	
	void updateVars();

public:

	Matrix<double, 9, 1> x, f;

	double comd_q, com_q, fd_q, com_r, comdd_r;

	double dt, m, g, I_xx,I_yy;

    double bias_fx, bias_fy, bias_fz;
	bool firstrun;
	
	void init();

	void setdt(double dtt) {
		dt = dtt;
	}

	void setParams(double m_, double I_xx_, double I_yy_, double g_)
	{
		m = m_;
		I_xx = I_xx_;
		I_yy = I_yy_;
		g = g_;

	}

	void setCoMPos(Vector3d pos) {
		x(0) = pos(0);
		x(1) = pos(1);
		x(2) = pos(2);
	}
	void setCoMExternalForce(Vector3d force) {
		x(6) = force(0);
		x(7) = force(1);
		x(8) = force(2);
	}

	void predict(Vector3d COP_, Vector3d fN_, Vector3d L_);
	void update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot);
	void updateWithEnc(Vector3d Pos);
	void updateWithImu(Vector3d Acc, Vector3d Pos, Vector3d Gyro);

	double comX, comY, comZ, velX, velY, velZ, fX,
			fY, fZ;

};

#endif
