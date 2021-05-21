/* 
 * Copyright 2017-2021 Stylianos Piperakis, Foundation for Research and Technology Hellas (FORTH)
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
 *		 nor the names of its contributors may be used to endorse or promote products derived from
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
/**
 * @brief Nonlinear CoM Estimation based on encoder, force/torque or pressure, and IMU measurements
 * @author Stylianos Piperakis
 * @details Estimates the 3D CoM Position and Velocity
 */
#ifndef __COMEKF_H__
#define __COMEKF_H__
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace Eigen;
using namespace std;

class CoMEKF {

private:
	/// Linearized State Transition, Error-Covariance, Identity, State Uncertainity, Discrete Linearized State Transition matrices
	Matrix<double, 9, 9> F, P, I, Q, Fd;
	/// 3D Center of Pressure, 3D Ground Reaction Force, Linearized State-Input matrices
	Vector3d COP, fN, L, COP_p, fN_p, L_p;
	/// Linearized Measurement Model
	Matrix<double, 6, 9> H;
	/// Kalman Gain
	Matrix<double, 9, 6> K;
	/// Measurement Noise and Update Covariance
	Matrix<double, 6, 6> R, S;
	/// Innovation vector
	Matrix<double, 6, 1> z;
	/// temp variable
	double tmp;
	/// Update state estimates
	void updateVars();
	/// Euler Discretization - First Order Hold
	void euler(Vector3d COP_, Vector3d fN_, Vector3d L_);
	/// Compute the nonlinear dynamics
	Matrix<double,9,1> computeDyn(Matrix<double,9,1> x_, Vector3d COP_, Vector3d fN_, Vector3d L_);
	/// Compute Linearized matrices
	Matrix<double,9,9> computeTrans(Matrix<double,9,1> x_,  Vector3d COP_, Vector3d fN_, Vector3d L_);
	void RK4(Vector3d COP_, Vector3d fN_, Vector3d L_, Vector3d COP0, Vector3d fN0, Vector3d L0);

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	/// state and nonlinear dynamics vectors
	Matrix<double, 9, 1> x, f;
	/// com position, com velocity, external force, measured com position, measured com acceleration uncertainty 
	/// more info in Nonlinear State Estimation for Humanoid Robot Walking https://www.researchgate.net/publication/326194869_Nonlinear_State_Estimation_for_Humanoid_Robot_Walking
	double com_q, comd_q, fd_q, com_r, comdd_r;
	/// sampling time, robot's mass, gravity, Inertia in x and y axes
	double dt, m, g, I_xx,I_yy;
	/// Biases in external forces
    double bias_fx, bias_fy, bias_fz;
	/// flag to indicate initialization
	bool firstrun;
	/// flag to perform Euler discretization
	bool useEuler;
	 /** @fn void init()
     *   @brief initializes the nonlinear 3D CoM estimator
     */
	void init();
	 /** @fn void setdt(double dtt)
     *   @brief sets the discretization constant of the filter
     *   @param dtt discrete time sampling
     */
	void setdt(double dtt) {
		dt = dtt;
	}
	 /** @fn void setParams(double m_, double I_xx_, double I_yy_, double g_)
     *   @brief sets filter parameters
     *   @param m_ mass of the robot
	 *   @param I_xx_ inertia in the x axis
	 *   @param I_yy_ inertia in the y axis
	 *   @param g_  gravity constant
     */
	void setParams(double m_, double I_xx_, double I_yy_, double g_)
	{
		m = m_;
		I_xx = I_xx_;
		I_yy = I_yy_;
		g = g_;

	}
	/** @fn void setCoMPos(Vector3d pos)
     *   @brief sets initial 3D CoM position
     *   @param pos 3D CoM Position
     */
	void setCoMPos(Vector3d pos) {
		x(0) = pos(0);
		x(1) = pos(1);
		x(2) = pos(2);
	}
	/** @fn void setCoMExternalForce(Vector3d force)
     *   @brief sets the initial 3D external force
     *   @param force 3D external force acting on the CoM
     */
	void setCoMExternalForce(Vector3d force) {
		x(6) = force(0);
		x(7) = force(1);
		x(8) = force(2);
	}
	/** @fn void predict(Vector3d COP_, Vector3d fN_, Vector3d L_);
     *   @brief realizes the predict step of the EKF
     *   @param COP_ 3D COP position
	 *   @param fN_  3D GRF
	 *   @param L_   3D Angular momentum around the CoM 
     */
	void predict(Vector3d COP_, Vector3d fN_, Vector3d L_);
	/** @fn update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot);
     *   @brief realizes the update step of the EKF
     *   @param Acc  3D Body acceleration as measured with an IMU
	 *   @param Pos  3D CoM position as measured with encoders
	 *   @param Gyro 3D Base angular velocity as measured with an IMU
	 *   @param Gyrodot 3D Base angular acceleration as derived from an IMU
     */
	void update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot);
	//void updateWithEnc(Vector3d Pos);
	//void updateWithImu(Vector3d Acc, Vector3d Pos, Vector3d Gyro);
	/// estimated com position, velocity and external forces
	double comX, comY, comZ, velX, velY, velZ, fX,
			fY, fZ;

};
#endif