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
 
#ifndef  __IMUEKF_H__
#define  __IMUEKF_H__

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>       /* isnan, sqrt */
/* @author MUMRA
 */

/* @brief  IMU-Kinematics/Encoders-VO fusion
 * dx/dt=f(x,u,w)
 * y_k=Hx_k+v
 * State is  position in Inertial frame : r
 * velocity in  Body frame : v
 * orientation of Body frame wrt the Inertial frame: Rib
 * accelerometer bias in Body frame : bf
 * gyro bias in Body frame : bw
 * Measurement is Body Position/Orinetation in Inertial frame by Odometry-VO
 * and the relative Support Foot Position/Orientation by the Kinematics/Encoders
 */

 
using namespace Eigen;

class IMUEKF {

private:


	Matrix<double, 15, 12> Lcf;

	Matrix<double, 15, 15> P, Af, Acf, If, Qff;

	Matrix<double, 6, 15> Hf;
	Matrix<double, 3, 15> Hv;

	Matrix<double, 12, 12> Qf;

	Matrix<double, 15, 6> Kf;
	Matrix<double, 15, 3> Kv;

	//Correction vector
	Matrix<double, 15, 1> dxf;

	//General variables
	Matrix<double, 6, 6> s, R;

	Matrix<double, 3, 3> sv, Rv;


	//innovation, position, velocity , acc bias, gyro bias, bias corrected acc, bias corrected gyr, temp vectors
	Vector3d r, v, omega, f, fhat, omegahat, temp, omega_p, f_p;

	Matrix<double, 6, 1> z;
	Vector3d zv;

    double tau, zeta, f0, e0, e_t, f_t;
   
    //RK4 Integration 
    Matrix<double,15,1> computeDyn(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_);
	void RK4(Vector3d omega_, Vector3d f_, Vector3d omega0, Vector3d f0);
	Matrix<double,15,15> computeTrans(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_);

	void euler(Vector3d omega_, Vector3d f_);
	void updateOutlierDetectionParams(Eigen::Matrix<double, 6,6> B);
	double computePsi(double xx);


public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
             
	//State vector - with biases included
	Matrix<double, 15, 1> x;

	bool firstrun;
	bool useEuler;
	// Gravity vector
    Vector3d g, bgyr, bacc, gyro, acc, vel, pos, angle;

	//Noise Stds

	double  acc_qx,acc_qy,acc_qz,gyr_qx,gyr_qy,gyr_qz,gyrb_qx,gyrb_qy,gyrb_qz,
	accb_qx,accb_qy,accb_qz, odom_px, odom_py, odom_pz, odom_ax, odom_ay,odom_az,
	vel_px, vel_py, vel_pz;

	double gyroX, gyroY, gyroZ, angleX, angleY, angleZ, bias_gx, bias_gy, bias_gz,
			bias_ax, bias_ay, bias_az, ghat;

	double accX, accY, accZ, velX, velY, velZ, rX, rY, rZ;

	Matrix3d Rib;

	Affine3d  Tib;

	Quaterniond qib;

	//Sampling time = 1.0/Sampling Frequency
	double dt;

	IMUEKF();

	void updateVars();

	void setdt(double dtt) {
		dt = dtt;
	}

	void setGyroBias(Vector3d bgyr_)
	{
		bgyr = bgyr_;
		x(9) = bgyr_(0);
		x(10) = bgyr_(1);
		x(11) = bgyr_(2);
		bias_gx = bgyr_(0);
		bias_gy = bgyr_(1);
		bias_gz = bgyr_(2);
	}
	void setAccBias(Vector3d bacc_)
	{
		bacc = bacc_;
		x(12) = bacc_(0);
		x(13) = bacc_(1);
		x(14) = bacc_(2);
		bias_ax = bacc_(0);
		bias_ay = bacc_(1);
		bias_az = bacc_(2);
	}
	//Initialize the Position
	void setBodyPos(Vector3d bp) {
		x(6) = bp(0);
		x(7) = bp(1);
		x(8) = bp(2);
	}

	//Initialize the Rotation Matrix and the Orientation Error
	void setBodyOrientation(Matrix3d Rot_){
		Rib = Rot_;
	}

    void setBodyVel(Vector3d bv)
    {
        x.segment<3>(0).noalias() =  bv;
    }


	/** @fn void Filter(Matrix<double,3,1> f, Matrix<double,3,1> omega, Matrix<double,3,1>  y_r, Matrix<double,3,1>  y_q)
	 *  @brief filters the acceleration measurements from the IMU
	 */
	void predict(Vector3d omega_, Vector3d f_);
	void updateWithOdom(Vector3d y, Quaterniond qy);
	void updateWithTwist(Vector3d y);
 	//void updateWithSupport(Vector3d y, Quaterniond qy);

	// Initializing Variables
	void init();

	//Computes the skew symmetric matrix of a 3-D vector
	Matrix3d wedge(
			Vector3d v) {

		Matrix3d skew;

		skew = Matrix3d::Zero();
		skew(0, 1) = -v(2);
		skew(0, 2) = v(1);
		skew(1, 2) = -v(0);
		skew(1, 0) = v(2);
		skew(2, 0) = -v(1);
		skew(2, 1) = v(0);

		return skew;

	}
	Vector3d vec(Matrix3d M)
	{
		Vector3d v = Vector3d::Zero();
		v(0) = M(2,1);
		v(1) = M(0,2);
		v(2) = M(1,0);
		return v;
	}

	//Rodriguez Formula
	inline Matrix<double, 3, 3> expMap(
			Vector3d omega) {

		Matrix<double, 3, 3> res;
		double omeganorm;

		omeganorm = omega.norm();
		res =  Matrix<double, 3, 3>::Identity();

		if(omeganorm !=0.0)
		{
			Matrix<double, 3, 3>  omega_skew;
			omega_skew = Matrix<double, 3, 3>::Zero();

			omega_skew = wedge(omega);
			res += omega_skew * (sin(omeganorm) / omeganorm);
			res += (omega_skew * omega_skew) * (
					(1.000 - cos(omeganorm)) / (omeganorm * omeganorm));
		}

		return res;
	}

	inline Vector3d logMap(
			Matrix<double, 3, 3> Rt) {

		Vector3d res = Vector3d::Zero();
		double costheta = (Rt.trace()-1.0)/2.0;
		double theta = acos(costheta);

		if (theta != 0.000) {
			Matrix<double, 3, 3> lnR = Matrix<double, 3, 3>::Zero();
			lnR.noalias() =  Rt - Rt.transpose();
			lnR *= theta /(2.0*sin(theta));
			res = vec(lnR); 			
		}

		return res;
	}

inline Vector3d logMap(
			Quaterniond q) {

		Vector3d omega;
		omega = Vector3d::Zero();

		double temp = q.norm();

		Vector3d tempV = Vector3d(q.x(), q.y(), q.z());

		double temp_ = tempV.norm();
		tempV *= (1.000 / temp_);


		omega = tempV * (2.0 * acos(q.w() / temp));
		//omega = tempV * (2.0 * atan2(temp_,q.w()));
		if(std::isnan(omega(0) + omega(1) + omega(2)))
			omega = Vector3d::Zero();

		return omega;
	}




	
	//Get the Euler Angles from a Rotation Matrix
	inline Vector3d getEulerAngles(
			Matrix3d Rt) {
		Vector3d res;
		res = Vector3d::Zero();

		res(0) = atan2(Rt(2, 1), Rt(2, 2));
		res(1) = atan2(-Rt(2, 0), sqrt(pow(Rt(2, 1), 2) + pow(Rt(2, 2), 2)));
		res(2) = atan2(Rt(1, 0), Rt(0, 0));
		return res;
	}



	inline Matrix3d getRotationMatrix(
			Vector3d angles_) {
		Matrix3d res, Rz, Ry, Rx;
		Rz = Matrix3d::Zero();

		Rz(0, 0) = cos(angles_(2));
		Rz(0, 1) = -sin(angles_(2));
		Rz(1, 0) = sin(angles_(2));
		Rz(1, 1) = cos(angles_(2));
		Rz(2, 2) = 1.000;

		Ry = Matrix3d::Zero();
		Ry(0, 0) = cos(angles_(1));
		Ry(0, 2) = sin(angles_(1));
		Ry(1, 1) = 1.000;
		Ry(2, 0) = -sin(angles_(1));
		Ry(2, 2) = cos(angles_(1));

		Rx = Matrix3d::Zero();
		Rx(0, 0) = 1.00;
		Rx(1, 1) = cos(angles_(0));
		Rx(1, 2) = -sin(angles_(0));
		Rx(2, 1) = sin(angles_(0));
		Rx(2, 2) = cos(angles_(0));

		//YAW PITCH ROLL Convention for right handed counterclowise coordinate systems
		res = Rz * Ry * Rx;

		return res;
	}

	

};
#endif
