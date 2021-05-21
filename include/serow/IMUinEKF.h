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
#ifndef __IMUinEKF_H__
#define __IMUinEKF_H__

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <cmath>

using namespace Eigen;

class IMUinEKF
{

  private:
	Matrix<double, 21, 21> P, Af, Adj, Phi, If, Qff, Qf;
	Matrix3d I;
	Vector3d w_, a_, w, a;

	//State vector - with biases included

	Matrix<double, 6, 1> theta;
	void updateStateSingleContact(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix3d N_, Matrix<double, 3, 7> PI_);
	void updateStateDoubleContact(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_);

  public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Matrix<double, 7, 7> X;
	bool firstrun;

	// Gravity vector
	Vector3d g, pwb, vwb, dR, dL, bgyr, bacc, gyro, acc, angle;

	//Noise Stds

	double acc_qx, acc_qy, acc_qz, gyr_qx, gyr_qy, gyr_qz, gyrb_qx, gyrb_qy, gyrb_qz,
		accb_qx, accb_qy, accb_qz, foot_contactx, foot_contacty, foot_contactz, foot_kinx, foot_kiny, foot_kinz;

	double odom_px, odom_py, odom_pz, odom_ax, odom_ay, odom_az,
		vel_px, vel_py, vel_pz, leg_odom_px, leg_odom_py, leg_odom_pz, leg_odom_ax,
		leg_odom_ay, leg_odom_az;

	double gyroX, gyroY, gyroZ, angleX, angleY, angleZ, bias_gx, bias_gy, bias_gz,
		bias_ax, bias_ay, bias_az, ghat;

	double accX, accY, accZ, velX, velY, velZ, rX, rY, rZ;
	Matrix3d Rwb, Rib, Qc;

	Affine3d Tib;

	Quaterniond qib;

	//Sampling time = 1.0/Sampling Frequency
	double dt;

	IMUinEKF();

	void constructState(Matrix<double, 7, 7> &X_, Matrix<double, 6, 1> &theta_, Matrix3d R_, Vector3d v_, Vector3d p_, Vector3d dR_, Vector3d dL_, Vector3d bg_, Vector3d ba_);
	void seperateState(Matrix<double, 7, 7> X_, Matrix<double, 6, 1> theta_, Matrix3d &R_, Vector3d &v_, Vector3d &p_, Vector3d &dR_, Vector3d &dL_, Vector3d &bg_, Vector3d &ba_);
	Matrix<double, 7, 7> exp_SE3(Matrix<double, 15, 1> v);
	Matrix3d exp_SO3(Vector3d v);
	Matrix<double, 21, 21> Adjoint(Matrix<double, 7, 7> X_);

	void updateWithTwist(Vector3d vy, Matrix3d Rvy);
	void updateVelocity(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix3d N_, Matrix<double, 3, 7> PI_);
	void updateWithOrient(Quaterniond qy);
	void updateOrientation(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix<double, 3, 3> N_, Matrix<double, 3, 7> PI_);
	void updateVars();

	void setdt(double dtt)
	{
		dt = dtt;
	}

	void setGyroBias(Vector3d bgyr_)
	{
		bgyr = bgyr_;
		theta.segment<3>(0) = bgyr;
		bias_gx = bgyr(0);
		bias_gy = bgyr(1);
		bias_gz = bgyr(2);
	}
	void setAccBias(Vector3d bacc_)
	{
		bacc = bacc_;
		theta.segment<3>(3) = bacc;
		bias_ax = bacc(0);
		bias_ay = bacc(1);
		bias_az = bacc(2);
	}
	//Initialize the Position
	void setBodyPos(Vector3d bp)
	{
		pwb = bp;
		X.block<3, 1>(0, 4) = bp;
	}

	//Initialize the Rotation Matrix and the Orientation Error
	void setBodyOrientation(Matrix3d bR)
	{
		Rwb = bR;
		X.block<3, 3>(0, 0) = bR;
	}

	void setBodyVel(Vector3d bv)
	{
		vwb = bv;
		X.block<3, 1>(0, 3) = bv;
	}

	void setLeftContact(Vector3d bl)
	{
		dL = bl;
		X.block<3, 1>(0, 6) = bl;
	}

	void setRightContact(Vector3d br)
	{
		dR = br;
		X.block<3, 1>(0, 5) = br;
	}
	inline Vector3d logMap(
		Matrix<double, 3, 3> Rt)
	{

		Vector3d res = Vector3d::Zero();
		double costheta = (Rt.trace() - 1.0) / 2.0;
		double theta = acos(costheta);

		if (theta != 0.000)
		{
			Matrix<double, 3, 3> lnR = Matrix<double, 3, 3>::Zero();
			lnR.noalias() = Rt - Rt.transpose();
			lnR *= theta / (2.0 * sin(theta));
			res = vec(lnR);
		}

		return res;
	}

	void predict(Vector3d angular_velocity, Vector3d linear_acceleration, Vector3d pbr, Vector3d pbl, Matrix3d hR_R, Matrix3d hR_L, int contactR, int contactL);
	void updateWithContacts(Vector3d s_pR, Vector3d s_pL, Matrix3d JRQeJR, Matrix3d JLQeJL, int contactR, int contactL,double weightR, double weightL);

	void updatePositionOrientation(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_);
	void updateWithOdom(Vector3d py, Quaterniond qy);
	void updateVelocityOrientation(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_);
	void updateWithTwistOrient(Vector3d vy, Quaterniond qy);
	// Initializing Variables
	void init();

	//Computes the skew symmetric matrix of a 3-D vector
	Matrix3d skew(Vector3d v)
	{
		Matrix3d res;
		res = Matrix3d::Zero();
		res(0, 1) = -v(2);
		res(0, 2) = v(1);
		res(1, 2) = -v(0);
		res(1, 0) = v(2);
		res(2, 0) = -v(1);
		res(2, 1) = v(0);
		return res;
	}
	Vector3d vec(Matrix3d M)
	{
		Vector3d v = Vector3d::Zero();
		v(0) = M(2, 1);
		v(1) = M(0, 2);
		v(2) = M(1, 0);
		return v;
	}

	//Get the Euler Angles from a Rotation Matrix
	inline Vector3d getEulerAngles(
		Matrix3d Rt)
	{
		Vector3d res;
		res = Vector3d::Zero();

		res(0) = atan2(Rt(2, 1), Rt(2, 2));
		res(1) = atan2(-Rt(2, 0), sqrt(pow(Rt(2, 1), 2) + pow(Rt(2, 2), 2)));
		res(2) = atan2(Rt(1, 0), Rt(0, 0));
		return res;
	}
};
#endif
