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

#include "humanoid_state_estimation/IMUEKF.h"


IMUEKF::IMUEKF()
{
	//Gravity Vector
	g = Vector3d::Zero();
	g(2) = -9.80665;

	odom_px = 1.0e-04;
	odom_py = 1.0e-04;
	odom_pz = 1.0e-04;
	odom_ax = 1.0e-02;
	odom_ay = 1.0e-02;
	odom_az = 1.0e-02;


	gyr_qx = 0.009525;
	gyr_qy = 0.00641;
	gyr_qz = 6.5190e-06;
	gyrb_qx = 0.00283344;
	gyrb_qy = 0.00137368;
	gyrb_qz = 6.5190e-06;

	acc_qx = 0.0163;
	acc_qy = 0.0135;
	acc_qz = 0.0156;

	accb_qx = 0.003904;
	accb_qy = 0.003904;
	accb_qz = 0.003904;

	support_qpx = 5.0e-03;
	support_qpy = 5.0e-03;
	support_qpz = 5.0e-03;
	support_qax = 5.0e-03;
	support_qay = 5.0e-03;
	support_qaz = 5.0e-03;
    
	support_px = 5.0e-05;
	support_py = 5.0e-05;
	support_pz = 5.0e-05;
	support_ax = 5.0e-05;
	support_ay = 5.0e-05;
	support_az = 5.0e-05;


}

void IMUEKF::init() {


	firstrun = true;
	P = Matrix<double, 21, 21>::Identity() * (1e-2);
	P(3,3) = 1e-5;
	P(4,4) = 1e-5;
	P(5,5) = 1e-5;
	P(6,6)  = 1e-6;
	P(7,7)  = 1e-6;
	P(8,8)  = 1e-6;
	P(15,15) = 1e-6;
	P(16,16) = 1e-6;
	P(17,17) = 1e-6;
	P(18,18) = 1e-5;
	P(19,19) = 1e-5;
	P(20,20) = 1e-5;

	//Biases
	P(9, 9) = 1e-2;
	P(10, 10) = 1e-2;
	P(11, 11) = 1e-2;
	P(12, 12) = 1e-2;
	P(13, 13) = 1e-2;
	P(14, 14) = 1e-2;



	//Construct C
	Hf = Matrix<double, 6, 21>::Zero();

	/*Initialize the state **/

	//Rotation Matrix from Inertial to body frame
	Rib = Matrix3d::Identity();
	Ris = Matrix3d::Identity();

	x = Matrix<double,21,1>::Zero();

	//Initializing w.r.t NAO Robot -- CHANGE IF NEEDED

	//Innovation Vector
	z = Matrix<double, 6, 1>::Zero();



	//Initializing vectors and matrices
	r = Vector3d::Zero();
	ps = Vector3d::Zero();
	v = Vector3d::Zero();
	chi = Vector3d::Zero();
	phi = Vector3d::Zero();

	dxf = Matrix<double, 21, 1>::Zero();

	temp = Vector3d::Zero();
	tempM = Matrix3d::Zero();
	Kf = Matrix<double, 21, 6>::Zero();
	s = Matrix<double, 6, 6>::Zero();
	If = Matrix<double, 21, 21>::Identity();

	Acf = Matrix<double, 21, 21>::Zero();
	Lcf = Matrix<double, 21, 18>::Zero();
	Qff = Matrix<double, 21, 21>::Zero();
	Qf = Matrix<double, 18, 18>::Zero();
	Af = Matrix<double, 21, 21>::Zero();

	bw = Vector3d::Zero();
	bf = Vector3d::Zero();

	//bias removed acceleration and gyro rate
	fhat = Vector3d::Zero();
	omegahat = Vector3d::Zero();

	Tis = Affine3d::Identity();
	Tib = Affine3d::Identity();
	//Output Variables
	angleX = 0.000;
	angleY = 0.000;
	angleZ = 0.000;
	gyroX = 0.000;
	gyroY = 0.000;
	gyroZ = 0.000;
	accX = 0.000;
	accY = 0.000;
	accZ = 0.000;
	rX = 0.000;
	rY = 0.000;
	rZ = 0.000;
	velX = 0.000;
	velY = 0.000;
	velZ = 0.000;

	std::cout << "IMU EKF Initialized Successfully" << std::endl;

}

/** ------------------------------------------------------------- **/


/** IMU EKF filter to  deal with the Noise **/
void IMUEKF::predict(Vector3d omega_, Vector3d f_)
{

		omega = omega_;
		f = f_;

		// relative velocity
		v = x.segment<3>(0);
		// absolute position
		r = x.segment<3>(6);
		// biases
		bw = x.segment<3>(9);
		bf = x.segment<3>(12);
		// support foot position
		ps = x.segment<3>(15);

		// Correct the inputs
		fhat = f - bf;
		omegahat = omega - bw;

		/** Linearization **/
		//Transition matrix Jacobian
		Acf.block<3,3>(0,0) = -wedge(omegahat);
		Acf.block<3,3>(0,3) = wedge(Rib.transpose() * g);
		Acf.block<3,3>(3,3) = -wedge(omegahat);
		Acf.block<3,3>(6,0) = Rib;
		Acf.block<3,3>(6,3) = -Rib * wedge(v);
		Acf.block<3,3>(0,9) = -wedge(v);
		Acf.block<3,3>(0,12) = -Matrix3d::Identity();
		Acf.block<3,3>(3,9) = -Matrix3d::Identity();

		

		
		//State Noise Jacobian
		//gyro (0),acc (3),gyro_bias (6),acc_bias (9),foot_pos (12),foot_psi (15)		
		Lcf.block<3,3>(0,0) = wedge(v);
		Lcf.block<3,3>(0,3) = Matrix3d::Identity();
		Lcf.block<3,3>(3,0) = Matrix3d::Identity(); 	
		Lcf.block<3,3>(9,6) = Matrix3d::Identity();
		Lcf.block<3,3>(12,9) = Matrix3d::Identity();
		//For Support Foot
		Lcf.block<3,3>(15,12) = Matrix3d::Identity();
		Lcf.block<3,3>(18,15) = Matrix3d::Identity();



		// Covariance Q with full state + biases
		Qf(0, 0) = gyr_qx * gyr_qx ;
		Qf(1, 1) = gyr_qy * gyr_qy ;
		Qf(2, 2) = gyr_qz * gyr_qz ;
		Qf(3, 3) = acc_qx * acc_qx ;
		Qf(4, 4) = acc_qy * acc_qy ;
		Qf(5, 5) = acc_qz * acc_qz ;
		Qf(6, 6) = gyrb_qx * gyrb_qx ;
		Qf(7, 7) = gyrb_qy * gyrb_qy ;
		Qf(8, 8) = gyrb_qz * gyrb_qz  ;
		Qf(9, 9) = accb_qx * accb_qx  ;
		Qf(10, 10) = accb_qy * accb_qy ;
		Qf(11, 11) = accb_qz * accb_qz ;
		Qf(12, 12) = support_qpx * support_qpx ;
		Qf(13, 13) = support_qpy * support_qpy ;
		Qf(14, 14) = support_qpz * support_qpz ;
		Qf(15, 15) = support_qax * support_qax ;
		Qf(16, 16) = support_qay * support_qay ;
		Qf(17, 17) = support_qaz * support_qaz ;

		//Euler Discretization
		Af = If + Acf * dt;
		Qff =  Lcf * Qf * Lcf.transpose() * dt ;
        //Qff =  Af * Lcf * Qf * Lcf.transpose() * Af.transpose() * dt ;
		/** Predict Step: Propagate the Error Covariance  **/
		P = Af * P * Af.transpose() + Qff;
  		
		/** Predict Step : Propagate the Mean estimate **/
		//Body Velocity

		temp = v.cross(omegahat) + Rib.transpose() * g + fhat;
		temp *= dt;
		
		x(0) = v(0) + temp(0);
		x(1) = v(1) + temp(1);
		x(2) = v(2) + temp(2);

		x(3) = 0;
		x(4) = 0;
		x(5) = 0;

		//Body position
		temp = Rib * v;
		temp *= dt;
		x(6) = r(0) + temp(0);
		x(7) = r(1) + temp(1);
		x(8) = r(2) + temp(2);

		//Gyro bias
		x(9) = bw(0);
		x(10) = bw(1);
		x(11) = bw(2);

		//Acc bias
		x(12) = bf(0);
		x(13) = bf(1);
		x(14) = bf(2);

		//Support Position
		x(15) = ps(0);
		x(16) = ps(1);
		x(17) = ps(2);

		//Support Orientation Error phi
		x(18) = 0;
		x(19) = 0;
		x(20) = 0;

		//Propagate only if non-zero input
		temp = omegahat;
		temp *= dt;
		if (temp(0) != 0.0000 && temp(1) != 0.0000 && temp(2) != 0.0000) {
			Rib  *=  expMap(temp, 1.0);
		}
		updateVars();

}

/** Update **/


void IMUEKF::updateWithSupport(Vector3d y,  Quaterniond qy){

		//Support Foot Position
		Hf = Matrix<double,6,21>::Zero();
		R(0, 0) = support_px * support_px;
		R(1, 1) = support_py * support_py;
		R(2, 2) = support_px * support_pz;

		R(3, 3) = support_ax * support_ax;
		R(4, 4) = support_ay * support_ay;
		R(5, 5) = support_az * support_az;


		ps= x.segment<3>(15);
		r = x.segment<3>(6);
		temp = Rib.transpose() * (ps - r);
		z.segment<3>(0) = y - temp;

		Hf.block<3,3>(0, 3) = wedge(temp);
		Hf.block<3,3>(0,6) = -Rib.transpose();
		Hf.block<3,3>(0,15) = Rib.transpose();
		
		//Support Foot Orientation
		Quaterniond qbs(Rib.transpose() * Ris);
		//z.segment<3>(3) = logMap(qy * qbs.inverse()); 
		z.segment<3>(3) = logMap( qbs.inverse() * qy);
        //z.segment<3>(3) = logMap( qy.inverse() * qbs);
		Hf.block<3,3>(3,3) = -Ris.transpose()*Rib;
		Hf.block<3,3>(3,18) = Matrix3d::Identity();


		//Compute the Kalman gain
	    //s = R;
		s = Hf * P * Hf.transpose() + R;
		Kf = P * Hf.transpose() * s.inverse();


		//Correction
		dxf = Kf * z;

		//Update the mean estimate
		x += dxf;

		//Update the error covariance
		P = (If - Kf * Hf) * P * (If - Kf * Hf).transpose() + Kf * R * Kf.transpose();



		if (dxf(18) != 0.000 && dxf(19) != 0.000 && dxf(20) != 0.000) {
			temp(0) = dxf(18);
			temp(1) = dxf(19);
			temp(2) = dxf(20);
			Ris *= expMap(temp, 1.0);
		}
		x.segment<3>(18) = Vector3d::Zero();
		
		if (dxf(3) != 0.000 && dxf(4) != 0.000 && dxf(5) != 0.000) {
			temp(0) = dxf(3);
			temp(1) = dxf(4);
			temp(2) = dxf(5);
			Rib *=  expMap(temp, 1.0);
		}
		x.segment<3>(3) = Vector3d::Zero();

		updateVars();


}

void IMUEKF::updateWithOdom(Vector3d y, Quaterniond qy)
{

		Hf = Matrix<double,6,21>::Zero();

		R(0, 0) = odom_px * odom_px;
		R(1, 1) = odom_py * odom_py;
		R(2, 2) = odom_pz * odom_pz;

		R(3, 3) = odom_ax * odom_ax;
		R(4, 4) = odom_ay * odom_ay;
		R(5, 5) = odom_az * odom_az;

		r = x.segment<3>(6);


		//Innovetion vector
		z.segment<3>(0) = y - r;


		Hf.block<3,3>(0,6) = Matrix3d::Identity();


		Quaterniond qib(Rib);
		//z.segment<3>(3) = logMap( (qy * qib.inverse() ));
		z.segment<3>(3) = logMap( (qib.inverse() * qy ));
        //z.segment<3>(3) = logMap( (qy.inverse() * qib ));
    
		Hf.block<3,3>(3,3) = Matrix3d::Identity();



        //s = R;
        s = Hf * P * Hf.transpose() + R;
		Kf = P * Hf.transpose() * s.inverse();

		dxf = Kf * z;

		//Update the mean estimate
		x += dxf;

		//Update the error covariance
		P = (If - Kf * Hf) * P * (If - Kf * Hf).transpose() + Kf * R * Kf.transpose();


		if (dxf(18) != 0.000 && dxf(19) != 0.000 && dxf(20) != 0.000) {
			temp(0) = dxf(18);
			temp(1) = dxf(19);
			temp(2) = dxf(20);
			Ris *= expMap(temp, 1.0);
		}
		x.segment<3>(18) = Vector3d::Zero();
		
		if (dxf(3) != 0.000 && dxf(4) != 0.000 && dxf(5) != 0.000) {
			temp(0) = dxf(3);
			temp(1) = dxf(4);
			temp(2) = dxf(5);
			Rib *=  expMap(temp, 1.0);
		}
		x.segment<3>(3) = Vector3d::Zero();

		updateVars();

}



void IMUEKF::updateVars()
{


	updateTF();
	
		//Update the biases
		bias_gx = x(9);
		bias_gy = x(10);
		bias_gz = x(11);
		bias_ax = x(12);
		bias_ay = x(13);
		bias_az = x(14);


	omegahat = omega - Vector3d(x(9), x(10), x(11));
	fhat = f - Vector3d(x(12), x(13), x(14));

	temp = Rib * omegahat;
	gyroX = temp(0);
	gyroY = temp(1);
	gyroZ = temp(2);

	temp =  Rib * fhat;
	accX = temp(0);
	accY = temp(1);
	accZ = temp(2);
	velX = x(0);
	velY = x(1);
	velZ = x(2);

	rX = x(6);
	rY = x(7);
	rZ = x(8);

	temp = getEulerAngles(Rib);
	angleX = temp(0);
	angleY = temp(1);
	angleZ = temp(2);



}
void IMUEKF::updateTF() {

	Tis.linear() = Ris;
	Tis.translation() = ps;

	Tib.linear() = Rib;
	Tib.translation() = r;
	qib_ = Quaterniond(Tib.linear());
	qis_ = Quaterniond(Tis.linear());
}
