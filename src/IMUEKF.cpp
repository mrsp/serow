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

#include <serow/IMUEKF.h>


IMUEKF::IMUEKF()
{
	//Gravity Vector
	g = Vector3d::Zero();
	g(2) = -9.80665;
}

void IMUEKF::init() {


	firstrun = true;
	P = Matrix<double, 15, 15>::Identity() * (1e-2);
	P(3,3) = 1e-5;
	P(4,4) = 1e-5;
	P(5,5) = 1e-5;
	P(6,6)  = 1e-6;
	P(7,7)  = 1e-6;
	P(8,8)  = 1e-6;


	//Biases
	P(9, 9) = 1e-2;
	P(10, 10) = 1e-2;
	P(11, 11) = 1e-2;
	P(12, 12) = 1e-2;
	P(13, 13) = 1e-2;
	P(14, 14) = 1e-2;



	//Construct C
	Hf = Matrix<double, 6, 15>::Zero();
	Hv = Matrix<double, 3, 15>::Zero();

	/*Initialize the state **/

	//Rotation Matrix from Inertial to body frame
	Rib = Matrix3d::Identity();

	x = Matrix<double,15,1>::Zero();

	//Initializing w.r.t NAO Robot -- CHANGE IF NEEDED

	//Innovation Vector
	z = Matrix<double, 6, 1>::Zero();
	zv = Vector3d::Zero();



	//Initializing vectors and matrices
	r = Vector3d::Zero();
	v = Vector3d::Zero();
	chi = Vector3d::Zero();
	dxf = Matrix<double, 15, 1>::Zero();



	temp = Vector3d::Zero();
	tempM = Matrix3d::Zero();
	Kf = Matrix<double, 15, 6>::Zero();
	Kv = Matrix<double, 15, 3>::Zero();

	s = Matrix<double, 6, 6>::Zero();
	sv = Matrix<double, 3, 3>::Zero();

	If = Matrix<double, 15, 15>::Identity();
    R = Matrix<double, 6, 6>::Zero();
	Rv = Matrix<double, 3, 3>::Zero();

	Acf = Matrix<double, 15, 15>::Zero();
	Lcf = Matrix<double, 15, 12>::Zero();
	Qff = Matrix<double, 15, 15>::Zero();
	Qf = Matrix<double, 12, 12>::Zero();
	Af = Matrix<double, 15, 15>::Zero();
  
	bw = Vector3d::Zero();
	bf = Vector3d::Zero();

	//bias removed acceleration and gyro rate
	fhat = Vector3d::Zero();
	omegahat = Vector3d::Zero();

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
Matrix<double,15,1> IMUEKF::computeDyn(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_)
{
	Matrix<double,15,1> res = Matrix<double,15,1>::Zero();

	//Inputs without bias
	omega_ -= x_.segment<3>(9);
	f_ -= x_.segment<3>(12);

	//Nonlinear Process Model
	v = x_.segment<3>(0);
	res.segment<3>(0).noalias() = -wedge(omega_) * v;
	res.segment<3>(0).noalias() -= Rib_.transpose()*g;
	res.segment<3>(0) += f_;
	res.segment<3>(6).noalias() = Rib_ * v;

	return res;
}

Matrix<double,15,1> IMUEKF::computeDynRK4(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_)
{
	Matrix<double,15,1> k1 = Matrix<double,15,1>::Zero();
	Matrix<double,15,1> k2 = Matrix<double,15,1>::Zero();
	Matrix<double,15,1> k3 = Matrix<double,15,1>::Zero();
	Matrix<double,15,1> k4 = Matrix<double,15,1>::Zero();
	Matrix<double,15,1> res;

	k1 = computeDyn(x_,Rib_, omega__, f__);
	x_temp.noalias() = x_ + k1 * dt/2.00;
	omega_temp.noalias() = (omega_ + omega__)/2.00;
	Rib_temp.noalias() = Rib_ * expMap(omega_temp);
	f_temp.noalias() = (f_ + f__)/2.00;
	k2 = computeDyn(x_temp,Rib_temp, omega_temp, f_temp);


	x_temp.noalias() = x_ + k2 * dt/2.00;
	k3 = computeDyn(x_temp,Rib_temp, omega_temp, f_temp);
	
	
	x_temp.noalias() = x_ + k3 * dt;
	Rib_temp.noalias() = Rib_ * expMap(omega_);
	k4 = computeDyn(x_temp,Rib_temp, omega_, f_);

	res = x_;
	res.noalias() += dt/6.00 * (k1 + 2*k2 +2*k3 + k4); 

	return res;

}

Matrix<double,15,15> IMUEKF::computeTrans(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_)
{
	omega_ -= x_.segment<3>(9);
	f_ -= x_.segment<3>(12);

	Matrix<double,15,15> res = Matrix<double,15,15>::Zero();

	res.block<3,3>(0,0) = -wedge(omegahat);
	res.block<3,3>(0,3).noalias() = wedge(Rib.transpose() * g);
	res.block<3,3>(3,3) = -wedge(omegahat);
	res.block<3,3>(6,0) = Rib;
	res.block<3,3>(6,3).noalias() = -Rib * wedge(v);
	res.block<3,3>(0,9) = -wedge(v);
	res.block<3,3>(0,12) = -Matrix3d::Identity();
	res.block<3,3>(3,9) = -Matrix3d::Identity();


	return res;
}
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
		//gyro (0),acc (3),gyro_bias (6),acc_bias (9)	
		Lcf.block<3,3>(0,0) = wedge(v);
		Lcf.block<3,3>(0,3) = Matrix3d::Identity();
		Lcf.block<3,3>(3,0) = Matrix3d::Identity(); 	
		Lcf.block<3,3>(9,6) = Matrix3d::Identity();
		Lcf.block<3,3>(12,9) = Matrix3d::Identity();




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


		//Euler Discretization
		Af = If + Acf * dt;
		//Qff =  Lcf * Qf * Lcf.transpose() * dt ;
        	Qff =  Af * Lcf * Qf * Lcf.transpose() * Af.transpose() * dt ;

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

		//Propagate only if non-zero input
		if (omegahat(0) != 0.000 && omegahat(1) != 0.000 && omegahat(2) != 0.000) 
		{
			Rib  *=  expMap(omegahat*dt);
		}
		updateVars();

		f__ = f_;
		omega__ = omega_;

}

/** Update **/
void IMUEKF::updateWithTwist(Vector3d y)
{
	Hv = Matrix<double,3,15>::Zero();
	Rv = Matrix<double,3,3>::Zero();
	Rv(0, 0) = vel_px * vel_px;
	Rv(1, 1) = vel_py * vel_py;
	Rv(2, 2) = vel_pz * vel_pz;

	v = x.segment<3>(0);

	//Innovetion vector
	zv = Vector3d::Zero();
	zv.segment<3>(0).noalias() = y - Rib * v;

	Hv.block<3,3>(0,0) = Rib;
	Hv.block<3,3>(0,3).noalias() = -Rib * wedge(v);
	sv = Rv;
	sv.noalias() += Hv * P * Hv.transpose();
	Kv.noalias() = P * Hv.transpose() * sv.inverse();

	dxf.noalias() = Kv * zv;

		//Update the mean estimate
		x += dxf;

		//Update the error covariance
		P = (If - Kv * Hv) * P * (If - Kv * Hv).transpose() + Kv * Rv * Kv.transpose();


		if (dxf(3) != 0.000 && dxf(4) != 0.000 && dxf(5) != 0.000) 
		{
			Rib *=  expMap(dxf.segment<3>(3));
		}
		x.segment<3>(3) = Vector3d::Zero();

		updateVars();


}

void IMUEKF::updateWithOdom(Vector3d y, Quaterniond qy)
{
	   

		Hf = Matrix<double,6,15>::Zero();
		R = Matrix<double,6,6>::Zero();
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


		//Quaterniond qib(Rib);
		//z.segment<3>(3) = logMap( (qib.inverse() * qy ));
        z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));
		Hf.block<3,3>(3,3) = Matrix3d::Identity();


	   
        s = R;
        s.noalias() = Hf * P * Hf.transpose();
		Kf.noalias() = P * Hf.transpose() * s.inverse();

		dxf.noalias() = Kf * z;

		//Update the mean estimate
		x += dxf;


		//Update the error covariance
		P = (If - Kf * Hf) * P * (If - Kf * Hf).transpose() + Kf * R * Kf.transpose();

		
		if (dxf(3) != 0.000 && dxf(4) != 0.000 && dxf(5) != 0.000)
		{
			Rib *=  expMap(dxf.segment<3>(3));
		}
		x.segment<3>(3) = Vector3d::Zero();

		updateVars();

}



void IMUEKF::updateVars()
{


	Tib.linear() = Rib;
	Tib.translation() = r;
	qib_ = Quaterniond(Tib.linear());	
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
