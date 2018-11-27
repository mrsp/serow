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
 
#include <serow/CoMEKF.h>

void CoMEKF::init() {

	F  = Matrix<double,9,9>::Zero();
	Fd  = Matrix<double,9,9>::Zero();
	I = Matrix<double,9,9>::Identity();


	Q = Matrix<double,9,9>::Zero();


	//Measurement Noise
	x = Matrix<double,9,1>::Zero();
	f = Matrix<double,9,1>::Zero();
	P = Matrix<double,9,9>::Identity();
	z = Matrix<double,6,1>::Zero();
	R = Matrix<double,6,6>::Zero();
	H=Matrix<double,6,9>::Zero();
	K=Matrix<double,9,6>::Zero();
	P.block<3,3>(0,0)= 1e-6 * Matrix<double,3,3>::Identity();
	P.block<3,3>(3,3)= 1e-2 * Matrix<double,3,3>::Identity();
	P.block<3,3>(6,6)= 1e-1* Matrix<double,3,3>::Identity();


	comX = 0.000;
	comY = 0.000;
	comZ = 0.000;

	velX = 0.000;
	velY = 0.000;
	velZ = 0.000;

	fX = 0.000;
	fY = 0.000;
	fZ = 0.000;
	
	firstrun = true;
	
	std::cout << "Non-linear CoM Estimator Initialized!" << std::endl;
}


void CoMEKF::predict(Vector3d COP_, Vector3d fN_, Vector3d L_){
      
	COP = COP_;
	fN = fN_;
	L = L_;

	/*Predict Step*/
	F.block<3,3>(0,3) = Matrix3d::Identity();
	tmp = x(2)-COP(2);

	F(3, 0) = (fN(2) + x(8)) / (m * tmp);

	F(3, 2) = -((fN(2) + x(8)) * (x(0) - COP(0))) / (m * tmp * tmp) +
	 L(1) / (m * tmp * tmp);

	F(3, 6) = 1.000 / m;
	
	F(3, 8) = (x(0) - COP(0)) / (m * tmp);
	
	F(4, 1) = (fN(2) + x(8)) / (m * tmp);

	F(4, 2) = - ( fN(2) + x(8) ) * ( x(1) - COP(1) ) / (m * tmp * tmp) -
	L(0) / (m * tmp * tmp);
	
	F(4, 7) =  1.000 / m;
	
	F(4, 8) = (x(1) - COP(1)) / (m * tmp);

	F(5, 8) =   1.000 / m;


	//Discretization
	Q(0, 0) = com_q * com_q;
	Q(1, 1) = Q(0,0);
	Q(2, 2) = Q(0,0);

	Q(3, 3) = comd_q * comd_q;
	Q(4, 4) = Q(3, 3);
	Q(5, 5) = Q(3, 3);

	Q(6, 6) = fd_q * fd_q;
	Q(7, 7) = Q(6, 6);
	Q(8, 8) = Q(6, 6);
	
	Fd = I;
	Fd.noalias() += F * dt ;


	
	P = Fd * P * Fd.transpose() + Q;


	//Forward Euler Integration of dynamics f
	f(0) = x(3);
	f(1) = x(4);
	f(2) = x(5);

	f(3) = (x(0) - COP(0)) / (m * tmp) * (fN(2) + x(8)) + x(6) / m - L(1) / (m * tmp); 
	f(4) = (x(1) - COP(1)) / (m * tmp) * (fN(2) + x(8)) + x(7) / m + L(0) / (m * tmp);
 	f(5) = (fN(2) + x(8)) / m - g;


	x  += f * dt;

	updateVars();

}

// void CoMEKF::updateWithEnc(Vector3d Pos)
// {
// 	z.noalias() = Pos - x.segment<3>(0);
	
// 	H = Matrix<double,3,9>::Zero();
// 	H.block<3,3>(0,0) = Matrix3d::Identity();
	
// 	R(0, 0) = com_r;
// 	R(1, 1) = com_r;
// 	R(2, 2) = com_r;
	
// 	S = R;
// 	S.noalias() += H * P * H.transpose();
// 	K.noalias() = P * H.transpose() * S.inverse();

// 	x += K * z;
// 	P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();
// 	updateVars();
// }

// void CoMEKF::updateWithImu(Vector3d Acc, Vector3d Pos, Vector3d Gyro){


// 	/* Update Step */
// 	//Compute the CoM Acceleration
// 	//Acc += Gyro.cross(Gyro.cross(Pos)) + Gyrodot.cross(Pos);  
 
// 	tmp = x(2)-COP(2);

// 	z(0) = Acc(0) - ( (x(0) - COP(0)) / (m * tmp) * (fN(2) + x(8)) + x(6) / m - I_yy  * Gyrodot(1) / (m * tmp) );
// 	z(1) = Acc(1) - ( (x(1) - COP(1)) / (m * tmp) * (fN(2) + x(8)) + x(7) / m + I_xx  * Gyrodot(0) / (m * tmp) );
// 	z(2) = Acc(2) - ( (fN(2) + x(8)) / m - g );




// 	H = Matrix<double,3,9>::Zero();
	
// 	H(0, 0) = (fN(2) + x(8)) / (m * tmp);
// 	H(0, 2) =  -((fN(2) + x(8)) * (x(0) - COP(0))) / (m * tmp * tmp) +  I_yy * Gyrodot(1) / (m * tmp * tmp);

// 	H(0, 6) = 1.000 / m;
// 	H(0, 8) = (x(0) - COP(0)) / (m * tmp);

// 	H(1, 1) = (fN(2) + x(8)) / (m * tmp);
// 	H(1, 2) = - ( fN(2) + x(8) ) * ( x(1) - COP(1) ) / (m * tmp * tmp) - I_xx * Gyrodot(0) / (m * tmp * tmp);
// 	H(1, 7) = 1.000 / m;
// 	H(1, 8) = (x(1) - COP(1)) / (m * tmp);
// 	H(2, 8) = 1.000 / m;


// 	R(0, 0) = comdd_r;
// 	R(1, 1) = comdd_r;
// 	R(2, 2) = comdd_r;
		
// 	S = R;
// 	S.noalias() += H * P * H.transpose();
// 	K.noalias() = P * H.transpose() * S.inverse();

// 	x += K * z;
// 	P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();
// 	updateVars();
// }


void CoMEKF::update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot){

	/* Update Step */
	//Compute the CoM Acceleration
	Acc += Gyro.cross(Gyro.cross(Pos)) + Gyrodot.cross(Pos);  


	tmp = x(2)-COP(2);

	z.segment<3>(0).noalias() = Pos - x.segment<3>(0);

	z(3) = Acc(0) - ( (x(0) - COP(0)) / (m * tmp) * (fN(2) + x(8)) + x(6) / m - L(1) / (m * tmp) );
	z(4) = Acc(1) - ( (x(1) - COP(1)) / (m * tmp) * (fN(2) + x(8)) + x(7) / m + L(0) / (m * tmp) );
	z(5) = Acc(2) - ( (fN(2) + x(8)) / m - g );




	H = Matrix<double,6,9>::Zero();
	H.block<3,3>(0,0) = Matrix3d::Identity();

	H(3, 0) = (fN(2) + x(8)) / (m * tmp);
	H(3, 2) =  -((fN(2) + x(8)) * (x(0) - COP(0))) / (m * tmp * tmp) +  L(1) / (m * tmp * tmp);

	H(3, 6) = 1.000 / m;
	H(3, 8) = (x(0) - COP(0)) / (m * tmp);

	H(4, 1) = (fN(2) + x(8)) / (m * tmp);
	H(4, 2) = - ( fN(2) + x(8) ) * ( x(1) - COP(1) ) / (m * tmp * tmp) - L(0) / (m * tmp * tmp);
	H(4, 7) = 1.000 / m;
	H(4, 8) = (x(1) - COP(1)) / (m * tmp);
	H(5, 8) = 1.000 / m;


	R(0, 0) = com_r * com_r;
	R(1, 1) = R(0, 0);
	R(2, 2) = R(0, 0);
	
	R(3, 3) = comdd_r * comdd_r;
	R(4, 4) = R(3, 3);
	R(5, 5) = R(3, 3);

	S = R;
	S.noalias() += H * P * H.transpose();
	K.noalias() = P * H.transpose() * S.inverse();

	x += K * z;
	P = (I - K * H) * P * (I - K * H).transpose() + K * R * K.transpose();
	updateVars();
}


void CoMEKF::updateVars()
{
	
	comX = x(0);
	comY = x(1);
	comZ = x(2);

	velX = x(3);
	velY = x(4);
	velZ = x(5);

	fX = x(6);
	fY = x(7);
	fZ = x(8) + fN(2);

}

