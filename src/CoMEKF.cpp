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


	useEuler = true;
	F  = Matrix<double,9,9>::Zero();
	

	Fd  = Matrix<double,9,9>::Zero();
	I = Matrix<double,9,9>::Identity();


	Q = Matrix<double,9,9>::Zero();
	COP = Vector3d::Zero();
	fN = Vector3d::Zero();
	L = Vector3d::Zero();
	COP_p = Vector3d::Zero();
	fN_p = Vector3d::Zero();
	L_p = Vector3d::Zero();	

	//Measurement Noise
	x = Matrix<double,9,1>::Zero();
	f = Matrix<double,9,1>::Zero();
	z = Matrix<double,6,1>::Zero();
	R = Matrix<double,6,6>::Zero();
	H = Matrix<double,6,9>::Zero();
	H.block<3,3>(0,0) = Matrix3d::Identity();
	K=Matrix<double,9,6>::Zero();
	P = Matrix<double,9,9>::Zero();
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
	
	std::cout << "Nonlinear CoM Estimator Initialized!" << std::endl;
}

Matrix<double,9,9> CoMEKF::computeTrans(Matrix<double,9,1> x_,  Vector3d COP_, Vector3d fN_, Vector3d L_)
{
  
    Matrix<double,9,9> res = Matrix<double,9,9>::Zero();
    
	res.block<3,3>(0,3) = Matrix3d::Identity();
	res(3, 6) = 1.000 / m;
	res(4, 7) = 1.000 / m;
	res(5, 8) = 1.000 / m;



	tmp = x_(2)-COP_(2);

   	res(3, 0) = (fN_(2)) / (m * tmp);

	res(3, 2) = -((fN_(2)) * (x_(0) - COP_(0))) / (m * tmp * tmp) +
	 L_(1) / (m * tmp * tmp);

	
	//res(3, 8) = (x_(0) - COP_(0)) / (m * tmp);
	
	res(4, 1) = (fN_(2)) / (m * tmp);

	res(4, 2) = - (fN_(2)) * ( x_(1) - COP_(1) ) / (m * tmp * tmp) -
	L_(0) / (m * tmp * tmp);
	
	
	//res(4, 8) = (x_(1) - COP_(1)) / (m * tmp);
    
    return res;
}


void CoMEKF::euler(Vector3d COP_, Vector3d fN_, Vector3d L_)
{
	F = computeTrans(x,  COP_,  fN_,  L_);
    //Euler Discretization - First order Truncation
	Fd = I;
	Fd.noalias() += F * dt ;
    
    /** Predict Step : Propagate the Mean estimate **/   
	//Forward Euler Integration of dynamics f

    f = computeDyn(x,  COP_,  fN_,  L_);
    x.noalias() += (f*dt);
}

Matrix<double,9,1> CoMEKF::computeDyn(Matrix<double,9,1> x_, Vector3d COP_, Vector3d fN_, Vector3d L_)
{
    Matrix<double,9,1> res = Matrix<double,9,1>::Zero();
    tmp = x_(2)-COP_(2);

	res.segment<3>(0) = x_.segment<3>(3);
    res(3) = (x_(0) - COP_(0)) / (m * tmp) * (fN_(2)) + x_(6) / m - L_(1) / (m * tmp); 
	res(4) = (x_(1) - COP_(1)) / (m * tmp) * (fN_(2)) + x_(7) / m + L_(0) / (m * tmp);
 	res(5) = (fN_(2) + x_(8)) / m - g;
    return res;
}


void CoMEKF::RK4(Vector3d COP_, Vector3d fN_, Vector3d L_, Vector3d COP0, Vector3d fN0, Vector3d L0)
{
    
    Matrix<double,9,1> k, k1, k2, k3, k4, x_mid, x0;
    Matrix<double,9,9> K1, K2, K3, K4, K0;
    Vector3d COP_mid, fN_mid, L_mid;
    

    k1 = Matrix<double,9,1>::Zero();
    k2 = Matrix<double,9,1>::Zero();
    k3 = Matrix<double,9,1>::Zero();
    k4 = Matrix<double,9,1>::Zero();
    K1 = Matrix<double,9,9>::Zero();
    K2 = Matrix<double,9,9>::Zero();
    K3 = Matrix<double,9,9>::Zero();
    K4 = Matrix<double,9,9>::Zero();
    
    x0 = x;
    //compute first coefficient
    k1 = computeDyn(x0,  COP0,  fN0,  L0);
    
    //Compute mid point with k1
    x_mid.noalias() = x0 + k1 * dt/2.00;
    COP_mid.noalias() = (COP_ + COP0)/2.00;
    fN_mid.noalias() = (fN_ + fN0)/2.00;
    L_mid.noalias() = (L_ + L0)/2.00;
    
    //Compute second coefficient
    k2 = computeDyn(x_mid,  COP_mid,  fN_mid,  L_mid);
    
    //Compute mid point with k2
    x_mid.noalias() = x0 + k2 * dt/2.00;
    //Compute third coefficient
    k3 = computeDyn(x_mid,  COP_mid,  fN_mid,  L_mid);
    
    //Compute point with k3
    x_mid.noalias() = x0 + k3 * dt;
    //Compute fourth coefficient
    k4 = computeDyn(x_mid, COP_,  fN_,  L_);
    

    //RK4 approximation of x
    k.noalias() =  (k1 + 2*k2 +2*k3 + k4)/6.00;

    //Compute the RK4 approximated mid point
    x_mid = x0;
    x_mid.noalias() += dt/2.00 * k;
    
    //Next state
    x.noalias() += dt * k;

    

    K1 = computeTrans(x0,COP0, fN0, L0);
    K2 = computeTrans(x_mid, COP_mid,  fN_mid,  L_mid);
    K3 = K2;
    
    K0  = I;
    K0.noalias() += dt/2.00 * K1;
    K2 = K2 * K0;
    
    K0 = I;
    K0.noalias() += dt/2.00 * K2;
    K3 = K3 * K0;
    
   
    //Compute the 4th Coefficient
    K4 =  computeTrans(x,  COP_,  fN_,  L_);
    K0 = I;
    K0.noalias() += dt * K3;
    K4 = K4 *  K0;
    
    //RK4 approximation of Transition Matrix
    Fd =  I;
    Fd.noalias() += (K1 + 2*K2 + 2*K3 + K4) * dt/6.00;
}








void CoMEKF::predict(Vector3d COP_, Vector3d fN_, Vector3d L_){
      
	COP = COP_;
	fN = fN_;
	L = L_;

	/*Predict Step*/

	if(useEuler)
	{
		euler(COP_,  fN_,  L_);
	}
	else
	{
		RK4(COP_,  fN_,  L_, COP_p, fN_p, L_p);
		COP_p = COP;
		fN_p = fN_;
		L_p = L_;
	}

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
	
	
	P = Fd * P * Fd.transpose();
	P.noalias() += Q*dt;


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

	z(3) = Acc(0) - ( (x(0) - COP(0)) / (m * tmp) * (fN(2) ) + x(6) / m - L(1) / (m * tmp) );
	z(4) = Acc(1) - ( (x(1) - COP(1)) / (m * tmp) * (fN(2) ) + x(7) / m + L(0) / (m * tmp) );
	z(5) = Acc(2) - ( (fN(2) + x(8)) / m - g );





	H(3, 0) = (fN(2) ) / (m * tmp);
	H(3, 2) =  -((fN(2) ) * (x(0) - COP(0))) / (m * tmp * tmp) +  L(1) / (m * tmp * tmp);

	//H(3, 8) = (x(0) - COP(0)) / (m * tmp);
	//H(4, 8) = (x(1) - COP(1)) / (m * tmp);

	H(4, 1) = (fN(2)) / (m * tmp);
	H(4, 2) = - ( fN(2)) * ( x(1) - COP(1) ) / (m * tmp * tmp) - L(0) / (m * tmp * tmp);
	
	H(3, 6) = 1.000 / m;
	H(4, 7) = 1.000 / m;
	H(5, 8) = 1.000 / m;


	R(0, 0) = com_r * com_r;
	R(1, 1) = R(0, 0);
	R(2, 2) = R(0, 0);
	
	R(3, 3) = comdd_r * comdd_r;
	R(4, 4) = R(3, 3);
	R(5, 5) = R(3, 3);
    R = R * dt;
	
    S = R;
	S.noalias() += H * P * H.transpose();
	K.noalias() = P * H.transpose() * S.inverse();

	x += K * z;
	P = (I - K * H) * P * (I - K * H).transpose();
	P.noalias() += K * R * K.transpose();
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

