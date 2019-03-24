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

#include <serow/IMUinEKF.h>


IMUinEKF::IMUinEKF()
{
    //Gravity Vector
    g = Vector3d::Zero();
    g(2) = -9.80;
}

void IMUinEKF::init() {
    
    
    firstrun = true;
    If = Matrix<double, 21, 21>::Identity();
    I = Matrix3d::Identity();
    P = Matrix<double, 21, 21>::Zero();
    X = Matrix<double,7,7>::Identity();
    /* State is X where
    X(0:2,0:2) is Rwb 
    X(0:2,3) is v_wb 
    X(0:2,4) is p_wb
    X(0:2,5) is dR
    X(0:2,6) is dL
    Parameters are theta where
    theta(0:2) is Gyro Bias
    thata(3:5) is Acc Bias
    */

    //Rot error
    P(0,0) = 1e-5;
    P(1,1) = 1e-5;
    P(2,2) = 1e-5;
    //Vel error
    P(3,3) = 1e-5;
    P(4,4) = 1e-5;
    P(5,5) = 1e-5;
    //Pos error
    P(6,6)  = 1e-6;
    P(7,7)  = 1e-6;
    P(8,8)  = 1e-6;
    //dR error
    P(9,9)  = 1e-6;
    P(10,10)  = 1e-6;
    P(11,11)  = 1e-6;
    //dL error
    P(12,12)  = 1e-6;
    P(13,13)  = 1e-6;
    P(14,14)  = 1e-6;
    //Biases
    P(15, 15) = 1e-6;
    P(16, 16) = 1e-6;
    P(17, 17) = 1e-6;
    P(18, 19) = 1e-5;
    P(19, 19) = 1e-5;
    P(20, 20) = 1e-5;
    
    Af = Matrix<double,21,21>::Zero();
    Af.block<3,3>(3,0).noalias() = skew(g);
    Af.block<3,3>(6,3).noalias() = Matrix3d::Identity();

    Qff = Matrix<double, 21, 21>::Zero();
    Qf = Matrix<double, 21, 21>::Zero();
    Qc = Matrix3d::Zero();
    Adj = Matrix<double, 21, 21>::Zero();
    Phi = Matrix<double, 21, 21>::Zero();
    Rwb = Matrix3d::Identity();
    pwb = Vector3d::Zero();
    vwb = Vector3d::Zero();
    dL = Vector3d::Zero();
    dR = Vector3d::Zero();
    bgyr = Vector3d::Zero();
    bacc = Vector3d::Zero();
    gyro = Vector3d::Zero();
    acc = Vector3d::Zero();
    angle = Vector3d::Zero();
    
    
    
    
    std::cout << "IMU EKF Initialized Successfully" << std::endl;
}






void IMUinEKF::constructState(Matrix<double,7,7>&X_, Matrix<double,6,1> &theta_, Matrix3d R_, Vector3d v_, Vector3d  p_, Vector3d dR_,  Vector3d dL_, Vector3d bg_, Vector3d ba_)
{
    X_=  Matrix<double,7,7>::Identity();
    X_.block<3,3>(0,0) = R_;
    X_.block<1,3>(0,3) = v_;
    X_.block<1,3>(0,4) = p_;
    X_.block<1,3>(0,5) = dR_;
    X_.block<1,3>(0,6) = dL_;

    theta_.segment<3>(0) = bg_;
    theta_.segment<3>(3) = ba_;
  
}

void IMUinEKF::seperateState(Matrix<double,7,7>X_, Matrix<double,6,1> theta_, Matrix3d& R_, Vector3d& v_, Vector3d&  p_, Vector3d& dR_,  Vector3d& dL_, Vector3d& bg_, Vector3d& ba_)
{
    
    R_ = X_.block<3,3>(0,0);
    v_ =  X_.block<1,3>(0,3);
    p_ = X_.block<1,3>(0,4);
    dR_ = X_.block<1,3>(0,5);
    dL_ = X_.block<1,3>(0,6);

    bg_ = theta_.segment<3>(0);
    ba_ = theta_.segment<3>(3);
  
}

Matrix<double,7,7> IMUinEKF::exp(Matrix<double,15,1> v)
{
    Matrix<double,7,7> dX = Matrix<double,7,7>::Identity();
    Matrix3d A = skew(v.segment<3>(0));
    Matrix3d A2 = A * A;
    double phi = v.segment<3>(0).norm();
    double phi2 = phi * phi;
    Matrix3d R = Matrix3d::Identity();
    Matrix3d Jr = Matrix3d::Identity();
    
    if(phi >=1e-6)
    {
        R.noalias() += sin(phi)/phi*A;
        R.noalias() += (1-cos(phi))/(phi2) * A2;

        Jr.noalias() += (1-cos(phi))/(phi2)*A;
        Jr.noalias() += (phi-sin(phi))/(phi2*phi)*A2;
    }
    dX.block<3,3>(0,0) = R;
    dX.block<3,3>(0,3).noalias() = Jr * v.segment<3>(3);
    dX.block<3,3>(0,6).noalias() = Jr * v.segment<3>(6);
    dX.block<3,3>(0,9).noalias() = Jr * v.segment<3>(9);
    dX.block<3,3>(0,12).noalias() = Jr * v.segment<3>(12);
    return dX;
}

Matrix3d IMUinEKF::exp_SO3(Vector3d v);
{
    Matrix3d R = Matrix3d::Identity();
    Matrix3d A = skew(v);
    double phi = v.norm();
    if(phi>=1e-6)
    {
       R.noalias() += (sin(phi)/phi)*A;
       R.noalias() += (1-cos(phi))/(phi*phi)*A*A;
    }

    return R;
}

Matrix<double,21,21> IMUinEKF::Adjoint(Matrix<double,7,7> X_)
{
    Matrix<double,21,21> AdjX = Matrix<double,21,21>::Identity();
    Matrix3d R_;
    Vector3d v_,p_,dR_,dL_,bg_,ba_;
    seperateState(X_, Matrix<double,6,1>::Zero(),  R_,  v_,   p_,  dR_,   dL_,  bg_,  ba_);
   
    AdjX.block<3,3>(0,0) = R_;
    AdjX.block<3,3>(3,3) = R_;
    AdjX.block<3,3>(6,6) = R_;
    AdjX.block<3,3>(9,9) = R_;
    AdjX.block<3,3>(12,12) = R_;


    AdjX.block<3,3>(3,0).noalias() = skew(v_) * R_;
    AdjX.block<3,3>(6,0).noalias() = skew(p_) * R_;
    AdjX.block<3,3>(9,0).noalias() = skew(dR_) * R_;
    AdjX.block<3,3>(12,0).noalias() = skew(dL_) * R_;

    return AdjX;

}

void IMUinEKF::predict(Vector3d angular_velocity, Vector3d linear_acceleration, Vector3d hR_R, Vector3d hR_L, int contactR, int, contactL)
{

   seperateState(X,theta,Rib,vwb,pwb,dR,dL,bgyr,bacc)
   w_ = angular_velocity;
   a_ = linear_acceleration;

   //Bias removed gyro and acc
   w -= bgyr;   
   a -= bacc;

   
    
    Af.block<3,3>(3,15).noalias() = -Rwb;
    Af.block<3,3>(6,18).noalias() = -Rwb;
    Af.block<3,3>(6,15).noalias() = -skew(vwb) * Rwb;
    Af.block<3,3>(9,15).noalias() = -skew(pwb) * Rwb;
    Af.block<3,3>(12,15).noalias() = -skew(dR) * Rwb;
    Af.block<3,3>(15,15).noalias() = -skew(dL) * Rwb;    
    
    
    Phi = If + Af*dt;
    Adj = Adjoint(X);
    

    
    
    // Covariance Q with full state + biases
    Qf(0, 0) = gyr_qx * gyr_qx * dt ;
    Qf(1, 1) = gyr_qy * gyr_qy * dt ;
    Qf(2, 2) = gyr_qz * gyr_qz * dt ;
    Qf(3, 3) = acc_qx * acc_qx * dt ;
    Qf(4, 4) = acc_qy * acc_qy * dt ;
    Qf(5, 5) = acc_qz * acc_qz * dt ;
    
    Qc(0,0) = foot_contactx * foot_contacty;
    Qc(1,1) = foot_contacty * foot_contacty;
    Qc(2,2) = foot_contactz * foot_contactz;

    Qf.block<3,3>(9,9) = hR_R*(Qc+1e4*I*(1-contactR))*hR_R.transpose();
    Qf.block<3,3>(12,12) = hR_L*(Qc+1e4*I*(1-contactL))*hR_L.transpose();




    Qf(15, 15) = gyrb_qx * gyrb_qx ;
    Qf(16, 16) = gyrb_qy * gyrb_qy ;
    Qf(17, 17) = gyrb_qz * gyrb_qz ;
    Qf(18, 18) = accb_qx * accb_qx ;
    Qf(19, 19) = accb_qy * accb_qy ;
    Qf(20, 20) = accb_qz * accb_qz ;
    
    
    Qff.noalias() =  Phi * Adj * Qf * Adj.transpose() * Phi.transpose() * dt ;
    
    /** Predict Step: Propagate the Error Covariance  **/
    P = Af * P * Af.transpose();
    P.noalias() += Qff;
    
    pwb += vwb* dt + 0.5 * (Rwb * a + g)*dt * dt;
    vwb += (Rwb*a + g)*dt
    Rwb *= exp_SO3(w*dt);

    constructState(X,theta, Rwb, vwb, pwb, dR, dL, bgyr, bacc);
    updateVars();
}




void IMUinEKF::updateStateSingleContact(Matrix<double,7,1> Y, Matrix<double,7,1> b, Matrix<double,3,21> H, Matrix3d N, Matrix<double,3,7> PI)
{

    Matrix3d S = N;
    S += H * P * H.transpose();

    Matrix<double,21,3> K = P * H * S.inverse();

    Matrix<double,7,7> BigX = Matrix<double,7,7>::Zero();
    BigX.block<7,7>(0,6) = X;
  
    
    Matrix<double,7,1> Z =  BigX * Y - b;


    //Update State
    Matrix<double,21,1> delta = K * PI * Z;
    Matrix<double,7,7> dX = exp(delta.segment<15>(0));
    Matrix<double,6,1> dtheta = delta.segment<6>(15);
    X = dX * X;
    theta += dtheta;

    Matrix<double,21,21> IKH = If - K*H;
    P = IKH * P * IKH.transpose() + K * N * K.transpose();
}

void IMUinEKF::updateStateDoubleContact(Matrix<double,14,1>Y, Matrix<double,14,1> b, Matrix<double,6,21> H, Matrix<double,6,6> N, Matrix<double,6,14> PI)
{
    Matrix<double,6,6> S = N;
    S += H * P * H.transpose();

    Matrix<double,21,6> K = P * H * S.inverse();

    Matrix<double,14,14> BigX = Matrix<double,14,14>::Zero();
    
    int i = 1;
    while(i<=2)
    {
        BigX.block<7,7>(7*(i-1),7*i-1) = X;
        i++;
    }
    Matrix<double,14,1> Z =  BigX * Y - b;


    //Update State
    Matrix<double,21,1> delta = K * PI * Z;
    Matrix<double,7,7> dX = exp(delta.segment<15>(0));
    Matrix<double,6,1> dtheta = delta.segment<6>(15);
    X = dX * X;
    theta += dtheta;

    Matrix<double,21,21> IKH = If - K*H;
    P = IKH * P * IKH.transpose() + K * N * K.transpose();
}


void IMUinEKF::updateKinematics(Vector3d s_pR, Vector3d s_pL, Matrix3d JRQeJR, Matrix3d JLQeJL, int contactL, int contactR)
{
   Rwb = X.block<3,3>(0,0);
   if(contactL && contactR)
   {
       Matrix<double,14,1> Y = Matrix<double,14,1>::Zeros();
       Y.segment<3>(0) = s_pR;
       Y(4) = 1.00;
       Y(5) = -1.00;
       Y.segment<3>(7) = s_pL;
       Y(11) = 1.00;
       Y(13) = -1.00;
       Matrix<double,14,1> b = Matrix<double,14,1>::Zeros();
       b(4) = 1.00;
       b(5) = -1.00;
       b(11) = 1.00;
       b(13) = -1.00;
       Matrix<double,6,21> H = Matrix<double,6,21>::Zero();
       H.block<3,3>(0,6) = -Matrix3d::Identity();
       H.block<3,3>(0,9) = Matrix3d::Identity();
       H.block<3,3>(3,6) = -Matrix3d::Identity();
       H.block<3,3>(3,12) = Matrix3d::Identity();

       Matrix<double,6,6> N = Matrix<double,6,6>::Zero();
       N.block<3,3>(0,0) = Rwb * JRQeJR * Rwb.transpose() +  R;
       N.block<3,3>(3,3) = Rwb * JLQeJL * Rwb.transpose() +  R;
       Matrix<double,6,14> PI = Matrix<double,6,14>::Zero();
       PI.block<3,3>(0,0) = Matrix3d::Identity();
       PI.block<3,3>(3,3) = Matrix3d::Identity();  
       updateStateDoubleContact(Y,b,H,N,PI);
   }
   else if(contactR)
   {
       Matrix<double,7,1> Y = Matrix<double,7,1>::Zeros();
       Y.segment<3>(0) = s_pR;
       Y(4) = 1.00;
       Y(5) = -1.00;
       Matrix<double,7,1> b = Matrix<double,7,1>::Zeros();
       b(4) = 1.00;
       b(5) = -1.00;
     
       Matrix<double,3,21> H = Matrix<double,3,21>::Zero();
       H.block<3,3>(0,6) = -Matrix3d::Identity();
       H.block<3,3>(0,9) = Matrix3d::Identity();
      
       Matrix3d N = Matrix3d::Zero();
       N = Rwb * JRQeJR * Rwb.transpose() +  R;
       Matrix<double,3,14> PI = Matrix<double,3,14>::Zero();
       PI.block<3,3>(0,0) = Matrix3d::Identity();
       updateStateSingleContact(Y,b,H,N,PI);
   }
   else if(contactL)
   {
       Matrix<double,7,1> Y = Matrix<double,7,1>::Zeros();
       Y.segment<3>(0) = s_pR;
       Y(4) = 1.00;
       Y(6) = -1.00;
       Matrix<double,7,1> b = Matrix<double,7,1>::Zeros();
       b(4) = 1.00;
       b(6) = -1.00;
     
       Matrix<double,3,21> H = Matrix<double,3,21>::Zero();
       H.block<3,3>(0,6) = -Matrix3d::Identity();
       H.block<3,3>(0,12) = Matrix3d::Identity();
      
       Matrix3d N = Matrix3d::Zero();
       N = Rwb * JLQeJL * Rwb.transpose() +  R;
       Matrix<double,3,14> PI = Matrix<double,3,14>::Zero();
       PI.block<3,3>(0,0) = Matrix3d::Identity();
       updateStateSingleContact(Y,b,H,N,PI);
   }
}

void IMUinEKF::updateVars()
{

    seperateState(X,theta, Rwb, vwb, pwb, dR, dL, bgyr, bacc);
    rX = pwb(0);
    rY = pwb(1);
    rZ = pwb(2);
    Tib.linear() = Rwb;
    Tib.translation() = pwb;
    qib = Quaterniond(Rwb);
    Rib = Rwb;
    
    //Update the biases
    bgyr = x.segment<3>(9);
    bacc = x.segment<3>(12);
    //std::cout<<"bacc"<<std::endl;
    //std::cout<<bacc<<std::endl;
    //std::cout<<"bgyr"<<std::endl;
    //std::cout<<bgyr<<std::endl;
    bias_gx = bgyr(0);
    bias_gy = bgyr(1);
    bias_gz = bgyr(2);
    bias_ax = bacc(0);
    bias_ay = bacc(1);
    bias_az = bacc(2);
    
    
    w = w_ - bgyr;
    a = a_ - bacc;

    gyro  = Rwb * w;
    gyroX = gyro(0);
    gyroY = gyro(1);
    gyroZ = gyro(2);
    
    acc =  (Rwb * a + g);
    accX = acc(0);
    accY = acc(1);
    accZ = acc(2);
    
    velX = vwb(0);
    velY = vwb(1);
    velZ = vwb(2);
    
    
    //ROLL - PITCH - YAW
    angle = getEulerAngles(Rwb);
    angleX = angle(0);
    angleY = angle(1);
    angleZ = angle(2);
}
