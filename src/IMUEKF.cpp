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
    useEuler = true;
    If = Matrix<double, 15, 15>::Identity();
    P = Matrix<double, 15, 15>::Zero();
    //vel
    P(0,0) = 1e-2;
    P(1,1) = 1e-2;
    P(2,2) = 1e-2;
    //Rot error
    P(3,3) = 1e-3;
    P(4,4) = 1e-3;
    P(5,5) = 1e-3;
    //Pos
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
        
    //Innovation Vector
    z = Matrix<double, 6, 1>::Zero(); //For Odometry
    zv = Vector3d::Zero(); //For Twist
    
    
    
    //Initializing vectors and matrices
    r = Vector3d::Zero();
    v = Vector3d::Zero();
    dxf = Matrix<double, 15, 1>::Zero();
    
    
    
    temp = Vector3d::Zero();
    Kf = Matrix<double, 15, 6>::Zero();
    Kv = Matrix<double, 15, 3>::Zero();
    
    s = Matrix<double, 6, 6>::Zero();
    sv = Matrix<double, 3, 3>::Zero();
    
    R = Matrix<double, 6, 6>::Zero();
    Rv = Matrix<double, 3, 3>::Zero();
    
    Acf = Matrix<double, 15, 15>::Zero();
    Lcf = Matrix<double, 15, 12>::Zero();
    Qff = Matrix<double, 15, 15>::Zero();
    Qf = Matrix<double, 12, 12>::Zero();
    Af = Matrix<double, 15, 15>::Zero();
    
    bw = Vector3d::Zero();
    bf = Vector3d::Zero();
    gyro = Vector3d::Zero();
    acc = Vector3d::Zero();
    angle = Vector3d::Zero();
    //bias removed acceleration and gyro rate
    fhat = Vector3d::Zero();
    omegahat = Vector3d::Zero();
    f_p = Vector3d::Zero();
    omega_p = Vector3d::Zero();
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

void IMUEKF::computeRK4(Matrix3d& Rib_,  Matrix<double,15,1>& x_,  Matrix<double,15,15>& A_trans, Vector3d omega_, Vector3d f_, Vector3d omega0, Vector3d f0)
{
    
    Matrix<double,15,1> k, k1, k2, k3, k4, x_mid, x0;
    Matrix<double,15,15> K1, K2, K3, K4, K0;
    Vector3d f_mid, omega_mid;
    Matrix3d Rib_mid, Rib0;
    

    Rib_mid = Matrix3d::Identity();
    k1 = Matrix<double,15,1>::Zero();
    k2 = Matrix<double,15,1>::Zero();
    k3 = Matrix<double,15,1>::Zero();
    k4 = Matrix<double,15,1>::Zero();
    K1 = Matrix<double,15,15>::Zero();
    K2 = Matrix<double,15,15>::Zero();
    K3 = Matrix<double,15,15>::Zero();
    K4 = Matrix<double,15,15>::Zero();
    
    x0 = x_;
    Rib0 = Rib_;
    //compute first coefficient
    k1 = computeDyn(x0,Rib0, omega0, f0);
    
    //Compute mid point with k1
    x_mid.noalias() = x0 + k1 * dt/2.00;
    omega_mid.noalias() = (omega_ + omega0)/2.00;
    Rib_mid.noalias() = Rib0 * expMap(omega_mid);
    f_mid.noalias() = (f_ + f0)/2.00;
    
    //Compute second coefficient
    k2 = computeDyn(x_mid,Rib_mid, omega_mid, f_mid);
    
    //Compute mid point with k2
    x_mid.noalias() = x0 + k2 * dt/2.00;
    //Compute third coefficient
    k3 = computeDyn(x_mid, Rib_mid, omega_mid, f_mid);
    
    //Compute point with k3
    x_mid.noalias() = x0 + k3 * dt;
    Rib_mid.noalias() = Rib0 * expMap(omega_);
    //Compute fourth coefficient
    k4 = computeDyn(x_mid,Rib_mid, omega_, f_);
    

    //RK4 approximation of x
    k.noalias() =  (k1 + 2*k2 +2*k3 + k4)/6.00;

    //Compute the RK4 approximated mid point
    x_mid = x0;
    x_mid.noalias() += dt/2.00 * k;
    Rib_mid.noalias() = Rib0 * expMap(omega_mid);
    
    //Next state
    x_.noalias() += dt * k;

    

    K1 = computeTrans(x0,Rib0, omega0, f0);
    K2 = computeTrans(x_mid,Rib_mid, omega_mid, f_mid);
    K3 = K2;
    
    K0  = If;
    K0.noalias() += dt/2.00 * K1;
    K2 = K2 * K0;
    
    K0 = If;
    K0.noalias() += dt/2.00 * K2;
    K3 = K3 * K0;
    
    //Update Rotation
    temp.noalias() = omega_- x_.segment<3>(9);
    temp *= dt;
    if(temp(0)!=0 && temp(1) !=0 && temp(2)!=0)
        Rib_ *=  expMap(temp);
    //Compute the 4th Coefficient
    K4 =  computeTrans(x_, Rib_, omega_, f_);
    K0 = If;
    K0.noalias() += dt * K3;
    K4 = K4 *  K0;
    
    //RK4 approximation of Transition Matrix
    A_trans =  If;
    A_trans.noalias() += (K1 + 2*K2 + 2*K3 + K4) * dt/6.00;
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


void IMUEKF::euler(Vector3d omega_, Vector3d f_)
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
    fhat.noalias() = f - bf;
    omegahat.noalias() = omega - bw;
    
    /** Linearization **/
    //Transition matrix Jacobian
    Acf.block<3,3>(0,0).noalias() = -wedge(omegahat);
    Acf.block<3,3>(0,3).noalias() = wedge(Rib.transpose() * g);
    Acf.block<3,3>(3,3).noalias() = -wedge(omegahat);
    Acf.block<3,3>(6,0) = Rib;
    Acf.block<3,3>(6,3).noalias() = -Rib * wedge(v);
    Acf.block<3,3>(0,9).noalias() = -wedge(v);
    Acf.block<3,3>(0,12) = -Matrix3d::Identity();
    Acf.block<3,3>(3,9) = -Matrix3d::Identity();
    
    
    
    
    //State Noise Jacobian
    //gyro (0),acc (3),gyro_bias (6),acc_bias (9)
    Lcf.block<3,3>(0,0).noalias() = wedge(v);
    Lcf.block<3,3>(0,3) = Matrix3d::Identity();
    Lcf.block<3,3>(3,0) = Matrix3d::Identity();
    Lcf.block<3,3>(9,6) = Matrix3d::Identity();
    Lcf.block<3,3>(12,9) = Matrix3d::Identity();
    
    
    
    
    // Covariance Q with full state + biases
    Qf(0, 0) = gyr_qx * gyr_qx  * dt;
    Qf(1, 1) = gyr_qy * gyr_qy  * dt;
    Qf(2, 2) = gyr_qz * gyr_qz  * dt;
    Qf(3, 3) = acc_qx * acc_qx  * dt;
    Qf(4, 4) = acc_qy * acc_qy  * dt;
    Qf(5, 5) = acc_qz * acc_qz  * dt;
    Qf(6, 6) = gyrb_qx * gyrb_qx ;
    Qf(7, 7) = gyrb_qy * gyrb_qy ;
    Qf(8, 8) = gyrb_qz * gyrb_qz  ;
    Qf(9, 9) = accb_qx * accb_qx  ;
    Qf(10, 10) = accb_qy * accb_qy ;
    Qf(11, 11) = accb_qz * accb_qz ;
    
    
    //Euler Discretization - First order Truncation
    Af = If;
    Af.noalias() += Acf * dt;
    Qff.noalias() =  Lcf * Qf * Lcf.transpose() * dt ;
    //Qff =  Af * Lcf * Qf * Lcf.transpose() * Af.transpose() * dt ;
    
    /** Predict Step: Propagate the Error Covariance  **/
    P = Af * P * Af.transpose() + Qff;
    
    /** Predict Step : Propagate the Mean estimate **/
    //Body Velocity
    
    temp.noalias() = v.cross(omegahat) + Rib.transpose() * g + fhat;
    temp *= dt;
    
    x(0) = v(0) + temp(0);
    x(1) = v(1) + temp(1);
    x(2) = v(2) + temp(2);
    
    x(3) = 0;
    x(4) = 0;
    x(5) = 0;
    
    //Body position
    temp.noalias() = Rib * v;
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
    if (omegahat(0) != 0 && omegahat(1) != 0 && omegahat(2) != 0)
    {
        Rib  *=  expMap(omegahat*dt);
    }

}




/** IMU EKF filter to  deal with the Noise **/
void IMUEKF::predict(Vector3d omega_, Vector3d f_)
{
    if(useEuler)
        euler(omega_,f_);
    else
    {
        //Store the input
        f_p = f;
        omega_p = omega;
    }
    updateVars();
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
    zv = y;
    zv.noalias() -= Rib * v;
    
    Hv.block<3,3>(0,0) = Rib;
    Hv.block<3,3>(0,3).noalias() = -Rib * wedge(v);
    sv = Rv;
    sv.noalias() += Hv * P * Hv.transpose();
    Kv.noalias() = P * Hv.transpose() * sv.inverse();
    
    dxf.noalias() = Kv * zv;
    
    //Update the mean estimate
    x += dxf;
    
    //Update the error covariance
    P = (If - Kv * Hv) * P * (If - Hv.transpose()*Kv.transpose());
    P.noalias()+= Kv * Rv * Kv.transpose();
    
    
    if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
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
    
    
    if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
    {
        Rib *=  expMap(dxf.segment<3>(3));
    }
    x.segment<3>(3) = Vector3d::Zero();
    
    updateVars();
    
}



void IMUEKF::updateVars()
{
    
    
    Tib.linear() = Rib;
    Tib.translation() = x.segment<3>(6);
    qib = Quaterniond(Tib.linear());
    //Update the biases
    bgyr = x.segment<3>(9);
    bacc = x.segment<3>(12);

    bias_gx = x(9);
    bias_gy = x(10);
    bias_gz = x(11);
    bias_ax = x(12);
    bias_ay = x(13);
    bias_az = x(14);
    
    
    omegahat = omega - bgyr;
    fhat = f - bacc;
    
    gyro  = Rib * omegahat;
    gyroX = gyro(0);
    gyroY = gyro(1);
    gyroZ = gyro(2);
    
    acc =  Rib * fhat;
    accX = acc(0);
    accY = acc(1);
    accZ = acc(2);
    
    vel = Rib * x.segment<3>(0);
    velX = vel(0);
    velY = vel(1);
    velZ = vel(2);
    
    
    pos = x.segment<3>(6);
    rX = x(6);
    rY = x(7);
    rZ = x(8);
    
    
    //ROLL - PITCH - YAW
    angle = getEulerAngles(Rib);
    angleX = angle(0);
    angleY = angle(1);
    angleZ = angle(2);
}
