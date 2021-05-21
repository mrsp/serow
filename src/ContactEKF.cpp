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

#include <serow/ContactEKF.h>

ContactEKF::ContactEKF()
{
    //Gravity Vector
    g = Vector3d::Zero();
    g(2) = -9.80;
}

void ContactEKF::init()
{

    firstrun = true;
    useEuler = true;
    If = Matrix<double, 27, 27>::Identity();
    P = Matrix<double, 27, 27>::Zero();
    //  velocity error in base coordinates
    P(0, 0) = 1e-3;
    P(1, 1) = 1e-3;
    P(2, 2) = 1e-3;
    //  Rotetional error in base coordinates
    P(3, 3) = 1e-3;
    P(4, 4) = 1e-3;
    P(5, 5) = 1e-3;
    //  Positional error in world coordinates
    P(6, 6) = 1e-5;
    P(7, 7) = 1e-5;
    P(8, 8) = 1e-5;
    //  Gyro and Acc Biases
    P(9, 9) = 1e-3;
    P(10, 10) = 1e-3;
    P(11, 11) = 1e-3;
    P(12, 12) = 1e-3;
    P(13, 13) = 1e-3;
    P(14, 14) = 1e-3;
    // Left foot error in world coordinates
    P(15, 15) = 1e-5;
    P(16, 16) = 1e-5;
    P(17, 17) = 1e-5;
    //  Left foot Rotetional error in foot coordinates
    P(18, 18) = 1e-3;
    P(19, 19) = 1e-3;
    P(20, 20) = 1e-3;
    // Right foot error in world coordinates
    P(21, 21) = 1e-5;
    P(22, 22) = 1e-5;
    P(23, 23) = 1e-5;
    //  Right foot Rotetional error in foot coordinates
    P(24, 24) = 1e-3;
    P(25, 25) = 1e-3;
    P(26, 26) = 1e-3;

    // Construct Measurement Model Linerazition
    Hcf = Matrix<double, 12, 27>::Zero();
    Hf = Matrix<double, 6, 27>::Zero();
    Hf.block<3, 3>(0, 6) = Matrix3d::Identity();
    Hf.block<3, 3>(3, 3) = Matrix3d::Identity();
    Hvf = Matrix<double, 6, 27>::Zero();
    Hvf.block<3, 3>(3, 3) = Matrix3d::Identity();
    Hv = Matrix<double, 3, 27>::Zero();

    //   Rotation Matrix from base to world frame initialization
    Rib = Matrix3d::Identity();
    qib = Quaterniond(Rib);
    //   Rotation Matrix from left foot to world frame initialization
    Ril = Matrix3d::Identity();
    qil = Quaterniond(Ril);
    //   Rotation Matrix from right foot to world frame initialization
    Rir = Matrix3d::Identity();
    qir = Quaterniond(Rir);

    // State Vector
    x = Matrix<double, 27, 1>::Zero();
    //  Innovation Vector
    //  For Odometry Update
    z = Matrix<double, 6, 1>::Zero();
    //  For Twist Update
    zv = Vector3d::Zero();
    // For Contact Update
    zf = Matrix<double, 12, 1>::Zero();

    // Initializing rest vectors and matrices needed by the filter
    v = Vector3d::Zero();
    dxf = Matrix<double, 27, 1>::Zero();
    Kf = Matrix<double, 27, 6>::Zero();
    Kv = Matrix<double, 27, 3>::Zero();

    s = Matrix<double, 6, 6>::Zero();
    sv = Matrix<double, 3, 3>::Zero();
    sc = Matrix<double, 12,12>::Zero();

    R = Matrix<double, 6, 6>::Zero();
    Rv = Matrix<double, 3, 3>::Zero();
    Rc = Matrix<double, 12, 12>::Zero();

    Acf = Matrix<double, 27, 27>::Zero();
    Qff = Matrix<double, 27, 27>::Zero();
    Qf = Matrix<double, 24, 24>::Zero();
    Af = Matrix<double, 27, 27>::Zero();

    bgyr = Vector3d::Zero();
    bacc = Vector3d::Zero();
    gyro = Vector3d::Zero();
    acc = Vector3d::Zero();
    angle = Vector3d::Zero();

    //  bias removed acceleration and gyro rate
    fhat = Vector3d::Zero();
    omegahat = Vector3d::Zero();

    //  Compute some parts of the Input-Noise Jacobian once since they are constants
    //  gyro (0),acc (3),gyro_bias (6),acc_bias (9), lpos (12), lorient (15), rpos (18), rorient (21)
    Lcf = Matrix<double, 27, 24>::Zero();
    Lcf.block<3, 3>(0, 3) = -Matrix3d::Identity();
    Lcf.block<3, 3>(3, 0) = -Matrix3d::Identity();
    Lcf.block<3, 3>(9, 6) = Matrix3d::Identity();
    Lcf.block<3, 3>(12, 9) = Matrix3d::Identity();
    Lcf.block<3, 3>(18, 15) = Matrix3d::Identity();
    Lcf.block<3, 3>(24, 21) = Matrix3d::Identity();

    //  Output Variables
    //  roll-pitch-yaw in world coordinates
    angleX = 0.000;
    angleY = 0.000;
    angleZ = 0.000;
    //  angular velocity in world coordinates
    gyroX = 0.000;
    gyroY = 0.000;
    gyroZ = 0.000;
    //  absolute linear acceleration in world coordinates
    accX = 0.000;
    accY = 0.000;
    accZ = 0.000;
    //  base position in world coordinates
    rX = 0.000;
    rY = 0.000;
    rZ = 0.000;
    //  base velocity in world coordinates
    velX = 0.000;
    velY = 0.000;
    velZ = 0.000;
    //  base transformation to the world frame
    Tib = Affine3d::Identity();
    //  Left foot transformation to the world frame
    Til = Affine3d::Identity();
    //  Left foot transformation to the world frame
    Tir = Affine3d::Identity();

    std::cout << "Contact EKF Initialized Successfully" << std::endl;
}

Matrix<double, 27, 27> ContactEKF::computeTrans(Matrix<double, 27, 1> x_, Matrix<double, 3, 3> Rib_, Vector3d omega_, Vector3d f_)
{
    omega_.noalias() -= x_.segment<3>(9);
    f_.noalias() -= x_.segment<3>(12);
    v = x_.segment<3>(0);
    Matrix<double, 27, 27> res = Matrix<double, 27, 27>::Zero();
    res.block<3, 3>(0, 0).noalias() = -wedge(omega_);
    res.block<3, 3>(0, 3).noalias() = wedge(Rib_.transpose() * g);
    res.block<3, 3>(0, 12).noalias() = -Matrix3d::Identity();
    res.block<3, 3>(0, 9).noalias() = -wedge(v);
    res.block<3, 3>(3, 3).noalias() = -wedge(omega_);
    res.block<3, 3>(3, 9).noalias() = -Matrix3d::Identity();
    res.block<3, 3>(6, 0) = Rib_;
    res.block<3, 3>(6, 3).noalias() = -Rib_ * wedge(v);
    return res;
}

void ContactEKF::euler(Vector3d omega_, Vector3d f_, Vector3d pbl_, Vector3d pbr_,  int contactL_, int contactR_)
{
    Acf = computeTrans(x, Rib, omega_, f_);
    //Euler Discretization - First order Truncation
    Af = If;
    Af.noalias() += Acf * dt;
    x = computeDiscreteDyn(x, Rib, omega_, f_, pbl_,  pbr_,  contactL_,  contactR_ );
    //x.noalias() += computeContinuousDyn(x,Rib,omega_,f_)*dt;
}

Matrix<double, 27, 1> ContactEKF::computeContinuousDyn(Matrix<double, 27, 1> x_, Matrix<double, 3, 3> Rib_, Vector3d omega_, Vector3d f_)
{
    Matrix<double, 27, 1> res = Matrix<double, 27, 1>::Zero();

    // Inputs without bias
    omega_ -= x_.segment<3>(9);
    f_ -= x_.segment<3>(12);

    // Nonlinear Process Model
    v = x_.segment<3>(0);
    res.segment<3>(0).noalias() = v.cross(omega_);
    res.segment<3>(0).noalias() += Rib_.transpose() * g;
    res.segment<3>(0) += f_;
    res.segment<3>(6).noalias() = Rib_ * v;

    return res;
}

Matrix<double, 27, 1> ContactEKF::computeDiscreteDyn(Matrix<double, 27, 1> x_, Matrix<double, 3, 3> Rib_, Vector3d omega_, Vector3d f_, Vector3d pbl_, Vector3d pbr_, int contactL_, int contactR_)
{

    Matrix<double, 27, 1> res = Matrix<double, 27, 1>::Zero();

    omega_.noalias() -= x_.segment<3>(9);
    f_.noalias() -= x_.segment<3>(12);

    //Nonlinear Process Model


    //Compute \dot{v}_b @ k
    v = x_.segment<3>(0);
    res.segment<3>(0).noalias() = v.cross(omega_);
    res.segment<3>(0).noalias() += Rib_.transpose() * g;
    res.segment<3>(0) += f_;

    //Position
    r = x_.segment<3>(6);
    res.segment<3>(6).noalias() = Rib_ * res.segment<3>(0) * dt * dt / 2.00;
    res.segment<3>(6).noalias() += Rib_ * v * dt;
    res.segment<3>(6) += r;

    //Velocity
    res.segment<3>(0) *= dt;
    res.segment<3>(0) += v;

    //Biases
    res.segment<3>(9) = x_.segment<3>(9);
    res.segment<3>(12) = x_.segment<3>(12);

    //Left Foot
    res.segment<3>(15) = contactL_ * x_.segment<3>(15) + (1 - contactL_) * (x_.segment<3>(6) + Rib_ * pbl_);
    //Right Foot
    res.segment<3>(21) = contactR_ * x_.segment<3>(21) + (1 - contactR_) * (x_.segment<3>(6) + Rib_ * pbr_);


    return res;
}

void ContactEKF::predict(Vector3d omega_, Vector3d f_, Vector3d pbl_, Vector3d pbr_, Matrix3d Rbl_, Matrix3d Rbr_, int contactL_, int contactR_)
{
    omega = omega_;
    f = f_;
    //Used in updating Rib with the Rodriquez formula
    omegahat.noalias() = omega - x.segment<3>(9);
    v = x.segment<3>(0);

    //Update the Input-noise Jacobian
    Lcf.block<3, 3>(0, 0).noalias() = -wedge(v);
    Lcf.block<3, 3>(15, 12) = Ril;
    Lcf.block<3, 3>(21, 18) = Rir;

    euler(omega_, f_, pbl_,  pbr_,  contactL_,  contactR_);

    // Covariance Q with full state + biases
    Qf(0, 0) = gyr_qx * gyr_qx;
    Qf(1, 1) = gyr_qy * gyr_qy;
    Qf(2, 2) = gyr_qz * gyr_qz;

    Qf(3, 3) = acc_qx * acc_qx;
    Qf(4, 4) = acc_qy * acc_qy;
    Qf(5, 5) = acc_qz * acc_qz;

    Qf(6, 6) = gyrb_qx * gyrb_qx;
    Qf(7, 7) = gyrb_qy * gyrb_qy;
    Qf(8, 8) = gyrb_qz * gyrb_qz;

    Qf(9, 9) = accb_qx * accb_qx;
    Qf(10, 10) = accb_qy * accb_qy;
    Qf(11, 11) = accb_qz * accb_qz;

    Qf(12,12) = lp_qx * lp_qx + 1e4 * (1 - contactL_);
    Qf(13,13) = lp_qy * lp_qy + 1e4 * (1 - contactL_);
    Qf(14,14) = lp_qz * lp_qz + 1e4 * (1 - contactL_);

    Qf(15,15) = lo_qx * lo_qx + 1e4 * (1 - contactL_);
    Qf(16,16) = lo_qy * lo_qy + 1e4 * (1 - contactL_);
    Qf(17,17) = lo_qz * lo_qz + 1e4 * (1 - contactL_);

    Qf(18,18) = rp_qx * rp_qx + 1e4 * (1 - contactR_);
    Qf(19,19) = rp_qy * rp_qy + 1e4 * (1 - contactR_);
    Qf(20,20) = rp_qz * rp_qz + 1e4 * (1 - contactR_);

    Qf(21,21) = ro_qx * ro_qx + 1e4 * (1 - contactR_);
    Qf(22,22) = ro_qy * ro_qy + 1e4 * (1 - contactR_);
    Qf(23,23) = ro_qz * ro_qz + 1e4 * (1 - contactR_);
   

    //Qff.noalias() =  Lcf * Qf * Lcf.transpose() * dt;
    Qff.noalias() = Af * Lcf * Qf * Lcf.transpose() * Af.transpose() * dt;

    /** Predict Step: Propagate the Error Covariance  **/
    P = Af * P * Af.transpose();
    P.noalias() += Qff;

    //Propagate only if non-zero input
    
    if (omegahat(0) != 0 || omegahat(1) != 0 || omegahat(2) != 0)
    {
        Rib *= expMap(omegahat * dt);
    }
    Ril = contactL_ * Ril + (1 - contactL_) * Rib * Rbl_;
    Rir = contactR_ * Rir + (1 - contactR_) * Rib * Rbr_;


    x.segment<3>(3) = Vector3d::Zero();
    x.segment<3>(18) = Vector3d::Zero();
    x.segment<3>(24) = Vector3d::Zero();

    updateVars();
}

/** Update **/
void ContactEKF::updateWithLegContacts(Vector3d yl, Quaterniond qyl, Vector3d yr, Quaterniond qyr,  Matrix3d JLQeJL, Matrix3d JRQeJR, double probL_, double probR_, int contactL_, int contactR_)
{
    std::cout<<"Contact Status is L/R "<<probL_<<" "<<probR_<<std::endl;
    std::cout<<"L/R Vector "<<std::endl;
    std::cout<<yl<<std::endl;
    std::cout<<yr<<std::endl;


    Rc.setZero();
    Rc(0, 0) = lp_px * lp_px + (1 - contactL_)*1e4 ;
    Rc(1, 1) = lp_py * lp_py + (1 - contactL_)*1e4 ;
    Rc(2, 2) = lp_pz * lp_pz + (1 - contactL_)*1e4 ;
    Rc(3, 3) = lo_px * lo_px + (1 - contactL_)*1e4 ;
    Rc(4, 4) = lo_py * lo_py + (1 - contactL_)*1e4 ;
    Rc(5, 5) = lo_pz * lo_pz + (1 - contactL_)*1e4 ;


    Rc.block<3,3>(0,0) += probL_ * JLQeJL;
    Rc.block<3,3>(3,3) += probL_ * JLQeJL;


    Rc(6, 6) = rp_px * rp_px+ (1 - contactR_)*1e4 ;
    Rc(7, 7) = rp_py * rp_py+ (1 - contactR_)*1e4 ;
    Rc(8, 8) = rp_pz * rp_pz+ (1 - contactR_)*1e4 ;
    Rc(9, 9) = ro_px * ro_px+ (1 - contactR_)*1e4 ;
    Rc(10, 10) = ro_py * ro_py + (1 - contactR_)*1e4 ;
    Rc(11, 11) = ro_pz * ro_pz + (1 - contactR_)*1e4 ;

    Rc.block<3,3>(6,6) += probR_ * JRQeJR;
    Rc.block<3,3>(9,9) += probR_ * JRQeJR;

    Vector3d dl =  Rib.transpose() * (x.segment<3>(15) - x.segment<3>(6));
    Vector3d dr =  Rib.transpose() * (x.segment<3>(21) - x.segment<3>(6));

    Matrix3d dRl = Ril.transpose()*Rib;
    Matrix3d dRr = Rir.transpose()*Rib;


    zf.segment<3>(0) = yl;
    zf.segment<3>(0) -=  dl;

    zf.segment<3>(3) = logMap(qyl.toRotationMatrix() * dRl);
    //zf.segment<3>(3) = logMap(dRl * qyl.toRotationMatrix());


    zf.segment<3>(6) = yr;
    zf.segment<3>(6) -=  dr;

    zf.segment<3>(9) = logMap( qyr.toRotationMatrix() * dRr);
    //zf.segment<3>(9) = logMap( dRr * qyr.toRotationMatrix());



    Hcf.block<3, 3>(0, 6) = -Rib.transpose();
    Hcf.block<3, 3>(0, 15) = Rib.transpose();
    Hcf.block<3, 3>(0, 3) = wedge(dl);

    Hcf.block<3, 3>(3, 3) = -dRl;
    Hcf.block<3, 3>(3, 18) = Matrix3d::Identity();
    
    Hcf.block<3, 3>(6, 6) = -Rib.transpose();
    Hcf.block<3, 3>(6, 21) = Rib.transpose();
    Hcf.block<3, 3>(6, 3) = wedge(dr);

    Hcf.block<3, 3>(9, 3) = -dRr;
    Hcf.block<3, 3>(9, 24) = Matrix3d::Identity();


    sc = Rc;
    sc.noalias() += Hcf * P * Hcf.transpose();
    Kcf.noalias() = P * Hcf.transpose() * sc.inverse();

    dxf.noalias() = Kcf * zf;

    //Update the mean estimate
    x.noalias() += dxf;

    //Update the error covariance
    P = (If - Kcf * Hcf) * P * (If - Hcf.transpose() * Kcf.transpose());
    P.noalias() += Kcf * Rc * Kcf.transpose();

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dxf.segment<3>(3));
    }
    if (dxf(18) != 0 || dxf(19) != 0 || dxf(20) != 0)
    {
        Ril *= expMap(dxf.segment<3>(18));
    }
    if (dxf(24) != 0 || dxf(25) != 0 || dxf(26) != 0)
    {
        Rir *= expMap(dxf.segment<3>(24));
    }

    x.segment<3>(3) = Vector3d::Zero();
    x.segment<3>(18) = Vector3d::Zero();
    x.segment<3>(24) = Vector3d::Zero();
    updateVars();
}





void ContactEKF::updateWithTwist(Vector3d y)
{

    Rv(0, 0) = vel_px * vel_px;
    Rv(1, 1) = vel_py * vel_py;
    Rv(2, 2) = vel_pz * vel_pz;

    v = x.segment<3>(0);
    //std::cout<<y<<std::endl;
    //Innovetion vector
    zv = y;
    zv.noalias() -= Rib * v;

    Hv.block<3, 3>(0, 0) = Rib;
    Hv.block<3, 3>(0, 3).noalias() = -Rib * wedge(v);
    sv = Rv;
    sv.noalias() += Hv * P * Hv.transpose();
    Kv.noalias() = P * Hv.transpose() * sv.inverse();

    dxf.noalias() = Kv * zv;

    //Update the mean estimate
    x.noalias() += dxf;

    //Update the error covariance
    P = (If - Kv * Hv) * P * (If - Hv.transpose() * Kv.transpose());
    P.noalias() += Kv * Rv * Kv.transpose();

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dxf.segment<3>(3));
    }
    if (dxf(18) != 0 || dxf(19) != 0 || dxf(20) != 0)
    {
        Ril *= expMap(dxf.segment<3>(18));
    }
    if (dxf(24) != 0 || dxf(25) != 0 || dxf(26) != 0)
    {
        Rir *= expMap(dxf.segment<3>(24));
    }

    x.segment<3>(3) = Vector3d::Zero();
    x.segment<3>(18) = Vector3d::Zero();
    x.segment<3>(24) = Vector3d::Zero();
    updateVars();
}

void ContactEKF::updateWithTwistRotation(Vector3d y, Quaterniond qy)
{

    R(0, 0) = vel_px * vel_px;
    R(1, 1) = vel_py * vel_py;
    R(2, 2) = vel_pz * vel_pz;
    R(3, 3) = leg_odom_ax * leg_odom_ax;
    R(4, 4) = leg_odom_ay * leg_odom_ay;
    R(5, 5) = leg_odom_az * leg_odom_az;

    v = x.segment<3>(0);

    //Innovetion vector
    z.segment<3>(0) = y;
    z.segment<3>(0).noalias() -= Rib * v;
    z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));

    Hvf.block<3, 3>(0, 0) = Rib;
    Hvf.block<3, 3>(0, 3).noalias() = -Rib * wedge(v);

    s = R;
    s.noalias() += Hvf * P * Hvf.transpose();
    Kf.noalias() = P * Hvf.transpose() * s.inverse();

    dxf.noalias() = Kf * z;

    //Update the mean estimate
    x.noalias() += dxf;

    //Update the error covariance
    P = (If - Kf * Hvf) * P * (If - Hvf.transpose() * Kf.transpose());
    P.noalias() += Kf * R * Kf.transpose();

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dxf.segment<3>(3));
    }
    if (dxf(18) != 0 || dxf(19) != 0 || dxf(20) != 0)
    {
        Ril *= expMap(dxf.segment<3>(18));
    }
    if (dxf(24) != 0 || dxf(25) != 0 || dxf(26) != 0)
    {
        Rir *= expMap(dxf.segment<3>(24));
    }

    x.segment<3>(3) = Vector3d::Zero();
    x.segment<3>(18) = Vector3d::Zero();
    x.segment<3>(24) = Vector3d::Zero();
    updateVars();
}

void ContactEKF::updateWithLegOdom(Vector3d y, Quaterniond qy)
{
    R(0, 0) = leg_odom_px * leg_odom_px;
    R(1, 1) = leg_odom_py * leg_odom_py;
    R(2, 2) = leg_odom_pz * leg_odom_pz;

    R(3, 3) = leg_odom_ax * leg_odom_ax;
    R(4, 4) = leg_odom_ay * leg_odom_ay;
    R(5, 5) = leg_odom_az * leg_odom_az;

    r = x.segment<3>(6);
    
    //Innovetion vector
    z.segment<3>(0) = y - r;
    z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));
    //z.segment<3>(3) = logMap(qy.toRotationMatrix().transpose() * Rib);

    //Compute the Kalman Gain
    s = R;
    s.noalias() += Hf * P * Hf.transpose();
    Kf.noalias() = P * Hf.transpose() * s.inverse();

    //Update the error covariance
    P = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
    P.noalias() += Kf * R * Kf.transpose();

    dxf.noalias() = Kf * z;
    x.noalias() += dxf;

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dxf.segment<3>(3));
    }
    if (dxf(18) != 0 || dxf(19) != 0 || dxf(20) != 0)
    {
        Ril *= expMap(dxf.segment<3>(18));
    }
    if (dxf(24) != 0 || dxf(25) != 0 || dxf(26) != 0)
    {
        Rir *= expMap(dxf.segment<3>(24));
    }

    x.segment<3>(3) = Vector3d::Zero();
    x.segment<3>(18) = Vector3d::Zero();
    x.segment<3>(24) = Vector3d::Zero();
    updateVars();
}

void ContactEKF::updateWithOdom(Vector3d y, Quaterniond qy)
{
    R(0, 0) = odom_px * odom_px;
    R(1, 1) = odom_py * odom_py;
    R(2, 2) = odom_pz * odom_pz;

    R(3, 3) = odom_ax * odom_ax;
    R(4, 4) = odom_ay * odom_ay;
    R(5, 5) = odom_az * odom_az;

    r = x.segment<3>(6);
    //Innovetion vector
    z.segment<3>(0) = y - r;
    z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));
    //z.segment<3>(3) = logMap(qy.toRotationMatrix() * Rib.transpose());
    //z.segment<3>(3) = logMap(qy.toRotationMatrix().transpose() * Rib);
    //Compute the Kalman Gain
    s = R;
    s.noalias() += Hf * P * Hf.transpose();
    Kf.noalias() = P * Hf.transpose() * s.inverse();

    //Update the error covariance
    P = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
    P.noalias() += Kf * R * Kf.transpose();

    dxf.noalias() = Kf * z;
    //Update the mean estimate
    x += dxf;

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dxf.segment<3>(3));
    }
    if (dxf(18) != 0 || dxf(19) != 0 || dxf(20) != 0)
    {
        Ril *= expMap(dxf.segment<3>(18));
    }
    if (dxf(24) != 0 || dxf(25) != 0 || dxf(26) != 0)
    {
        Rir *= expMap(dxf.segment<3>(24));
    }

    x.segment<3>(3) = Vector3d::Zero();
    x.segment<3>(18) = Vector3d::Zero();
    x.segment<3>(24) = Vector3d::Zero();
    updateVars();
}

void ContactEKF::updateVars()
{

    pos = x.segment<3>(6);
    rX = pos(0);
    rY = pos(1);
    rZ = pos(2);

    Tib.linear() = Rib;
    Tib.translation() = pos;
    qib = Quaterniond(Tib.linear());



    Til.linear() = Ril;
    Til.translation() = x.segment<3>(15);
    qil = Quaterniond(Til.linear());

    Tir.linear() = Rir;
    Tir.translation() = x.segment<3>(21);
    qir = Quaterniond(Tir.linear());  


    //Update the biases
    bgyr = x.segment<3>(9);
    bacc = x.segment<3>(12);
    //std::cout<<"bacc"<<std::endl;
    //std::cout<<bacc<<std::endl;
    //std::cout<<"bgyr"<<std::endl;
    //std::cout<<bgyr<<std::endl;
    bias_gx = x(9);
    bias_gy = x(10);
    bias_gz = x(11);
    bias_ax = x(12);
    bias_ay = x(13);
    bias_az = x(14);

    omegahat = omega - bgyr;
    fhat = f - bacc;

    gyro = Rib * omegahat;
    gyroX = gyro(0);
    gyroY = gyro(1);
    gyroZ = gyro(2);

    acc = Rib * fhat;
    accX = acc(0);
    accY = acc(1);
    accZ = acc(2);

    v = x.segment<3>(0);
    vel = Rib * v;
    velX = vel(0);
    velY = vel(1);
    velZ = vel(2);

    //ROLL - PITCH - YAW
    angle = getEulerAngles(Rib);
    angleX = angle(0);
    angleY = angle(1);
    angleZ = angle(2);
}
