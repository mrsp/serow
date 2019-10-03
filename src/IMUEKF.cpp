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
    g(2) = -9.80;
    mahalanobis_TH = -1;
}

void IMUEKF::init() {
    
    firstrun = true;
    useEuler = true;
    If = Matrix<double, 15, 15>::Identity();
    P = Matrix<double, 15, 15>::Zero();
    //vel
    P(0,0) = 1e-3;
    P(1,1) = 1e-3;
    P(2,2) = 1e-3;
    //Rot error
    P(3,3) = 1e-3;
    P(4,4) = 1e-3;
    P(5,5) = 1e-3;
    //Pos
    P(6,6)  = 1e-5;
    P(7,7)  = 1e-5;
    P(8,8)  = 1e-5;
    //Biases
    P(9, 9) = 1e-3;
    P(10, 10) = 1e-3;
    P(11, 11) = 1e-3;
    P(12, 12) = 1e-3;
    P(13, 13) = 1e-3;
    P(14, 14) = 1e-3;
    
    //For outlier detection
    f0 = 0.1;
    e0 = 0.9;
    //Construct C
    Hf = Matrix<double, 6, 15>::Zero();
    Hf.block<3,3>(0,6) = Matrix3d::Identity();
    Hf.block<3,3>(3,3) = Matrix3d::Identity();
    Hvf = Matrix<double, 6, 15>::Zero();
    Hvf.block<3,3>(3,3) = Matrix3d::Identity();
    Hv = Matrix<double, 3, 15>::Zero();
    
    /*Initialize the state **/
    //Rotation Matrix from Inertial to body frame
    Rib = Matrix3d::Identity();
    x = Matrix<double,15,1>::Zero();
    
    //Innovation Vector
    z = Matrix<double, 6, 1>::Zero(); //For Odometry
    zv = Vector3d::Zero(); //For Twist
    
    //Initializing vectors and matrices
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
    Qff = Matrix<double, 15, 15>::Zero();
    Qf = Matrix<double, 12, 12>::Zero();
    Af = Matrix<double, 15, 15>::Zero();
    
    bgyr = Vector3d::Zero();
    bacc = Vector3d::Zero();
    gyro = Vector3d::Zero();
    acc = Vector3d::Zero();
    angle = Vector3d::Zero();
    
    
    //bias removed acceleration and gyro rate
    fhat = Vector3d::Zero();
    omegahat = Vector3d::Zero();
    Tib = Affine3d::Identity();
    
    
    
    //Compute some parts of the Input-Noise Jacobian once since they are constants
    //gyro (0),acc (3),gyro_bias (6),acc_bias (9)
    Lcf = Matrix<double, 15, 12>::Zero();
    Lcf.block<3,3>(0,3) = -Matrix3d::Identity();
    Lcf.block<3,3>(3,0) = -Matrix3d::Identity();
    Lcf.block<3,3>(9,6) = Matrix3d::Identity();
    Lcf.block<3,3>(12,9) = Matrix3d::Identity();
    
    
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


Matrix<double,15,15> IMUEKF::computeTrans(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_)
{
    omega_.noalias() -= x_.segment<3>(9);
    f_.noalias() -= x_.segment<3>(12);
    v = x_.segment<3>(0);
    Matrix<double,15,15> res = Matrix<double,15,15>::Zero();
    res.block<3,3>(0,0).noalias()  = -wedge(omega_);
    res.block<3,3>(0,3).noalias()  = wedge(Rib_.transpose() * g);
    res.block<3,3>(0,12).noalias() = -Matrix3d::Identity();
    res.block<3,3>(0,9).noalias()  = -wedge(v);
    res.block<3,3>(3,3).noalias() = -wedge(omega_);
    res.block<3,3>(3,9).noalias() = -Matrix3d::Identity();
    res.block<3,3>(6,0) = Rib_;
    res.block<3,3>(6,3).noalias() = -Rib_ * wedge(v);
    return res;
}


void IMUEKF::euler(Vector3d omega_, Vector3d f_)
{
    Acf=computeTrans(x,Rib,omega_,f_);
    //Euler Discretization - First order Truncation
    Af = If;
    Af.noalias() += Acf * dt;
    x = computeDiscreteDyn(x,Rib,omega_,f_);
    //x.noalias() += computeContinuousDyn(x,Rib,omega_,f_)*dt; 
}


Matrix<double,15,1> IMUEKF::computeContinuousDyn(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_)
{
    Matrix<double,15,1> res = Matrix<double,15,1>::Zero();
    
    //Inputs without bias
    omega_ -= x_.segment<3>(9);
    f_ -= x_.segment<3>(12);
    
    //Nonlinear Process Model
    v = x_.segment<3>(0);
    res.segment<3>(0).noalias() = v.cross(omega_);
    res.segment<3>(0).noalias() += Rib_.transpose()*g;
    res.segment<3>(0) += f_;
    res.segment<3>(6).noalias() = Rib_ * v;
    
    return res;
}

Matrix<double,15,1> IMUEKF::computeDiscreteDyn(Matrix<double,15,1> x_, Matrix<double,3,3> Rib_, Vector3d omega_, Vector3d f_)
{

    Matrix<double,15,1> res = Matrix<double,15,1>::Zero();

    omega_.noalias() -= x_.segment<3>(9);
    f_.noalias() -= x_.segment<3>(12);

    //Nonlinear Process Model
    //Compute \dot{v}_b @ k
    v = x_.segment<3>(0);
    res.segment<3>(0).noalias() = v.cross(omega_);
    res.segment<3>(0).noalias() += Rib_.transpose()*g;
    res.segment<3>(0) += f_;

    //Position
    r = x_.segment<3>(6);
    res.segment<3>(6).noalias() = Rib_*res.segment<3>(0)*dt*dt/2.00;
    res.segment<3>(6).noalias() += Rib_*v*dt;
    res.segment<3>(6) += r;


    //Velocity
    res.segment<3>(0) *= dt;
    res.segment<3>(0) += v;

    //Biases
    res.segment<3>(9) =  x_.segment<3>(9);
    res.segment<3>(12) =  x_.segment<3>(12);
    return res;
}


/** IMU EKF filter to  deal with the Noise **/
void IMUEKF::predict(Vector3d omega_, Vector3d f_)
{
    omega = omega_;
    f = f_;
    //Used in updating Rib with the Rodriquez formula
    omegahat.noalias() = omega - x.segment<3>(9);
    v = x.segment<3>(0);
    
    //Update the Input-noise Jacobian
    Lcf.block<3,3>(0,0).noalias() = -wedge(v);
    
    
    euler(omega_,f_);

    
    
    
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
    
    
    //Qff.noalias() =  Lcf * Qf * Lcf.transpose() * dt;
    Qff.noalias() =  Af * Lcf * Qf * Lcf.transpose() * Af.transpose() * dt ;
    
    /** Predict Step: Propagate the Error Covariance  **/
    P = Af * P * Af.transpose();
    P.noalias() += Qff;
    
    //Propagate only if non-zero input
    
    if (omegahat(0) != 0 && omegahat(1) != 0 && omegahat(2) != 0)
    {
        Rib  *=  expMap(omegahat*dt);
    }
    
    x.segment<3>(3) = Vector3d::Zero();
    updateVars();
}


/** Update **/
void IMUEKF::updateWithTwist(Vector3d y)
{

    Rv(0, 0) = vel_px * vel_px;
    Rv(1, 1) = vel_py * vel_py;
    Rv(2, 2) = vel_pz * vel_pz;

    v = x.segment<3>(0);
    //std::cout<<y<<std::endl;
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
    x.noalias() += dxf;
    
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


/** Update **/
void IMUEKF::updateWithTwistRotation(Vector3d y,Quaterniond qy)
{

    R(0, 0) = vel_px * vel_px;
    R(1, 1) = vel_py * vel_py;
    R(2, 2) = vel_pz * vel_pz;
    R(3, 3) =  leg_odom_ax * leg_odom_ax;
    R(4, 4) =  leg_odom_ay * leg_odom_ay;
    R(5, 5) =  leg_odom_az * leg_odom_az;



    v = x.segment<3>(0);
    //std::cout<<" Update with Twist Rot" <<std::endl;
    //std::cout<<y<<std::endl;
    //Innovetion vector
    z.segment<3>(0) = y;
    z.segment<3>(0).noalias() -= Rib * v;
    z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));


    Hvf.block<3,3>(0,0) = Rib;
    Hvf.block<3,3>(0,3).noalias() = -Rib * wedge(v);
    s = R;
    s.noalias() += Hvf * P * Hvf.transpose();
    Kf.noalias() = P * Hvf.transpose() * s.inverse();
    
    dxf.noalias() = Kf * z;
   
    //Update the mean estimate
    x.noalias() += dxf;
    
    //Update the error covariance
    P = (If - Kf * Hvf) * P * (If - Hvf.transpose()*Kf.transpose());
    P.noalias()+= Kf * R * Kf.transpose();
    
    
    if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
    {
        Rib *=  expMap(dxf.segment<3>(3));
    }
    x.segment<3>(3) = Vector3d::Zero();
    
    updateVars();
    
    
}

void IMUEKF::updateWithLegOdom(Vector3d y, Quaterniond qy)
{
    R(0, 0) = leg_odom_px * leg_odom_px;
    R(1, 1) = leg_odom_py * leg_odom_py;
    R(2, 2) = leg_odom_pz * leg_odom_pz;
    
    R(3, 3) =  leg_odom_ax * leg_odom_ax;
    R(4, 4) =  leg_odom_ay * leg_odom_ay;
    R(5, 5) =  leg_odom_az * leg_odom_az;


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
        if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
        {
             Rib *=  expMap(dxf.segment<3>(3));
        }
        x.segment<3>(3) = Vector3d::Zero();


    updateVars();


}

bool IMUEKF::updateWithOdom(Vector3d y, Quaterniond qy, bool useOutlierDetection)
{
    R(0, 0) = odom_px * odom_px;
    R(1, 1) = odom_py * odom_py;
    R(2, 2) = odom_pz * odom_pz;
    
    R(3, 3) =  odom_ax * odom_ax;
    R(4, 4) =  odom_ay * odom_ay;
    R(5, 5) =  odom_az * odom_az;

    outlier = false;
    if(!useOutlierDetection)
    {
        if(mahalanobis_TH <= 0)
        {
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
            if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
            {
                 Rib *=  expMap(dxf.segment<3>(3));
            }
            x.segment<3>(3) = Vector3d::Zero();
        }
        else
        {
            r = x.segment<3>(6);
            //Innovetion vector
            z.segment<3>(0) = y - r;
            z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));
            //z.segment<3>(3) = logMap(qy.toRotationMatrix() * Rib.transpose());
            
            //Compute the Kalman Gain
            s = R;
            s.noalias() += Hf * P * Hf.transpose();
            Kf.noalias() = P * Hf.transpose() * s.inverse();
            
            //Update the error covariance
            P_i = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
            P_i.noalias() += Kf * R * Kf.transpose();
            dxf.noalias() = Kf * z;
            //Update the mean estimate
            x_i  = x + dxf;
            
            if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
            {
                Rib_i =  Rib*expMap(dxf.segment<3>(3));
            }
            s =  Hf * P_i * Hf.transpose() + R;
            temp = y-x_i.segment<3>(6);
	    double var_TH = temp.transpose()*s.block<3,3>(0,0).inverse()*temp;
	 
            if(var_TH<mahalanobis_TH)
            {
            
                P = P_i;
                x = x_i;
                Rib = Rib_i;
                x.segment<3>(3) = Vector3d::Zero();

            }
	    else{
		outlier = true;
	    }
        }
    }
    else
    {
        tau = 1.0;
        zeta = 1.0;
        e_t = e0;
        f_t = f0;
        P_i = P;
        x_i = x;
        x_i_ = x;
        Rib_i = Rib;
        //Innovetion vector
        r = x.segment<3>(6);
        z.segment<3>(0) = y -r;
        z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));
        //z.segment<3>(3) = logMap(qy.toRotationMatrix().transpose() * Rib);

        unsigned int j=0;
        while(j<4)
        {
            if(zeta>1.0e-5)
            {
                //Compute the Kalman Gain
	        R_z = R/zeta;
                s = R_z;
                s.noalias() += Hf * P * Hf.transpose();
                Kf.noalias() = P * Hf.transpose() * s.inverse();
                
                //Update the error covariance
                P_i = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
                P_i.noalias() += Kf * R_z * Kf.transpose();
                dxf.noalias() = Kf * z;
                
                //Update the mean estimate
                x_i_ = x_i;
                x_i  = x + dxf;
                
                if (dxf(3) != 0 && dxf(4) != 0 && dxf(5) != 0)
                {
                    Rib_i =  Rib *expMap(dxf.segment<3>(3));
                }
                x_i.segment<3>(3) = Vector3d::Zero();
                
                //outlier detection with the position measurement vector
		tempM = y*y.transpose();
		tempM.noalias() -=  2.0*y*x_i.segment<3>(6).transpose();
		tempM.noalias() += x_i.segment<3>(6)*x_i.segment<3>(6).transpose();
        	tempM.noalias() += P_i.block<3,3>(6,6);
                updateOutlierDetectionParams(tempM);
            }
            else
            {
                x_i = x;
                x_i_ = x;
                Rib_i = Rib;
                P_i = P;
                outlier = true;
                break;
            }
            
	    j++;

        }
        
        Rib = Rib_i;
        x = x_i;
        P = P_i;
        
    }
    updateVars();
    return outlier;
}


//Update the outlier indicator Zeta
void IMUEKF::updateOutlierDetectionParams(Eigen::Matrix<double, 3,3> BetaT)
{
    efpsi = computePsi(e_t+f_t);
    lnp = computePsi(e_t) - efpsi;
    ln1_p = computePsi(f_t) -efpsi;

    tempM = BetaT*(R.block<3,3>(0,0)).inverse();
  
    pzeta_1 =  exp(lnp - 0.5*(tempM).trace());
    pzeta_0 =  exp(ln1_p);
    
    //Normalization factor
    norm_factor = 1.0/(pzeta_1+pzeta_0);
    
    //p(zeta) are now proper probabilities
    pzeta_1 = norm_factor * pzeta_1;
    pzeta_0 = norm_factor * pzeta_0;
  
    //mean of bernulli
    zeta = pzeta_1 / (pzeta_1 + pzeta_0);
    
    //Update epsilon and f
    e_t = e0 + zeta;
    f_t = f0 + 1.0 - zeta;
}

//Digamma Function Approximation
double IMUEKF::computePsi(double xxx)
{
    
    double result = 0, xx, xx2, xx4;
    for ( ; xxx < 7; ++xxx)
    result -= 1/xxx;
    
    xxx -= 1.0/2.0;
    xx = 1.0/xxx;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += log(xxx)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
}

void IMUEKF::updateVars()
{
    
    pos = x.segment<3>(6);
    rX = pos(0);
    rY = pos(1);
    rZ = pos(2);
    Tib.linear() = Rib;
    Tib.translation() = pos;
    qib = Quaterniond(Tib.linear());
    
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

    gyro  = Rib * omegahat;
    gyroX = gyro(0);
    gyroY = gyro(1);
    gyroZ = gyro(2);
    
    acc =  Rib * fhat;
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
