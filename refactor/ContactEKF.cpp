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

#include "ContactEKF.h"

ContactEKF::ContactEKF()
{
    //Gravity Vector
    g = Vector3d::Zero();
    g(2) = -9.80;
}

void ContactEKF::init(State state)
{
    const int num_leg_end_effectors = state.getFootFrames().size();
    const int contact_dim = state.flat_feet ? 6 : 3;
    const int num_states = 15 + contact_dim * num_leg_end_effectors;
    const int num_inputs = 12 + contact_dim * num_leg_end_effectors;
    If.setIdentity(num_states, num_states);
    P = If;

    // Initialize state indices 
    v_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    r_idx_ = v_idx_ + 3;
    p_idx_ = r_idx_ + 3;
    bg_idx_ = p_idx_ + 3;
    ba_idx_ = bg_idx_ + 3;

    Eigen::ArrayXi pl_idx = Eigen::ArrayXi::LinSpaced(contact_dim, ba_idx_(2) + ba_idx_(2) + contact_dim);
    for (const auto& foot_frame : state.getFootFrames()) {
        pl_idx_.insert{foot_frame, pl_idx};
        pl_idx += contact_dim;
    }

    ng_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    na_idx_ = ng_idx_ + 3;
    nbg_idx_ = na_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;
    Eigen::ArrayXi npl_idx = Eigen::ArrayXi::LinSpaced(contact_dim, nba_idx_(2) + nba_idx_(2) + contact_dim);
    for (const auto &foot_frame : state.getFootFrames())
    {
        npl_idx_.insert{foot_frame, npl_idx};
        npl_idx += contact_dim;
    }

    // Initialize the error state covariance
    P(v_idx_, v_idx_) = state.getBaseLinearVelocityCov();
    P(r_idx_, r_idx_) = state.getBaseOrientationCov();
    P(p_idx_, p_idx_) = state.getBasePositionCov();
    P(bg_idx_, bg_idx_) = state.getImuAngularVelocityBiasCov();
    P(ba_idx_, ba_idx_) = state.getImuLinearAccelerationBiasCov();

    for (const auto& foot_frame : state.getFootFrames()) {
        P(pl_idx_.at(foot_frame), pl_idx_.at(foot_frame)) = state.getFootPoseCov(foot_frame);
    }

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), gyro_bias (6), acc_bias (9), leg end effectors (12 - 12 + contact_dim * N)
    Lcf.setZero(num_states, num_inputs);
    Lcf(v_idx_, na_idx_) = -Matrix3d::Identity();
    Lcf(r_idx_, ng_idx_) = -Matrix3d::Identity();
    Lcf(bg_idx_, nbg_idx_) = Matrix3d::Identity();
    Lcf(ba_idx_, nba_idx_) = Matrix3d::Identity();

    for (const auto& foot_frame : state.getFootFrames()) {
        Lcf(pl_idx_.at(foot_frame), npl_idx_.at(foot_frame)) = MatrixXd::Identity(contact_dim, contact_dim);
    }

    // Initialize the state vector
    state_ = std::move(state);

    std::cout << "Contact EKF Initialized Successfully" << std::endl;
}

MatrixXd ContactEKF::computeTrans(State state, Vector3d angular_velocity, Vector3d linear_acceleration)
{
    angular_velocity -= state.getImuAngularVelocityBias();
    const Eigen::Vector3d& v = state.getBaseLinearVelocity();
    const Eigen::Matrix3d& R = state.getBaseOrientation();
    const int num_leg_end_effectors = state.getFootFrames().size();
    const int contact_dim = state.flat_feet ? 6 : 3;
    const int num_states = 15 + contact_dim * num_leg_end_effectors;

    Eigen::MatrixXd res;
    res.setZero(num_states, num_states);
    res.block<3, 3>(0, 0).noalias() = -wedge(angular_velocity);
    res.block<3, 3>(0, 3).noalias() = wedge(R.transpose() * g);
    res.block<3, 3>(0, 12).noalias() = -Eigen::Matrix3d::Identity();
    res.block<3, 3>(0, 9).noalias() = -wedge(v);
    res.block<3, 3>(3, 3).noalias() = -wedge(angular_velocity);
    res.block<3, 3>(3, 9).noalias() = -Eigen::Matrix3d::Identity();
    res.block<3, 3>(6, 0) = R;
    res.block<3, 3>(6, 3).noalias() = -R * wedge(v);
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


Matrix<double, 27, 1> ContactEKF::computeDiscreteDyn(Matrix<double, 27, 1> x_, Matrix<double, 3, 3> Rib_, Vector3d omega_, Vector3d f_, Vector3d pbl_, Vector3d pbr_, int contactL_, int contactR_)
{
    State predicted_state;
    angular_velocity -= state.getImuAngularVelocityBias();
    linear_acceleration -= state.getImuLinearAccelarationBias();

    // Nonlinear Process Model
    // Compute \dot{v}_b @ k
    const Eigen::Vector3d& v = state.getBaseLinearVelocity();
    const Eigen::Matrix3d& R = state.getBaseOrientation();

    predicted_state.base_linear_velocity_.noalias() = v.cross(angular_velocity);
    predicted_state.base_linear_velocity_ += R.transpose() * g;
    predicted_state.base_linear_velocity_ += linear_acceleration;

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
