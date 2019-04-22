#include <serow/IMUinEKF.h>

using namespace std;

IMUinEKF::IMUinEKF()
{
    //Gravity Vector
    g = Vector3d::Zero();
    g(2) = -9.80;
    P = Matrix<double, 21, 21>::Zero();
    X = Matrix<double, 7, 7>::Identity();
    theta = Matrix<double, 6, 1>::Zero();
}

void IMUinEKF::init()
{

    firstrun = true;
    If = Matrix<double, 21, 21>::Identity();
    I = Matrix3d::Identity();
    w_ = Vector3d::Zero();
    a_ = Vector3d::Zero();
    a = Vector3d::Zero();
    w = Vector3d::Zero();

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
    P(0, 0) = 1e-1;
    P(1, 1) = 1e-1;
    P(2, 2) = 1e-1;
    //Vel error
    P(3, 3) = 1e-1;
    P(4, 4) = 1e-1;
    P(5, 5) = 1e-1;
    //Pos error
    P(6, 6) = 1e-2;
    P(7, 7) = 1e-2;
    P(8, 8) = 1e-2;
    //dR error
    P(9, 9) = 1e-2;
    P(10, 10) = 1e-2;
    P(11, 11) = 1e-2;
    //dL error
    P(12, 12) = 1e-2;
    P(13, 13) = 1e-2;
    P(14, 14) = 1e-2;
    //Biases
    P(15, 15) = 1e-1;
    P(16, 16) = 1e-1;
    P(17, 17) = 1e-1;
    P(18, 19) = 1e-1;
    P(19, 19) = 1e-1;
    P(20, 20) = 1e-1;

    Af = Matrix<double, 21, 21>::Zero();
    Af.block<3, 3>(3, 0).noalias() = skew(g);
    Af.block<3, 3>(6, 3).noalias() = Matrix3d::Identity();

    Qff = Matrix<double, 21, 21>::Zero();
    Qf = Matrix<double, 21, 21>::Zero();
    Qc = Matrix3d::Zero();
    Adj = Matrix<double, 21, 21>::Zero();
    Phi = Matrix<double, 21, 21>::Zero();
    Rwb = Matrix3d::Identity();
    Rib = Rwb;
    pwb = Vector3d::Zero();
    vwb = Vector3d::Zero();
    dL = Vector3d::Zero();
    dR = Vector3d::Zero();
    bgyr = Vector3d::Zero();
    bacc = Vector3d::Zero();
    gyro = Vector3d::Zero();
    acc = Vector3d::Zero();
    angle = Vector3d::Zero();
    Tib = Affine3d::Identity();

    std::cout << "IMU Right-Invariant EKF Initialized Successfully" << std::endl;
}

void IMUinEKF::constructState(Matrix<double, 7, 7> &X_, Matrix<double, 6, 1> &theta_, Matrix3d R_, Vector3d v_, Vector3d p_, Vector3d dR_, Vector3d dL_, Vector3d bg_, Vector3d ba_)
{
    X_.block<3, 3>(0, 0) = R_;
    X_.block<3, 1>(0, 3) = v_;
    X_.block<3, 1>(0, 4) = p_;
    X_.block<3, 1>(0, 5) = dR_;
    X_.block<3, 1>(0, 6) = dL_;

    theta_.segment<3>(0) = bg_;
    theta_.segment<3>(3) = ba_;
}

void IMUinEKF::seperateState(Matrix<double, 7, 7> X_, Matrix<double, 6, 1> theta_, Matrix3d &R_, Vector3d &v_, Vector3d &p_, Vector3d &dR_, Vector3d &dL_, Vector3d &bg_, Vector3d &ba_)
{
    R_ = X_.block<3, 3>(0, 0);
    v_ = X_.block<3, 1>(0, 3);
    p_ = X_.block<3, 1>(0, 4);
    dR_ = X_.block<3, 1>(0, 5);
    dL_ = X_.block<3, 1>(0, 6);

    bg_ = theta_.segment<3>(0);
    ba_ = theta_.segment<3>(3);
}

Matrix<double, 7, 7> IMUinEKF::exp_SE3(Matrix<double, 15, 1> v)
{
    Matrix<double, 7, 7> dX_ = Matrix<double, 7, 7>::Identity();
    double phi = v.segment<3>(0).norm();
    Matrix3d R_ = Matrix3d::Identity();
    Matrix3d Jr_ = Matrix3d::Identity();
    if (phi >= 1e-6)
    {
        Matrix3d A = skew(v.segment<3>(0));
        Matrix3d A2 = A * A;
        double phi2 = phi * phi;
        R_.noalias() += sin(phi) / phi * A;
        R_.noalias() += (1 - cos(phi)) / (phi2)*A2;

        Jr_.noalias() += (1 - cos(phi)) / (phi2)*A;
        Jr_.noalias() += (phi - sin(phi)) / (phi2 * phi) * A2;
    }
    dX_.block<3, 3>(0, 0) = R_;
    dX_.block<3, 1>(0, 3).noalias() = Jr_ * v.segment<3>(3);
    dX_.block<3, 1>(0, 4).noalias() = Jr_ * v.segment<3>(6);
    dX_.block<3, 1>(0, 5).noalias() = Jr_ * v.segment<3>(9);
    dX_.block<3, 1>(0, 6).noalias() = Jr_ * v.segment<3>(12);

    return dX_;
}

Matrix3d IMUinEKF::exp_SO3(Vector3d v)
{
    Matrix3d R_ = Matrix3d::Identity();
    Matrix3d A = skew(v);
    double phi = v.norm();
    if (phi >= 1e-6)
    {
        R_.noalias() += (sin(phi) / phi) * A;
        R_.noalias() += (1 - cos(phi)) / (phi * phi) * A * A;
    }

    return R_;
}

Matrix<double, 21, 21> IMUinEKF::Adjoint(Matrix<double, 7, 7> X_)
{
    Matrix<double, 21, 21> AdjX = Matrix<double, 21, 21>::Identity();
    Matrix3d R_;
    Vector3d v_, p_, dR_, dL_, bg_, ba_;
    seperateState(X_, Matrix<double, 6, 1>::Zero(), R_, v_, p_, dR_, dL_, bg_, ba_);

    AdjX.block<3, 3>(0, 0) = R_;
    AdjX.block<3, 3>(3, 3) = R_;
    AdjX.block<3, 3>(6, 6) = R_;
    AdjX.block<3, 3>(9, 9) = R_;
    AdjX.block<3, 3>(12, 12) = R_;

    AdjX.block<3, 3>(3, 0).noalias() = skew(v_) * R_;
    AdjX.block<3, 3>(6, 0).noalias() = skew(p_) * R_;
    AdjX.block<3, 3>(9, 0).noalias() = skew(dR_) * R_;
    AdjX.block<3, 3>(12, 0).noalias() = skew(dL_) * R_;

    return AdjX;
}

void IMUinEKF::predict(Vector3d angular_velocity, Vector3d linear_acceleration, Vector3d pbr, Vector3d pbl, Matrix3d hR_R, Matrix3d hR_L, int contactR, int contactL)
{

    seperateState(X, theta, Rwb, vwb, pwb, dR, dL, bgyr, bacc);
    w_ = angular_velocity;
    a_ = linear_acceleration;

    //Bias removed gyro and acc
    w = w_ - bgyr;
    a = a_ - bacc;

    Af.block<3, 3>(0, 15).noalias() = -Rwb;
    Af.block<3, 3>(3, 18).noalias() = -Rwb;
    Af.block<3, 3>(3, 15).noalias() = -skew(vwb) * Rwb;
    Af.block<3, 3>(6, 15).noalias() = -skew(pwb) * Rwb;
    Af.block<3, 3>(9, 15).noalias() = -skew(dR) * Rwb;
    Af.block<3, 3>(12, 15).noalias() = -skew(dL) * Rwb;

    Phi = If + Af * dt;
    Adj = Adjoint(X);

    // Covariance Q with full state + biases
    Qf(0, 0) = gyr_qx * gyr_qx ;
    Qf(1, 1) = gyr_qy * gyr_qy ;
    Qf(2, 2) = gyr_qz * gyr_qz ;
    Qf(3, 3) = acc_qx * acc_qx ;
    Qf(4, 4) = acc_qy * acc_qy ;
    Qf(5, 5) = acc_qz * acc_qz ;

    Qc(0, 0) = foot_contactx * foot_contacty;
    Qc(1, 1) = foot_contacty * foot_contacty;
    Qc(2, 2) = foot_contactz * foot_contactz;

    Qf.block<3, 3>(9, 9) = hR_R * (Qc + 1e4 * I * (1 - contactR)) * hR_R.transpose();
    Qf.block<3, 3>(12, 12) = hR_L * (Qc + 1e4 * I * (1 - contactL)) * hR_L.transpose();

    Qf(15, 15) = gyrb_qx * gyrb_qx;
    Qf(16, 16) = gyrb_qy * gyrb_qy;
    Qf(17, 17) = gyrb_qz * gyrb_qz;
    Qf(18, 18) = accb_qx * accb_qx;
    Qf(19, 19) = accb_qy * accb_qy;
    Qf(20, 20) = accb_qz * accb_qz;

    Qff.noalias() = Phi * Adj * Qf * Adj.transpose() * Phi.transpose() * dt;

    /** Predict Step: Propagate the Error Covariance  **/
    P = Phi * P * Phi.transpose();
    P.noalias() += Qff;

    pwb += vwb * dt + 0.5 * (Rwb * a + g) * dt * dt;
    vwb += (Rwb * a + g) * dt;
    Rwb *= exp_SO3(w * dt);
    //Foot Position Dynamics
    dR = contactR * dR + (1 - contactR) * (pwb + Rwb * pbr);
    dL = contactL * dL + (1 - contactL) * (pwb + Rwb * pbl);
    constructState(X, theta, Rwb, vwb, pwb, dR, dL, bgyr, bacc);
    updateVars();
}

void IMUinEKF::updateWithTwist(Vector3d vy, Matrix3d Rvy)
{
    Matrix<double, 7, 1> Y = Matrix<double, 7, 1>::Zero();
    Y.segment<3>(0) = vy;
    Y(3) = 1.00;

    Matrix<double, 7, 1> b = Matrix<double, 7, 1>::Zero();
    b(3) = 1.00;
    Matrix<double, 3, 21> H = Matrix<double, 3, 21>::Zero();
    H.block<3, 3>(0, 3) = Matrix3d::Identity();

    Matrix<double, 3, 3> N = Matrix<double, 3, 3>::Zero();
    N = Rvy;
    N(0, 0) += vel_px * vel_px;
    N(1, 1) += vel_py * vel_py;
    N(2, 2) += vel_pz * vel_pz;

    Matrix<double, 3, 7> PI = Matrix<double, 3, 7>::Zero();
    PI.block<3, 3>(0, 0) = Matrix3d::Identity();
    updateVelocity(Y, b, H, N, PI);
    updateVars();
}
void IMUinEKF::updateVelocity(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix3d N_, Matrix<double, 3, 7> PI_)
{
    //Transform P to Left Invariant in order to perform the update
    Adj = Adjoint(X);
    P = Adj.inverse() * P * (Adj.inverse()).transpose();

    Matrix3d S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 21, 3> K_ = P * H_.transpose() * S_.inverse();
    Matrix<double, 7, 1> Z_ = X.inverse() * Y_ - b_;

    //Update State
    Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
    Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

    X = X * dX_;
    theta += dtheta_;

    Matrix<double, 21, 21> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
    //Transform P to Right Invariant
    P = Adj * P * Adj.transpose();
}

void IMUinEKF::updateWithOdom(Vector3d py, Quaterniond qy)
{
    Matrix<double, 14, 1> Y = Matrix<double, 14, 1>::Zero();
    Y.segment<3>(0) = logMap(X.block<3, 3>(0, 0).inverse() * qy.toRotationMatrix());
    Y.segment<3>(7) = py;
    Y(11) = 1.000;

    Matrix<double, 14, 1> b = Matrix<double, 14, 1>::Zero();
    b(11) = 1.000;
    Matrix<double, 6, 21> H = Matrix<double, 6, 21>::Zero();
    H.block<3, 3>(0, 0) = Matrix3d::Identity();
    H.block<3, 3>(3, 6) = Matrix3d::Identity();

    Matrix<double, 6, 6> N = Matrix<double, 6, 6>::Zero();
    N(0, 0) = odom_ax * odom_ax;
    N(1, 1) = odom_ay * odom_ay;
    N(2, 2) = odom_az * odom_az;
    N(3, 3) = odom_px * odom_px;
    N(4, 4) = odom_py * odom_py;
    N(5, 5) = odom_pz * odom_pz;

    Matrix<double, 6, 14> PI = Matrix<double, 6, 14>::Zero();
    PI.block<3, 3>(0, 0) = Matrix3d::Identity();
    PI.block<3, 3>(3, 7) = Matrix3d::Identity();

    updatePositionOrientation(Y, b, H, N, PI);
    updateVars();
}

void IMUinEKF::updatePositionOrientation(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_)
{
    //Transform P to Left Invariant in order to perform the update
    Adj = Adjoint(X);
    P = Adj.inverse() * P * (Adj.inverse()).transpose();

    Matrix<double, 6, 6> S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 21, 6> K_ = P * H_.transpose() * S_.inverse();

    Matrix<double, 14, 14> BigX = Matrix<double, 14, 14>::Zero();

    BigX.block<7, 7>(0, 0) = X;
    BigX.block<7, 7>(7, 7) = X;

    Matrix<double, 14, 1> Z_ = BigX.inverse() * Y_ - b_;
    Z_.segment<3>(0) = Y_.segment<3>(0);

    //Update State
    Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
    Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

    X = X * dX_;
    theta += dtheta_;

    Matrix<double, 21, 21> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
    //Transform P to Right Invariant
    P = Adj * P * Adj.transpose();
}

void IMUinEKF::updateWithOrient(Quaterniond qy)
{
    Matrix<double, 7, 1> Y = Matrix<double, 7, 1>::Zero();

    Y.segment<3>(0) = logMap(X.block<3, 3>(0, 0).inverse() * qy.toRotationMatrix());
    Matrix<double, 3, 21> H = Matrix<double, 3, 21>::Zero();
    H.block<3, 3>(0, 0) = Matrix3d::Identity();
    Matrix<double, 3, 3> N = Matrix<double, 3, 3>::Zero();
    N(0, 0) = leg_odom_ax * leg_odom_ax;
    N(1, 1) = leg_odom_ay * leg_odom_ay;
    N(2, 2) = leg_odom_az * leg_odom_az;
    Matrix<double, 7, 1> b = Matrix<double, 7, 1>::Zero();
    Matrix<double, 3, 7> PI = Matrix<double, 3, 7>::Zero();
    PI.block<3, 3>(0, 0) = Matrix3d::Identity();
    updateOrientation(Y, b, H, N, PI);
    updateVars();
}


void IMUinEKF::updateOrientation(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix<double, 3, 3> N_, Matrix<double, 3, 7> PI_)
{
    //Transform P to Left Invariant in order to perform the update
    Adj = Adjoint(X);
    P = Adj.inverse() * P * (Adj.inverse()).transpose();
    Matrix<double, 3, 3> S_ = N_;
    S_ += H_ * P * H_.transpose();
    Matrix<double, 21, 3> K_ = P * H_.transpose() * S_.inverse();

    Matrix<double, 7, 1> Z_;
    Z_.segment<3>(0) = Y_.segment<3>(0);
    //Update State
    Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
    Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

    X = X * dX_;
    theta += dtheta_;

    Matrix<double, 21, 21> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
    //Transform P to Right Invariant
    P = Adj * P * Adj.transpose();
}




void IMUinEKF::updateWithTwistOrient(Vector3d vy, Quaterniond qy)
{
    Matrix<double, 14, 1> Y = Matrix<double, 14, 1>::Zero();
    Y.segment<3>(0) = logMap(X.block<3, 3>(0, 0).inverse() * qy.toRotationMatrix());
    Y.segment<3>(7) = vy;
    Y(10) = 1.00;

    Matrix<double, 14, 1> b = Matrix<double, 14, 1>::Zero();

    b(10) = 1.00;
    Matrix<double, 6, 21> H = Matrix<double, 6, 21>::Zero();
    H.block<3, 3>(0, 0) = Matrix3d::Identity();
    H.block<3, 3>(3, 3) = Matrix3d::Identity();

    Matrix<double, 6, 6> N = Matrix<double, 6, 6>::Zero();
    N(0, 0) = leg_odom_ax * leg_odom_ax;
    N(1, 1) = leg_odom_ay * leg_odom_ay;
    N(2, 2) = leg_odom_az * leg_odom_az;
    N(3, 3) = vel_px * vel_px;
    N(4, 4) = vel_py * vel_py;
    N(5, 5) = vel_pz * vel_pz;

    Matrix<double, 6, 14> PI = Matrix<double, 6, 14>::Zero();
    PI.block<3, 3>(0, 0) = Matrix3d::Identity();
    PI.block<3, 3>(3, 7) = Matrix3d::Identity();

    updateVelocityOrientation(Y, b, H, N, PI);
    updateVars();
}

void IMUinEKF::updateVelocityOrientation(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_)
{
    //Transform P to Left Invariant in order to perform the update
    Adj = Adjoint(X);
    P = Adj.inverse() * P * (Adj.inverse()).transpose();
    Matrix<double, 6, 6> S_ = N_;
    S_ += H_ * P * H_.transpose();
    Matrix<double, 21, 6> K_ = P * H_.transpose() * S_.inverse();
    Matrix<double, 14, 14> BigX = Matrix<double, 14, 14>::Zero();
    BigX.block<7, 7>(0, 0) = X;
    BigX.block<7, 7>(7, 7) = X;
    Matrix<double, 14, 1> Z_ = BigX.inverse() * Y_ - b_;
    Z_.segment<3>(0) = Y_.segment<3>(0);
    //Update State
    Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
    Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

    X = X * dX_;
    theta += dtheta_;

    Matrix<double, 21, 21> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
    //Transform P to Right Invariant
    P = Adj * P * Adj.transpose();
}

void IMUinEKF::updateWithContacts(Vector3d s_pR, Vector3d s_pL, Matrix3d JRQeJR, Matrix3d JLQeJL, int contactR, int contactL)
{

    Rwb = X.block<3, 3>(0, 0);

    Qc(0, 0) = leg_odom_px * leg_odom_px;
    Qc(1, 1) = leg_odom_py * leg_odom_py;
    Qc(2, 2) = leg_odom_pz * leg_odom_pz;

    if (contactL && contactR)
    {
        Matrix<double, 14, 1> Y = Matrix<double, 14, 1>::Zero();
        Y.segment<3>(0) = s_pR;
        Y(4) = 1.00;
        Y(5) = -1.00;
        Y.segment<3>(7) = s_pL;
        Y(11) = 1.00;
        Y(13) = -1.00;
        Matrix<double, 14, 1> b = Matrix<double, 14, 1>::Zero();
        b(4) = 1.00;
        b(5) = -1.00;
        b(11) = 1.00;
        b(13) = -1.00;
        Matrix<double, 6, 21> H = Matrix<double, 6, 21>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 9) = Matrix3d::Identity();
        H.block<3, 3>(3, 6) = -Matrix3d::Identity();
        H.block<3, 3>(3, 12) = Matrix3d::Identity();

        Matrix<double, 6, 6> N = Matrix<double, 6, 6>::Zero();
        N.block<3, 3>(0, 0) = Rwb * JRQeJR * Rwb.transpose() + Qc;
        N.block<3, 3>(3, 3) = Rwb * JLQeJL * Rwb.transpose() + Qc;

        Matrix<double, 6, 14> PI = Matrix<double, 6, 14>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        PI.block<3, 3>(3, 7) = Matrix3d::Identity();
        updateStateDoubleContact(Y, b, H, N, PI);
        updateVars();
    }
    else if (contactR)
    {
        Matrix<double, 7, 1> Y = Matrix<double, 7, 1>::Zero();
        Y.segment<3>(0) = s_pR;
        Y(4) = 1.00;
        Y(5) = -1.00;
        Matrix<double, 7, 1> b = Matrix<double, 7, 1>::Zero();
        b(4) = 1.00;
        b(5) = -1.00;

        Matrix<double, 3, 21> H = Matrix<double, 3, 21>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 9) = Matrix3d::Identity();

        Matrix3d N = Matrix3d::Zero();
        N = Rwb * JRQeJR * Rwb.transpose() + Qc;
        Matrix<double, 3, 7> PI = Matrix<double, 3, 7>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        updateStateSingleContact(Y, b, H, N, PI);
        updateVars();
    }
    else if (contactL)
    {
        Matrix<double, 7, 1> Y = Matrix<double, 7, 1>::Zero();
        Y.segment<3>(0) = s_pL;
        Y(4) = 1.00;
        Y(6) = -1.00;
        Matrix<double, 7, 1> b = Matrix<double, 7, 1>::Zero();
        b(4) = 1.00;
        b(6) = -1.00;

        Matrix<double, 3, 21> H = Matrix<double, 3, 21>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 12) = Matrix3d::Identity();

        Matrix3d N = Matrix3d::Zero();
        N = Rwb * JLQeJL * Rwb.transpose()+ Qc;

        Matrix<double, 3, 7> PI = Matrix<double, 3, 7>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        updateStateSingleContact(Y, b, H, N, PI);
        updateVars();
    }
}

void IMUinEKF::updateStateSingleContact(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix3d N_, Matrix<double, 3, 7> PI_)
{

    Matrix3d S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 21, 3> K_ = P * H_.transpose() * S_.inverse();
    Matrix<double, 7, 1> Z_ = X * Y_ - b_;

    //Update State
    Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
    Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

    X = dX_ * X;
    theta += dtheta_;

    Matrix<double, 21, 21> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
}

void IMUinEKF::updateStateDoubleContact(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_)
{
    Matrix<double, 6, 6> S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 21, 6> K_ = P * H_.transpose() * S_.inverse();

    Matrix<double, 14, 14> BigX = Matrix<double, 14, 14>::Zero();

    BigX.block<7, 7>(0, 0) = X;
    BigX.block<7, 7>(7, 7) = X;

    Matrix<double, 14, 1> Z_ = BigX * Y_ - b_;

    //Update State
    Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;

    Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);
    X = dX_ * X;
    theta += dtheta_;

    Matrix<double, 21, 21> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
}

void IMUinEKF::updateVars()
{

    seperateState(X, theta, Rwb, vwb, pwb, dR, dL, bgyr, bacc);
    rX = pwb(0);
    rY = pwb(1);
    rZ = pwb(2);
    Tib.linear() = Rwb;
    Tib.translation() = pwb;
    qib = Quaterniond(Rwb);
    Rib = Rwb;

    bias_gx = bgyr(0);
    bias_gy = bgyr(1);
    bias_gz = bgyr(2);
    bias_ax = bacc(0);
    bias_ay = bacc(1);
    bias_az = bacc(2);

    w = w_ - bgyr;
    a = a_ - bacc;

    gyro = Rwb * w;
    gyroX = gyro(0);
    gyroY = gyro(1);
    gyroZ = gyro(2);

    acc = (Rwb * a + g);
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
