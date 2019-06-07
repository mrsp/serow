#include <serow/IMUinEKFQuad.h>

using namespace std;

IMUinEKFQuad::IMUinEKFQuad()
{
    //Gravity Vector
    g = Vector3d::Zero();
    g(2) = -9.80;
    P = Matrix<double, 27, 27>::Zero();
    X = Matrix<double, 9, 9>::Identity();
    theta = Matrix<double, 6, 1>::Zero();
}

void IMUinEKFQuad::init()
{

    firstrun = true;
    If = Matrix<double, 27, 27>::Identity();
    I = Matrix3d::Identity();
    w_ = Vector3d::Zero();
    a_ = Vector3d::Zero();
    a = Vector3d::Zero();
    w = Vector3d::Zero();

    /* State is X where
    X(0:2,0:2) is Rwb 
    X(0:2,3) is v_wb 
    X(0:2,4) is p_wb
    X(0:2,5) is dRF
    X(0:2,6) is dRH
    X(0:2,7) is dLF
    X(0:2,8) is dLH
    Parameters are theta where
    theta(0:2) is Gyro Bias
    thata(3:5) is Acc Bias
    */

    //Rot error
    P(0, 0) = 1e-2;
    P(1, 1) = 1e-2;
    P(2, 2) = 1e-2;
    //Vel error
    P(3, 3) = 1e-2;
    P(4, 4) = 1e-2;
    P(5, 5) = 1e-2;
    //Pos error
    P(6, 6) = 1e-3;
    P(7, 7) = 1e-3;
    P(8, 8) = 1e-3;
    //dRF error
    P(9, 9) = 1e-3;
    P(10, 10) = 1e-3;
    P(11, 11) = 1e-3;
    //dRH error
    P(12, 12) = 1e-3;
    P(13, 13) = 1e-3;
    P(14, 14) = 1e-3;
    //dLF error
    P(15, 15) = 1e-3;
    P(16, 16) = 1e-3;
    P(17, 17) = 1e-3;
    //dLH error
    P(18, 18) = 1e-3;
    P(19, 19) = 1e-3;
    P(20, 20) = 1e-3;
    //Biases
    //Gyro
    P(21, 21) = 1e-2;
    P(22, 22) = 1e-2;
    P(23, 23) = 1e-2;
    //Acc
    P(24, 24) = 1e-1;
    P(25, 25) = 1e-1;
    P(26, 26) = 1e-1;

    Af = Matrix<double, 27, 27>::Zero();
    Af.block<3, 3>(3, 0).noalias() = skew(g);
    Af.block<3, 3>(6, 3).noalias() = Matrix3d::Identity();

    Qff = Matrix<double, 27, 27>::Zero();
    Qf = Matrix<double, 27, 27>::Zero();
    Qc = Matrix3d::Zero();
    Adj = Matrix<double, 27, 27>::Zero();
    Phi = Matrix<double, 27, 27>::Zero();
    Rwb = Matrix3d::Identity();
    Rib = Rwb;
    pwb = Vector3d::Zero();
    vwb = Vector3d::Zero();
    dLF = Vector3d::Zero();
    dLH = Vector3d::Zero();
    dRF = Vector3d::Zero();
    dRH = Vector3d::Zero();
    bgyr = Vector3d::Zero();
    bacc = Vector3d::Zero();
    gyro = Vector3d::Zero();
    acc = Vector3d::Zero();
    angle = Vector3d::Zero();
    Tib = Affine3d::Identity();

    std::cout << "IMU Right-Invariant EKF Initialized Successfully" << std::endl;
}

void IMUinEKFQuad::constructState(Matrix<double, 9, 9> &X_, Matrix<double, 6, 1> &theta_, Matrix3d R_, Vector3d v_, Vector3d p_, Vector3d dRF_,  Vector3d dRH_, Vector3d dLF_, Vector3d dLH_, Vector3d bg_, Vector3d ba_)
{
    X_.block<3, 3>(0, 0) = R_;
    X_.block<3, 1>(0, 3) = v_;
    X_.block<3, 1>(0, 4) = p_;
    X_.block<3, 1>(0, 5) = dRF_;
    X_.block<3, 1>(0, 6) = dRH_;
    X_.block<3, 1>(0, 7) = dLF_;
    X_.block<3, 1>(0, 8) = dLH_;

    theta_.segment<3>(0) = bg_;
    theta_.segment<3>(3) = ba_;
}

void IMUinEKFQuad::seperateState(Matrix<double, 9, 9> X_, Matrix<double, 6, 1> theta_, Matrix3d &R_, Vector3d &v_, Vector3d &p_, Vector3d &dRF_, Vector3d &dRH_,  Vector3d &dLF_, Vector3d &dLH_, Vector3d &bg_, Vector3d &ba_)
{
    R_ = X_.block<3, 3>(0, 0);
    v_ = X_.block<3, 1>(0, 3);
    p_ = X_.block<3, 1>(0, 4);
   
    dRF_ = X_.block<3, 1>(0, 5);
    dRH_ = X_.block<3, 1>(0, 6);
    dLF_ = X_.block<3, 1>(0, 7);
    dLH_ = X_.block<3, 1>(0, 8);

    bg_ = theta_.segment<3>(0);
    ba_ = theta_.segment<3>(3);
}

Matrix<double, 9, 9> IMUinEKFQuad::exp_SE3(Matrix<double, 21, 1> v)
{
    Matrix<double, 9, 9> dX_ = Matrix<double, 9, 9>::Identity();
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
    dX_.block<3, 1>(0, 7).noalias() = Jr_ * v.segment<3>(15);
    dX_.block<3, 1>(0, 8).noalias() = Jr_ * v.segment<3>(18);
    return dX_;
}

Matrix3d IMUinEKFQuad::exp_SO3(Vector3d v)
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

Matrix<double, 27, 27> IMUinEKFQuad::Adjoint(Matrix<double, 9, 9> X_)
{
    Matrix<double, 27, 27> AdjX = Matrix<double, 27, 27>::Identity();
    Matrix3d R_;
    Vector3d v_, p_, dRF_, dRH_, dLF_, dLH_, bg_, ba_;
    seperateState(X_, Matrix<double, 6, 1>::Zero(), R_, v_, p_, dRF_, dRH_, dLF_, dLH_, bg_, ba_);

    AdjX.block<3, 3>(0, 0) = R_;
    AdjX.block<3, 3>(3, 3) = R_;
    AdjX.block<3, 3>(6, 6) = R_;
    AdjX.block<3, 3>(9, 9) = R_;
    AdjX.block<3, 3>(12, 12) = R_;
    AdjX.block<3, 3>(15, 15) = R_;
    AdjX.block<3, 3>(18, 18) = R_;


    AdjX.block<3, 3>(3, 0).noalias() = skew(v_) * R_;
    AdjX.block<3, 3>(6, 0).noalias() = skew(p_) * R_;
    AdjX.block<3, 3>(9, 0).noalias() = skew(dRF_) * R_;
    AdjX.block<3, 3>(12, 0).noalias() = skew(dRH_) * R_;
    AdjX.block<3, 3>(15, 0).noalias() = skew(dLF_) * R_;
    AdjX.block<3, 3>(18, 0).noalias() = skew(dLH_) * R_;

    return AdjX;
}

void IMUinEKFQuad::predict(Vector3d angular_velocity, Vector3d linear_acceleration, Vector3d pbRF, Vector3d pbRH, Vector3d pbLF, Vector3d pbLH, Matrix3d hR_RF, Matrix3d hR_RH,  Matrix3d hR_LF, Matrix3d hR_LH, int contactRF, int contactRH, int contactLF, int contactLH)
{

    seperateState(X, theta, Rwb, vwb, pwb, dRF, dRH, dLF, dLH, bgyr, bacc);
    w_ = angular_velocity;
    a_ = linear_acceleration;

    //Bias removed gyro and acc
    w = w_ - bgyr;
    a = a_ - bacc;

    Af.block<3, 3>(0, 15).noalias() = -Rwb;
    Af.block<3, 3>(3, 18).noalias() = -Rwb;
    Af.block<3, 3>(3, 15).noalias() = -skew(vwb) * Rwb;
    Af.block<3, 3>(6, 15).noalias() = -skew(pwb) * Rwb;
    Af.block<3, 3>(9, 15).noalias() = -skew(dRF) * Rwb;
    Af.block<3, 3>(12, 15).noalias() = -skew(dRH) * Rwb;
    Af.block<3, 3>(15, 15).noalias() = -skew(dLF) * Rwb;
    Af.block<3, 3>(18, 15).noalias() = -skew(dLH) * Rwb;



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

    Qf.block<3, 3>(9, 9) = hR_RF * (Qc + 1e4 * I * (1 - contactRF)) * hR_RF.transpose();
    Qf.block<3, 3>(12, 12) = hR_RH * (Qc + 1e4 * I * (1 - contactRH)) * hR_RH.transpose();

    Qf.block<3, 3>(15, 15) = hR_LF * (Qc + 1e4 * I * (1 - contactLF)) * hR_LF.transpose();
    Qf.block<3, 3>(18, 18) = hR_LH * (Qc + 1e4 * I * (1 - contactLH)) * hR_LH.transpose();




    Qf(21, 21) = gyrb_qx * gyrb_qx;
    Qf(22, 22) = gyrb_qy * gyrb_qy;
    Qf(23, 23) = gyrb_qz * gyrb_qz;
    Qf(24, 24) = accb_qx * accb_qx;
    Qf(25, 25) = accb_qy * accb_qy;
    Qf(26, 26) = accb_qz * accb_qz;

    Qff.noalias() = Phi * Adj * Qf * Adj.transpose() * Phi.transpose() * dt;

    /** Predict Step: Propagate the Error Covariance  **/
    P = Phi * P * Phi.transpose();
    P.noalias() += Qff;

    pwb += vwb * dt + 0.5 * (Rwb * a + g) * dt * dt;
    vwb += (Rwb * a + g) * dt;
    Rwb *= exp_SO3(w * dt);
    //Foot Position Dynamics
    dRF = contactRF * dRF + (1 - contactRF) * (pwb + Rwb * pbRF);
    dRH = contactRH * dRH + (1 - contactRH) * (pwb + Rwb * pbRH);

    dLF = contactLF * dLF + (1 - contactLF) * (pwb + Rwb * pbLF);
    dLH = contactLH * dLH + (1 - contactLH) * (pwb + Rwb * pbLH);

    constructState(X, theta, Rwb, vwb, pwb, dRF, dRH, dLF, dLH, bgyr, bacc);
    updateVars();
}

// void IMUinEKFQuad::updateWithTwist(Vector3d vy, Matrix3d Rvy)
// {
//     Matrix<double, 7, 1> Y = Matrix<double, 7, 1>::Zero();
//     Y.segment<3>(0) = vy;
//     Y(3) = 1.00;

//     Matrix<double, 7, 1> b = Matrix<double, 7, 1>::Zero();
//     b(3) = 1.00;
//     Matrix<double, 3, 21> H = Matrix<double, 3, 21>::Zero();
//     H.block<3, 3>(0, 3) = Matrix3d::Identity();

//     Matrix<double, 3, 3> N = Matrix<double, 3, 3>::Zero();
//     N = Rvy;
//     N(0, 0) += vel_px * vel_px;
//     N(1, 1) += vel_py * vel_py;
//     N(2, 2) += vel_pz * vel_pz;

//     Matrix<double, 3, 7> PI = Matrix<double, 3, 7>::Zero();
//     PI.block<3, 3>(0, 0) = Matrix3d::Identity();
//     updateVelocity(Y, b, H, N, PI);
//     updateVars();
// }
// void IMUinEKFQuad::updateVelocity(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix3d N_, Matrix<double, 3, 7> PI_)
// {
//     //Transform P to Left Invariant in order to perform the update
//     Adj = Adjoint(X);
//     P = Adj.inverse() * P * (Adj.inverse()).transpose();

//     Matrix3d S_ = N_;
//     S_ += H_ * P * H_.transpose();

//     Matrix<double, 21, 3> K_ = P * H_.transpose() * S_.inverse();
//     Matrix<double, 7, 1> Z_ = X.inverse() * Y_ - b_;

//     //Update State
//     Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
//     Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
//     Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

//     X = X * dX_;
//     theta += dtheta_;

//     Matrix<double, 21, 21> IKH = If - K_ * H_;
//     P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
//     //Transform P to Right Invariant
//     P = Adj * P * Adj.transpose();
// }

// void IMUinEKFQuad::updateWithOdom(Vector3d py, Quaterniond qy)
// {
//     Matrix<double, 14, 1> Y = Matrix<double, 14, 1>::Zero();
//     Y.segment<3>(0) = logMap(X.block<3, 3>(0, 0).inverse() * qy.toRotationMatrix());
//     Y.segment<3>(7) = py;
//     Y(11) = 1.000;

//     Matrix<double, 14, 1> b = Matrix<double, 14, 1>::Zero();
//     b(11) = 1.000;
//     Matrix<double, 6, 21> H = Matrix<double, 6, 21>::Zero();
//     H.block<3, 3>(0, 0) = Matrix3d::Identity();
//     H.block<3, 3>(3, 6) = Matrix3d::Identity();

//     Matrix<double, 6, 6> N = Matrix<double, 6, 6>::Zero();
//     N(0, 0) = odom_ax * odom_ax;
//     N(1, 1) = odom_ay * odom_ay;
//     N(2, 2) = odom_az * odom_az;
//     N(3, 3) = odom_px * odom_px;
//     N(4, 4) = odom_py * odom_py;
//     N(5, 5) = odom_pz * odom_pz;

//     Matrix<double, 6, 14> PI = Matrix<double, 6, 14>::Zero();
//     PI.block<3, 3>(0, 0) = Matrix3d::Identity();
//     PI.block<3, 3>(3, 7) = Matrix3d::Identity();

//     updatePositionOrientation(Y, b, H, N, PI);
//     updateVars();
// }

// void IMUinEKFQuad::updatePositionOrientation(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_)
// {
//     //Transform P to Left Invariant in order to perform the update
//     Adj = Adjoint(X);
//     P = Adj.inverse() * P * (Adj.inverse()).transpose();

//     Matrix<double, 6, 6> S_ = N_;
//     S_ += H_ * P * H_.transpose();

//     Matrix<double, 21, 6> K_ = P * H_.transpose() * S_.inverse();

//     Matrix<double, 14, 14> BigX = Matrix<double, 14, 14>::Zero();

//     BigX.block<7, 7>(0, 0) = X;
//     BigX.block<7, 7>(7, 7) = X;

//     Matrix<double, 14, 1> Z_ = BigX.inverse() * Y_ - b_;
//     Z_.segment<3>(0) = Y_.segment<3>(0);

//     //Update State
//     Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
//     Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
//     Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

//     X = X * dX_;
//     theta += dtheta_;

//     Matrix<double, 21, 21> IKH = If - K_ * H_;
//     P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
//     //Transform P to Right Invariant
//     P = Adj * P * Adj.transpose();
// }

// void IMUinEKFQuad::updateWithOrient(Quaterniond qy)
// {
//     Matrix<double, 7, 1> Y = Matrix<double, 7, 1>::Zero();

//     Y.segment<3>(0) = logMap(X.block<3, 3>(0, 0).inverse() * qy.toRotationMatrix());
//     Matrix<double, 3, 21> H = Matrix<double, 3, 21>::Zero();
//     H.block<3, 3>(0, 0) = Matrix3d::Identity();
//     Matrix<double, 3, 3> N = Matrix<double, 3, 3>::Zero();
//     N(0, 0) = leg_odom_ax * leg_odom_ax;
//     N(1, 1) = leg_odom_ay * leg_odom_ay;
//     N(2, 2) = leg_odom_az * leg_odom_az;
//     Matrix<double, 7, 1> b = Matrix<double, 7, 1>::Zero();
//     Matrix<double, 3, 7> PI = Matrix<double, 3, 7>::Zero();
//     PI.block<3, 3>(0, 0) = Matrix3d::Identity();
//     updateOrientation(Y, b, H, N, PI);
//     updateVars();
// }


// void IMUinEKFQuad::updateOrientation(Matrix<double, 7, 1> Y_, Matrix<double, 7, 1> b_, Matrix<double, 3, 21> H_, Matrix<double, 3, 3> N_, Matrix<double, 3, 7> PI_)
// {
//     //Transform P to Left Invariant in order to perform the update
//     Adj = Adjoint(X);
//     P = Adj.inverse() * P * (Adj.inverse()).transpose();
//     Matrix<double, 3, 3> S_ = N_;
//     S_ += H_ * P * H_.transpose();
//     Matrix<double, 21, 3> K_ = P * H_.transpose() * S_.inverse();

//     Matrix<double, 7, 1> Z_;
//     Z_.segment<3>(0) = Y_.segment<3>(0);
//     //Update State
//     Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
//     Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
//     Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

//     X = X * dX_;
//     theta += dtheta_;

//     Matrix<double, 21, 21> IKH = If - K_ * H_;
//     P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
//     //Transform P to Right Invariant
//     P = Adj * P * Adj.transpose();
// }




// void IMUinEKFQuad::updateWithTwistOrient(Vector3d vy, Quaterniond qy)
// {
//     Matrix<double, 14, 1> Y = Matrix<double, 14, 1>::Zero();
//     Y.segment<3>(0) = logMap(X.block<3, 3>(0, 0).inverse() * qy.toRotationMatrix());
//     Y.segment<3>(7) = vy;
//     Y(10) = 1.00;

//     Matrix<double, 14, 1> b = Matrix<double, 14, 1>::Zero();

//     b(10) = 1.00;
//     Matrix<double, 6, 21> H = Matrix<double, 6, 21>::Zero();
//     H.block<3, 3>(0, 0) = Matrix3d::Identity();
//     H.block<3, 3>(3, 3) = Matrix3d::Identity();

//     Matrix<double, 6, 6> N = Matrix<double, 6, 6>::Zero();
//     N(0, 0) = leg_odom_ax * leg_odom_ax;
//     N(1, 1) = leg_odom_ay * leg_odom_ay;
//     N(2, 2) = leg_odom_az * leg_odom_az;
//     N(3, 3) = vel_px * vel_px;
//     N(4, 4) = vel_py * vel_py;
//     N(5, 5) = vel_pz * vel_pz;

//     Matrix<double, 6, 14> PI = Matrix<double, 6, 14>::Zero();
//     PI.block<3, 3>(0, 0) = Matrix3d::Identity();
//     PI.block<3, 3>(3, 7) = Matrix3d::Identity();

//     updateVelocityOrientation(Y, b, H, N, PI);
//     updateVars();
// }

// void IMUinEKFQuad::updateVelocityOrientation(Matrix<double, 14, 1> Y_, Matrix<double, 14, 1> b_, Matrix<double, 6, 21> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 14> PI_)
// {
//     //Transform P to Left Invariant in order to perform the update
//     Adj = Adjoint(X);
//     P = Adj.inverse() * P * (Adj.inverse()).transpose();
//     Matrix<double, 6, 6> S_ = N_;
//     S_ += H_ * P * H_.transpose();
//     Matrix<double, 21, 6> K_ = P * H_.transpose() * S_.inverse();
//     Matrix<double, 14, 14> BigX = Matrix<double, 14, 14>::Zero();
//     BigX.block<7, 7>(0, 0) = X;
//     BigX.block<7, 7>(7, 7) = X;
//     Matrix<double, 14, 1> Z_ = BigX.inverse() * Y_ - b_;
//     Z_.segment<3>(0) = Y_.segment<3>(0);
//     //Update State
//     Matrix<double, 21, 1> delta_ = K_ * PI_ * Z_;
//     Matrix<double, 7, 7> dX_ = exp_SE3(delta_.segment<15>(0));
//     Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(15);

//     X = X * dX_;
//     theta += dtheta_;

//     Matrix<double, 21, 21> IKH = If - K_ * H_;
//     P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
//     //Transform P to Right Invariant
//     P = Adj * P * Adj.transpose();
// }

void IMUinEKFQuad::updateWithContacts(Vector3d s_pRF, Vector3d s_pRH, Vector3d s_pLF, Vector3d s_pLH, Matrix3d JRFQeJRF, Matrix3d JRHQeJRH, Matrix3d JLFQeJLF,  Matrix3d JLHQeJLH, int contactRF, int contactRH, int contactLF, int contactLH)
{

    Rwb = X.block<3, 3>(0, 0);

    Qc(0, 0) = leg_odom_px * leg_odom_px;
    Qc(1, 1) = leg_odom_py * leg_odom_py;
    Qc(2, 2) = leg_odom_pz * leg_odom_pz;

    // if (contactLF && contactRF && contactLH && contactRH)
    // {
    //     Matrix<double, 36, 1> Y = Matrix<double, 36, 1>::Zero();
    //     Y.segment<3>(0) = s_pRF;
    //     Y(4) = 1.00;
    //     Y(5) = -1.00;
        
    //     Y.segment<3>(9) = s_pRH;
    //     Y(13) = 1.00;
    //     Y(15) = -1.00;

    //     Y.segment<3>(0) = s_pLF;
    //     Y(22) = 1.00;
    //     Y(25) = -1.00;
        
    //     Y.segment<3>(9) = s_pLH;
    //     Y(31) = 1.00;
    //     Y(35) = -1.00;



    //     Matrix<double, 36, 1> b = Matrix<double, 36, 1>::Zero();
    //     b(4) = 1.00;
    //     b(5) = -1.00;
    //     b(13) = 1.00;
    //     b(15) = -1.00;
    //     b(22) = 1.00;
    //     b(25) = -1.00;
    //     b(31) = 1.00;
    //     b(35) = -1.00;

    //     Matrix<double, 12, 27> H = Matrix<double, 12, 27>::Zero();
    //     H.block<3, 3>(0, 6) = -Matrix3d::Identity();
    //     H.block<3, 3>(0, 9) = Matrix3d::Identity();
    //     H.block<3, 3>(3, 6) = -Matrix3d::Identity();
    //     H.block<3, 3>(3, 12) = Matrix3d::Identity();

    //     Matrix<double, 12, 12> N = Matrix<double, 12, 12>::Zero();
    //     N.block<3, 3>(0, 0) = Rwb * JRFQeJRF * Rwb.transpose() + Qc;
    //     N.block<3, 3>(3, 3) = Rwb * JRHQeJRH * Rwb.transpose() + Qc;
    //     N.block<3, 3>(6, 6) = Rwb * JLFQeJLF * Rwb.transpose() + Qc;
    //     N.block<3, 3>(9, 9) = Rwb * JLHQeJLH * Rwb.transpose() + Qc;

    //     Matrix<double, 12, 36> PI = Matrix<double, 12, 36>::Zero();
    //     PI.block<3, 3>(0, 0) = Matrix3d::Identity();
    //     PI.block<3, 3>(3, 9) = Matrix3d::Identity();
    //     PI.block<3, 3>(6, 18) = Matrix3d::Identity();
    //     PI.block<3, 3>(9, 27) = Matrix3d::Identity();

    //     updateStateQuadContact(Y, b, H, N, PI);
    //     updateVars();
    // }
    if (contactRF)
    {
        Matrix<double, 9, 1> Y = Matrix<double, 9, 1>::Zero();
        Y.segment<3>(0) = s_pRF;
        Y(4) = 1.00;
        Y(5) = -1.00;
        Matrix<double, 9, 1> b = Matrix<double, 9, 1>::Zero();
        b(4) = 1.00;
        b(5) = -1.00;

        Matrix<double, 3, 27> H = Matrix<double, 3, 27>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 9) = Matrix3d::Identity();

        Matrix3d N = Matrix3d::Zero();
        N = Rwb * JRFQeJRF * Rwb.transpose() + Qc;
        Matrix<double, 3, 9> PI = Matrix<double, 3, 9>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        updateStateSingleContact(Y, b, H, N, PI);
        updateVars();
    }
    if (contactRH)
    {
        Matrix<double, 9, 1> Y = Matrix<double, 9, 1>::Zero();
        Y.segment<3>(0) = s_pRH;
        Y(4) = 1.00;
        Y(6) = -1.00;
        Matrix<double, 9, 1> b = Matrix<double, 9, 1>::Zero();
        b(4) = 1.00;
        b(6) = -1.00;

        Matrix<double, 3, 27> H = Matrix<double, 3, 27>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 12) = Matrix3d::Identity();

        Matrix3d N = Matrix3d::Zero();
        N = Rwb * JRHQeJRH * Rwb.transpose()+ Qc;

        Matrix<double, 3, 9> PI = Matrix<double, 3, 9>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        updateStateSingleContact(Y, b, H, N, PI);
        updateVars();
    }
    if (contactLF)
    {
        Matrix<double, 9, 1> Y = Matrix<double, 9, 1>::Zero();
        Y.segment<3>(0) = s_pLF;
        Y(4) = 1.00;
        Y(7) = -1.00;
        Matrix<double, 9, 1> b = Matrix<double, 9, 1>::Zero();
        b(4) = 1.00;
        b(7) = -1.00;

        Matrix<double, 3, 27> H = Matrix<double, 3, 27>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 15) = Matrix3d::Identity();

        Matrix3d N = Matrix3d::Zero();
        N = Rwb * JLFQeJLF * Rwb.transpose() + Qc;
        Matrix<double, 3, 9> PI = Matrix<double, 3, 9>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        updateStateSingleContact(Y, b, H, N, PI);
        updateVars();
    }
    if (contactLH)
    {
        Matrix<double, 9, 1> Y = Matrix<double, 9, 1>::Zero();
        Y.segment<3>(0) = s_pLH;
        Y(4) = 1.00;
        Y(8) = -1.00;
        Matrix<double, 9, 1> b = Matrix<double, 9, 1>::Zero();
        b(4) = 1.00;
        b(8) = -1.00;

        Matrix<double, 3, 27> H = Matrix<double, 3, 27>::Zero();
        H.block<3, 3>(0, 6) = -Matrix3d::Identity();
        H.block<3, 3>(0, 18) = Matrix3d::Identity();

        Matrix3d N = Matrix3d::Zero();
        N = Rwb * JLHQeJLH * Rwb.transpose()+ Qc;

        Matrix<double, 3, 9> PI = Matrix<double, 3, 9>::Zero();
        PI.block<3, 3>(0, 0) = Matrix3d::Identity();
        updateStateSingleContact(Y, b, H, N, PI);
        updateVars();
    }
}

void IMUinEKFQuad::updateStateSingleContact(Matrix<double, 9, 1> Y_, Matrix<double, 9, 1> b_, Matrix<double, 3, 27> H_, Matrix3d N_, Matrix<double, 3, 9> PI_)
{

    Matrix3d S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 27, 3> K_ = P * H_.transpose() * S_.inverse();
    Matrix<double, 9, 1> Z_ = X * Y_ - b_;

    //Update State
    Matrix<double, 27, 1> delta_ = K_ * PI_ * Z_;
    Matrix<double, 9, 9> dX_ = exp_SE3(delta_.segment<21>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(21);

    X = dX_ * X;
    theta += dtheta_;

    Matrix<double, 27, 27> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
}

void IMUinEKFQuad::updateStateDoubleContact(Matrix<double, 18, 1> Y_, Matrix<double, 18, 1> b_, Matrix<double, 6, 27> H_, Matrix<double, 6, 6> N_, Matrix<double, 6, 18> PI_)
{
    Matrix<double, 6, 6> S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 27, 6> K_ = P * H_.transpose() * S_.inverse();

    Matrix<double, 18, 18> BigX = Matrix<double, 18, 18>::Zero();

    BigX.block<9, 9>(0, 0) = X;
    BigX.block<9, 9>(9, 9) = X;

    Matrix<double, 18, 1> Z_ = BigX * Y_ - b_;

    //Update State
    Matrix<double, 27, 1> delta_ = K_ * PI_ * Z_;

    Matrix<double, 9, 9> dX_ = exp_SE3(delta_.segment<21>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(21);
    X = dX_ * X;
    theta += dtheta_;

    Matrix<double, 27, 27> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
}

void IMUinEKFQuad::updateStateTripleContact(Matrix<double, 27, 1> Y_, Matrix<double, 27, 1> b_, Matrix<double, 9, 27> H_, Matrix<double, 9, 9> N_, Matrix<double, 9, 27> PI_)
{
    Matrix<double, 9, 9> S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 27, 9> K_ = P * H_.transpose() * S_.inverse();

    Matrix<double, 27, 27> BigX = Matrix<double, 27, 27>::Zero();

    BigX.block<9, 9>(0, 0) = X;
    BigX.block<9, 9>(9, 9) = X;
    BigX.block<9, 9>(18, 18) = X;

    Matrix<double, 27, 1> Z_ = BigX * Y_ - b_;

    //Update State
    Matrix<double, 27, 1> delta_ = K_ * PI_ * Z_;

    Matrix<double, 9, 9> dX_ = exp_SE3(delta_.segment<21>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(21);
    X = dX_ * X;
    theta += dtheta_;

    Matrix<double, 27, 27> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
}

void IMUinEKFQuad::updateStateQuadContact(Matrix<double, 36, 1> Y_, Matrix<double, 36, 1> b_, Matrix<double, 12, 27> H_, Matrix<double, 12, 12> N_, Matrix<double, 12, 36> PI_)
{
    Matrix<double, 12, 12> S_ = N_;
    S_ += H_ * P * H_.transpose();

    Matrix<double, 27, 12> K_ = P * H_.transpose() * S_.inverse();

    Matrix<double, 36, 36> BigX = Matrix<double, 36, 36>::Zero();

    BigX.block<9, 9>(0, 0) = X;
    BigX.block<9, 9>(9, 9) = X;
    BigX.block<9, 9>(18, 18) = X;
    BigX.block<9, 9>(27, 27) = X;

    Matrix<double, 36, 1> Z_ = BigX * Y_ - b_;

    //Update State
    Matrix<double, 27, 1> delta_ = K_ * PI_ * Z_;

    Matrix<double, 9, 9> dX_ = exp_SE3(delta_.segment<21>(0));
    Matrix<double, 6, 1> dtheta_ = delta_.segment<6>(21);
    X = dX_ * X;
    theta += dtheta_;

    Matrix<double, 27, 27> IKH = If - K_ * H_;
    P = IKH * P * IKH.transpose() + K_ * N_ * K_.transpose();
}


void IMUinEKFQuad::updateVars()
{

    seperateState(X, theta, Rwb, vwb, pwb, dRF, dRH, dLF, dLH, bgyr, bacc);
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
