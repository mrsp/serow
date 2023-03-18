/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */


#include "IMUEKF.hpp"

IMUEKF::IMUEKF() {
    // Gravity Vector
    g = Vector3d::Zero();
    g(2) = -9.80;
    mahalanobis_TH = -1;
}

void IMUEKF::init() {
    firstrun = true;
    P_ = Matrix<double, 15, 15>::Zero();
    //  velocity error in base coordinates
    P_(0, 0) = 1e-3;
    P_(1, 1) = 1e-3;
    P_(2, 2) = 1e-3;
    //  Rotetional error in base coordinates
    P_(3, 3) = 1e-3;
    P_(4, 4) = 1e-3;
    P_(5, 5) = 1e-3;
    //  Positional error in world coordinates
    P_(6, 6) = 1e-5;
    P_(7, 7) = 1e-5;
    P_(8, 8) = 1e-5;
    //  Gyro and Acc Biases
    P_(9, 9) = 1e-3;
    P_(10, 10) = 1e-3;
    P_(11, 11) = 1e-3;
    P_(12, 12) = 1e-3;
    P_(13, 13) = 1e-3;
    P_(14, 14) = 1e-3;

    //  Beta-Bernoulli  parameters for outlier detection
    f0_ = 0.1;
    e0_ = 0.9;

    //  Compute some parts of the Input-Noise Jacobian once since they are
    // constants gyro (0), acc (3), gyro_bias (6), acc_bias (9)
    Lc_ = Matrix<double, 15, 12>::Zero();
    Lc_.block<3, 3>(0, 3) = -Matrix3d::Identity();
    Lc_.block<3, 3>(3, 0) = -Matrix3d::Identity();
    Lc_.block<3, 3>(9, 6) = Matrix3d::Identity();
    Lc_.block<3, 3>(12, 9) = Matrix3d::Identity();

    std::cout << "Floating Base Imu EKF Initialized Successfully" << std::endl;
}

Eigen::Matrix<double, 15, 15> IMUEKF::computeTransitionMatrix(const Eigen::Matrix<double, 15, 1>& x,
                                                              const Eigen::Matrix<double, 3, 3>& Rib,
                                                              Eigen::Vector3d imu_angular_velocity,
                                                              Eigen::Vector3d imu_linear_acceleration) {
    imu_angular_velocity -= x.segment<3>(9);
    imu_linear_acceleration -= x.segment<3>(12);
    const Eigen::Vector3d& v = x.segment<3>(0);
    Eigen::Matrix<double, 15, 15> res = Eigen::Matrix<double, 15, 15>::Zero();
    res.block<3, 3>(0, 0).noalias() = -wedge(imu_angular_velocity);
    res.block<3, 3>(0, 3).noalias() = wedge(Rib.transpose() * this->g_);
    res.block<3, 3>(0, 12).noalias() = -Eigen::Matrix3d::Identity();
    res.block<3, 3>(0, 9).noalias() = -wedge(v);
    res.block<3, 3>(3, 3).noalias() = -wedge(imu_angular_velocity);
    res.block<3, 3>(3, 9).noalias() = -Eigen::Matrix3d::Identity();
    res.block<3, 3>(6, 0) = Rib;
    res.block<3, 3>(6, 3).noalias() = -Rib * wedge(v);
    return res;
}

Eigen::Matrix<double, 15, 1> IMUEKF::computeDynamics(const Eigen::Matrix<double, 15, 1> x,
                                                     const Eigen::Matrix3d& Rib,
                                                     Eigen::Vector3d imu_angular_velocity,
                                                     Eigen::Vector3d imu_linear_acceleration,
                                                     double dt) {
    Eigen::Matrix<double, 15, 1> res = Eigen::Matrix<double, 15, 1>::Zero();

    imu_angular_velocity -= x.segment<3>(9);
    imu_linear_acceleration -= x.segment<3>(12);

    // Nonlinear Process Model
    // Compute \dot{v}_b @ k
    const Eigen::Vector3d& v = x.segment<3>(0);
    res.segment<3>(0) = v.cross(imu_angular_velocity);
    res.segment<3>(0) += Rib.transpose() * this->g_;
    res.segment<3>(0) += imu_linear_acceleration;

    // Position
    const Eigen::Vector3d& r = x.segment<3>(6);
    res.segment<3>(6).noalias() = Rib * res.segment<3>(0) * dt * dt / 2.00;
    res.segment<3>(6) += Rib * v * dt;
    res.segment<3>(6) += r;

    // Velocity
    res.segment<3>(0) *= dt;
    res.segment<3>(0) += v;

    // Biases
    res.segment<3>(9) = x.segment<3>(9);
    res.segment<3>(12) = x.segment<3>(12);
    return res;
}

void IMUEKF::predict(Eigen::Vector3d imu_angular_velocity, 
                     Eigen::Vector3d imu_linear_acceleration, 
                     double dt)
{
    // Used in updating Rib with the Rodriquez formula
    const Eigen::Vector3d& angular_velocity = imu_angular_velocity - x.segment<3>(9);
    const Eigen::Vector3d& v = x.segment<3>(0);

    // Update the Input-noise Jacobian
    Lc_.block<3, 3>(0, 0).noalias() = -wedge(v);

    // Compute the State dynamics Jacobian
    Eigen::Matrix<double, 15, 15> Ac =
        computeTransitionMatrix(x, Rib, imu_angular_velocity, imu_linear_acceleration);
    
    // Euler Discretization - First order Truncation
    Eigen::Matrix<double, 15, 15> Ad = Eigen::Matrix<double, 15, 15>::Identity();
    Ad += Ac * dt;
    // Propagete the state through the dynamics
    x = computeDynamics(x, Rib, imu_angular_velocity, imu_linear_acceleration, dt);

    // Propagate the orinetation seperately
    if (!angular_velocity.isZero()) {
        Rib *= expMap(angular_velocity * dt);
    }

    // Covariance Q with full state + biases
    Q_(0, 0) = gyr_qx * gyr_qx;
    Q_(1, 1) = gyr_qy * gyr_qy;
    Q_(2, 2) = gyr_qz * gyr_qz;
    Q_(3, 3) = acc_qx * acc_qx;
    Q_(4, 4) = acc_qy * acc_qy;
    Q_(5, 5) = acc_qz * acc_qz;
    Q_(6, 6) = gyrb_qx * gyrb_qx;
    Q_(7, 7) = gyrb_qy * gyrb_qy;
    Q_(8, 8) = gyrb_qz * gyrb_qz;
    Q_(9, 9) = accb_qx * accb_qx;
    Q_(10, 10) = accb_qy * accb_qy;
    Q_(11, 11) = accb_qz * accb_qz;

    Eigen::Matrix<double, 15, 15> Qd;
    Qd.noalias() = Ad * Lc_ * Q_ * Lc_.transpose() * Ad.transpose() * dt;

    // Predict Step: Propagate the Error Covariance 
    P_ = Ad * P_ * Ad.transpose() + Qd;

    // Reset orientation twist
    x.segment<3>(3) = Vector3d::Zero();
    updateVars();
}

/** Update **/
void IMUEKF::updateWithTwist(Vector3d y)
{
    Eigen::Matrix<double, 15, 3> Kv = Eigen::Matrix<double, 15, 3>::Zero();

    Matrix<double, 3, 3> Rv = Matrix<double, 3, 3>::Zero();

    Rv(0, 0) = vel_px * vel_px;
    Rv(1, 1) = vel_py * vel_py;
    Rv(2, 2) = vel_pz * vel_pz;

    v = x.segment<3>(0);
    // std::cout<<y<<std::endl;
    // Innovetion vector
    zv = y;
    zv.noalias() -= Rib * v;

    Matrix<double, 3, 15> Hv = Matrix<double, 3, 15>::Zero();
    Hv.block<3, 3>(0, 0) = Rib;
    Hv.block<3, 3>(0, 3).noalias() = -Rib * wedge(v);
    sv = Rv;
    sv.noalias() += Hv * P * Hv.transpose();
    Kv.noalias() = P * Hv.transpose() * sv.inverse();


    Eigen::Matrix<double, 15, 1> dxf;
    dxf.noalias() = Kv * zv;

    // Update the mean estimate
    x.noalias() += dxf;

    // Update the error covariance
    P = (If - Kv * Hv) * P * (If - Hv.transpose() * Kv.transpose());
    P.noalias() += Kv * Rv * Kv.transpose();

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dxf.segment<3>(3));
    }
    x.segment<3>(3) = Vector3d::Zero();

    updateVars();
}

void IMUEKF::updateWithBaseOrientation(const Eigen::Quaterniond& qy)
{

    R(0, 0) = vel_px * vel_px;
    R(1, 1) = vel_py * vel_py;
    R(2, 2) = vel_pz * vel_pz;

    z  = logMap((Rib.transpose() * qy.toRotationMatrix()));
    // z.segment<3>(3) = logMap((qy.toRotationMatrix() * Rib.transpose() ));
    H.block<3, 3>(0, 3) = Matrix3d::Identity();
    
    s = R;
    s.noalias() += Hvf * P * Hvf.transpose();
    K.noalias() = P * Hvf.transpose() * s.inverse();
    dx.noalias() = K * z;

    // Update the mean estimate
    x.noalias() += dxf;

    // Update the error covariance
    P = (If - Kf * Hvf) * P * (If - Hvf.transpose() * Kf.transpose());
    P.noalias() += Kf * R * Kf.transpose();

    if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
    {
        Rib *= expMap(dx.segment<3>(3));
    }
    x.segment<3>(3) = Vector3d::Zero();

    updateVars();
}


bool IMUEKF::updateWithOdom(Vector3d y, Quaterniond qy, bool useOutlierDetection)
{
    Matrix<double, 15, 6> Kf = Matrix<double, 15, 6>::Zero();
    Matrix<double, 6, 15> Hf = Matrix<double, 6, 15>::Zero();
    Hf.block<3, 3>(0, 6) = Matrix3d::Identity();
    Hf.block<3, 3>(3, 3) = Matrix3d::Identity();

    Matrix<double, 6, 6> R = Matrix<double, 6, 6>::Zero();

    R(0, 0) = odom_px * odom_px;
    R(1, 1) = odom_py * odom_py;
    R(2, 2) = odom_pz * odom_pz;

    R(3, 3) = odom_ax * odom_ax;
    R(4, 4) = odom_ay * odom_ay;
    R(5, 5) = odom_az * odom_az;

    outlier = false;
    if (!useOutlierDetection)
    {
        if (mahalanobis_TH <= 0)
        {
            r = x.segment<3>(6);
            // Innovetion vector
            z.segment<3>(0) = y - r;
            z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));

            // Compute the Kalman Gain
            s = R;
            s.noalias() += Hf * P * Hf.transpose();
            Kf.noalias() = P * Hf.transpose() * s.inverse();

            // Update the error covariance
            P = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
            P.noalias() += Kf * R * Kf.transpose();

            dxf.noalias() = Kf * z;
            // Update the mean estimate
            x += dxf;
            if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
            {
                Rib *= expMap(dxf.segment<3>(3));
            }
            x.segment<3>(3) = Vector3d::Zero();
        }
        else
        {
            r = x.segment<3>(6);
            // Innovetion vector
            z.segment<3>(0) = y - r;
            z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));
            // z.segment<3>(3) = logMap(qy.toRotationMatrix() * Rib.transpose());

            // Compute the Kalman Gain
            s = R;
            s.noalias() += Hf * P * Hf.transpose();
            Kf.noalias() = P * Hf.transpose() * s.inverse();

            // Update the error covariance
            P_i = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
            P_i.noalias() += Kf * R * Kf.transpose();
            dxf.noalias() = Kf * z;
            // Update the mean estimate
            x_i = x + dxf;

            if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
            {
                Rib_i = Rib * expMap(dxf.segment<3>(3));
            }
            s = Hf * P_i * Hf.transpose() + R;
            temp = y - x_i.segment<3>(6);
            double var_TH = temp.transpose() * s.block<3, 3>(0, 0).inverse() * temp;

            if (var_TH < mahalanobis_TH)
            {

                P = P_i;
                x = x_i;
                Rib = Rib_i;
                x.segment<3>(3) = Vector3d::Zero();
            }
            else
            {
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

        // Innovetion vector
        r = x.segment<3>(6);
        z.segment<3>(0) = y - r;
        z.segment<3>(3) = logMap((Rib.transpose() * qy.toRotationMatrix()));

        unsigned int j = 0;
        while (j < 4)
        {
            if (zeta > 1.0e-5)
            {
                // Compute the Kalman Gain
                R_z = R / zeta;
                s = R_z;
                s.noalias() += Hf * P * Hf.transpose();
                Kf.noalias() = P * Hf.transpose() * s.inverse();

                // Update the error covariance
                P_i = (If - Kf * Hf) * P * (If - Kf * Hf).transpose();
                P_i.noalias() += Kf * R_z * Kf.transpose();
                dxf.noalias() = Kf * z;

                // Update the mean estimate
                x_i_ = x_i;
                x_i = x + dxf;

                if (dxf(3) != 0 || dxf(4) != 0 || dxf(5) != 0)
                {
                    Rib_i = Rib * expMap(dxf.segment<3>(3));
                }
                x_i.segment<3>(3) = Vector3d::Zero();

                // outlier detection with the position measurement vector
                tempM = y * y.transpose();
                tempM.noalias() -= 2.0 * y * x_i.segment<3>(6).transpose();
                tempM.noalias() += x_i.segment<3>(6) *
                    x_i.segment<3>(6).transpose();
                tempM.noalias() += P_i.block<3, 3>(6, 6);
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

// Update the outlier indicator Zeta
void IMUEKF::updateOutlierDetectionParams(Eigen::Matrix<double, 3, 3> BetaT)
{
    efpsi = computePsi(e_t + f_t);
    lnp = computePsi(e_t) - efpsi;
    ln1_p = computePsi(f_t) - efpsi;

    tempM = BetaT * (R.block<3, 3>(0, 0)).inverse();

    pzeta_1 = exp(lnp - 0.5 * (tempM).trace());
    pzeta_0 = exp(ln1_p);

    // Normalization factor
    norm_factor = 1.0 / (pzeta_1 + pzeta_0);

    // p(zeta) are now proper probabilities
    pzeta_1 = norm_factor * pzeta_1;
    pzeta_0 = norm_factor * pzeta_0;

    // mean of Bernulli
    zeta = pzeta_1 / (pzeta_1 + pzeta_0);

    // Update epsilon and f
    e_t = e0 + zeta;
    f_t = f0 + 1.0 - zeta;
}

// Digamma Function Approximation
double IMUEKF::computePsi(double xxx)
{

    double result = 0, xx, xx2, xx4;
    for (; xxx < 7; ++xxx)
        result -= 1 / xxx;

    xxx -= 1.0 / 2.0;
    xx = 1.0 / xxx;
    xx2 = xx * xx;
    xx4 = xx2 * xx2;
    result += log(xxx) + (1. / 24.) * xx2 - (7.0 / 960.0) * xx4 +
        (31.0 / 8064.0) * xx4 * xx2 - (127.0 / 30720.0) * xx4 * xx4;
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

    // Update the biases
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

    // ROLL - PITCH - YAW
    angle = getEulerAngles(Rib);
    angleX = angle(0);
    angleY = angle(1);
    angleZ = angle(2);
}
