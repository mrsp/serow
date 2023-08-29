#include "CoMEKF.hpp"

#include <iostream>

void CoMEKF::init(double mass, double rate, double I_xx, double I_yy) {
    I_ = Eigen::Matrix<double, 9, 9>::Identity();
    P_ = Eigen::Matrix<double, 9, 9>::Zero();
    P_.block<3, 3>(0, 0) = 1e-6 * Eigen::Matrix<double, 3, 3>::Identity();
    P_.block<3, 3>(3, 3) = 1e-2 * Eigen::Matrix<double, 3, 3>::Identity();
    P_.block<3, 3>(6, 6) = 1e-1 * Eigen::Matrix<double, 3, 3>::Identity();
    mass_ = mass;
    nominal_dt_ = 1.0 / rate;
    I_xx_ = I_xx;
    I_yy_ = I_yy;
    g_ = 9.81;
    std::cout << "Nonlinear CoM Estimator Initialized!" << std::endl;
}

State CoMEKF::predict(State state, KinematicMeasurement kin, GroundReactionForceMeasurement grf) {
    State predicted_state = state;
    const auto& [Ac, Lc] =
        computePredictionJacobians(state, grf.COP, grf.force, kin.com_angular_momentum);

    // Discretization
    Eigen::Matrix<double, 9, 9> Qd;
    Qd.block<3, 3>(0, 0) = kin.com_position_process_cov;
    Qd.block<3, 3>(3, 3) = kin.com_linear_velocity_process_cov;
    Qd.block<3, 3>(6, 6) = kin.external_forces_process_cov;

    // Euler Discretization - First order Truncation
    Eigen::Matrix<double, 9, 9> Ad = I_;
    Ad += Ac * nominal_dt_;
    P_ = Ad * P_ * Ad.transpose() + Lc * Qd * Lc.transpose() * nominal_dt_;

    // Propagate the mean estimate, forward euler integration of dynamics f
    Eigen::Matrix<double, 9, 1> f =
        computeContinuousDynamics(state, grf.COP, grf.force, kin.com_angular_momentum);
    predicted_state.com_position_ += f.head<3>() * nominal_dt_;
    predicted_state.com_linear_velocity_ += f.segment<3>(3) * nominal_dt_;
    predicted_state.external_forces_ += f.tail<3>() * nominal_dt_;
    return predicted_state;
}

State CoMEKF::update(State state, KinematicMeasurement kin, GroundReactionForceMeasurement grf,
                     ImuMeasurement imu) {
    State updated_state =
        updateWithKinematics(state, kin.com_position, kin.com_position_cov);
    updated_state =
        updateWithImu(state, kin.com_position, imu.linear_acceleration, imu.angular_velocity,
                      imu.angular_acceleration, grf.COP, grf.force, kin.com_linear_acceleration_cov,
                      kin.com_angular_momentum);
    return updated_state;
}

Eigen::Matrix<double, 9, 1> CoMEKF::computeContinuousDynamics(State state, Eigen::Vector3d COP,
                                                              Eigen::Vector3d fN,
                                                              std::optional<Eigen::Vector3d> Ldot) {
    Eigen::Matrix<double, 9, 1> res = Eigen::Matrix<double, 9, 1>::Zero();
    double den = state.com_position_.z() - COP.z();

    res.segment<3>(0) = state.com_linear_velocity_;
    res(5) = (fN.z() + state.external_forces_.z()) / mass_ - g_;
    if (Ldot.has_value()) {
        res(3) = (state.com_position_.x() - COP.x()) / (mass_ * den) * (fN.z()) +
                 state.external_forces_.x() / mass_ - Ldot.value().y() / (mass_ * den);
        res(4) = (state.com_position_.y() - COP.y()) / (mass_ * den) * (fN.z()) +
                 state.external_forces_.y() / mass_ + Ldot.value().x() / (mass_ * den);
    } else {
        res(3) = (state.com_position_.x() - COP.x()) / (mass_ * den) * (fN(2)) +
                 state.external_forces_.x() / mass_;
        res(4) = (state.com_position_.y() - COP.y()) / (mass_ * den) * (fN(2)) +
                 state.external_forces_.y() / mass_;
    }
    return res;
}

std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>>
CoMEKF::computePredictionJacobians(State state, Eigen::Vector3d COP, Eigen::Vector3d fN,
                                   std::optional<Eigen::Vector3d> Ldot) {
    Eigen::Matrix<double, 9, 9> Ac = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> Lc = Eigen::Matrix<double, 9, 9>::Identity();
    Ac.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    Ac(3, 6) = 1.0 / mass_;
    Ac(4, 7) = 1.0 / mass_;
    Ac(5, 8) = 1.0 / mass_;
    double den = state.com_position_.z() - COP.z();

    Ac(3, 0) = fN.z() / (mass_ * den);
    Ac(4, 1) = fN.z() / (mass_ * den);

    if (Ldot.has_value()) {
        Ac(3, 2) = -((fN.z()) * (state.com_position_.x() - COP.z())) / (mass_ * den * den) +
                   Ldot.value().y() / (mass_ * den * den);
        Ac(4, 2) = -(fN.z()) * (state.com_position_.y() - COP.y()) / (mass_ * den * den) -
                   Ldot.value().x() / (mass_ * den * den);
    } else {
        Ac(3, 2) = -((fN.z()) * (state.com_position_.x() - COP.z())) / (mass_ * den * den);
        Ac(4, 2) = -(fN.z()) * (state.com_position_.y() - COP.y()) / (mass_ * den * den);
    }

    return std::make_tuple(Ac, Lc);
}

State CoMEKF::updateWithImu(State state, Eigen::Vector3d Acc, Eigen::Vector3d Pos,
                            Eigen::Vector3d Gyro, Eigen::Vector3d Gyrodot, Eigen::Vector3d COP,
                            Eigen::Vector3d fN, Eigen::Matrix3d com_linear_acceleration_cov,
                            std::optional<Eigen::Vector3d> Ldot) {
    State updated_state = state;
    // Approximate the CoM Acceleration
    Acc += Gyro.cross(Gyro.cross(Pos)) + Gyrodot.cross(Pos);

    double den = state.com_position_.z() - COP.z();
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    z.z() = Acc(2) - ((fN.z() + state.external_forces_.z()) / mass_ - g_);
    H(0, 0) = (fN.z()) / (mass_ * den);
    H(1, 1) = (fN.z()) / (mass_ * den);
    H(0, 6) = 1.000 / mass_;
    H(1, 7) = 1.000 / mass_;
    H(2, 8) = 1.000 / mass_;

    if (Ldot.has_value()) {
        z.x() = Acc(0) - ((state.com_position_.x() - COP.x()) / (mass_ * den) * (fN.z()) +
                          state.external_forces_.x() / mass_ - Ldot.value().y() / (mass_ * den));
        z.y() = Acc(1) - ((state.com_position_.y() - COP.y()) / (mass_ * den) * (fN.z()) +
                          state.external_forces_.y() / mass_ + Ldot.value().x() / (mass_ * den));

        H(0, 2) = -((fN.z()) * (state.com_position_.x() - COP.x())) / (mass_ * den * den) +
                  Ldot.value().y() / (mass_ * den * den);

        H(1, 2) = -(fN.z()) * (state.com_position_.y() - COP.y()) / (mass_ * den * den) -
                  Ldot.value().x() / (mass_ * den * den);
    } else {
        z.x() = Acc(0) - ((state.com_position_.x() - COP.x()) / (mass_ * den) * (fN.z()) +
                          state.external_forces_.x() / mass_);
        z.y() = Acc(1) - ((state.com_position_.y() - COP.y()) / (mass_ * den) * (fN.z()) +
                          state.external_forces_.y() / mass_);

        H(0, 2) = -((fN.z()) * (state.com_position_.x() - COP.x())) / (mass_ * den * den);

        H(1, 2) = -(fN.z()) * (state.com_position_.y() - COP.y()) / (mass_ * den * den);
    }

    Eigen::Matrix3d R = com_linear_acceleration_cov;
    Eigen::Matrix3d s = R + H * P_ * H.transpose();
    Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * s.inverse();
    Eigen::Matrix<double, 9, 1> dx = K * z;
    updated_state.com_position_ += dx.head<3>();
    updated_state.com_linear_velocity_ += dx.segment<3>(3);
    updated_state.external_forces_ += dx.tail<3>();
    P_ = (I_ - K * H) * P_ * (I_ - K * H).transpose() + K * R * K.transpose();
    return updated_state;
}

State CoMEKF::updateWithKinematics(State state, Eigen::Vector3d com_position, Eigen::Matrix3d com_position_cov)
{
    State updated_state = state;
    Eigen::Vector3d z = com_position - state.com_position_;
    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

    Eigen::Matrix3d R = com_position_cov;
    Eigen::Matrix3d s = R + H * P_ * H.transpose();
    Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * s.inverse();
    Eigen::Matrix<double, 9, 1> dx = K * z;
    updated_state.com_position_ += dx.head<3>();
    updated_state.com_linear_velocity_ += dx.segment<3>(3);
    updated_state.external_forces_ += dx.tail<3>();
    P_ = (I_ - K * H) * P_ * (I_ - K * H).transpose() + K * R * K.transpose();
    return updated_state;
}
