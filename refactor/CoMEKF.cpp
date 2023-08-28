#include "CoMEKF.hpp"

#include <iostream>

void CoMEKF::init(double mass, double rate, double I_xx, double I_yy) {
    I_ = Eigen::Matrix<double, 9, 9>::Identity();
    P_ = Eigen::Matrix<double, 9, 9>::Zero();
    P_.block<3, 3>(0, 0) = 1e-6 * Eigen::Matrix<double, 3, 3>::Identity();
    P_.block<3, 3>(3, 3) = 1e-2 * Eigen::Matrix<double, 3, 3>::Identity();
    P_.block<3, 3>(6, 6) = 1e-1 * Eigen::Matrix<double, 3, 3>::Identity();
    std::cout << "Nonlinear CoM Estimator Initialized!" << std::endl;
    mass_ = mass;
    nominal_dt_ = 1.0 / rate;
    I_xx_ = I_xx;
    I_yy_ = I_yy;
    g_ = 9.81;
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

State CoMEKF::predict(State state, KinematicMeasurement kin, GroundReactionForceMeasurement grf) {
    State predicted_state = state;
    const auto& [Ac, Lc] = computePredictionJacobians(state, grf.COP, grf.force, kin.com_angular_momentum);

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

// void CoMEKF::update(Vector3d Acc, Vector3d Pos, Vector3d Gyro, Vector3d Gyrodot){

// 	/* Update Step */
// 	//Compute the CoM Acceleration
// 	Acc += Gyro.cross(Gyro.cross(Pos)) + Gyrodot.cross(Pos);

// 	tmp = x(2)-COP(2);

// 	z.segment<3>(0).noalias() = Pos - x.segment<3>(0);

// 	z(3) = Acc(0) - ( (x(0) - COP(0)) / (m * tmp) * (fN(2) ) + x(6) / m - L(1) / (m * tmp) );
// 	z(4) = Acc(1) - ( (x(1) - COP(1)) / (m * tmp) * (fN(2) ) + x(7) / m + L(0) / (m * tmp) );
// 	z(5) = Acc(2) - ( (fN(2) + x(8)) / m - g );

// 	H(3, 0) = (fN(2) ) / (m * tmp);
// 	H(3, 2) =  -((fN(2) ) * (x(0) - COP(0))) / (m * tmp * tmp) +  L(1) / (m * tmp * tmp);

// 	//H(3, 8) = (x(0) - COP(0)) / (m * tmp);
// 	//H(4, 8) = (x(1) - COP(1)) / (m * tmp);

// 	H(4, 1) = (fN(2)) / (m * tmp);
// 	H(4, 2) = - ( fN(2)) * ( x(1) - COP(1) ) / (m * tmp * tmp) - L(0) / (m * tmp * tmp);

// 	H(3, 6) = 1.000 / m;
// 	H(4, 7) = 1.000 / m;
// 	H(5, 8) = 1.000 / m;

// 	R(0, 0) = com_r * com_r;
// 	R(1, 1) = R(0, 0);
// 	R(2, 2) = R(0, 0);

// 	R(3, 3) = comdd_r * comdd_r;
// 	R(4, 4) = R(3, 3);
// 	R(5, 5) = R(3, 3);
//     //R = R * dt;

//     S = R;
// 	S.noalias() += H * P * H.transpose();
// 	K.noalias() = P * H.transpose() * S.inverse();

// 	x += K * z;
// 	P = (I - K * H) * P * (I - K * H).transpose();
// 	P.noalias() += K * R * K.transpose();
// 	updateVars();
// }
