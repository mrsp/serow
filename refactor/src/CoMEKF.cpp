/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "CoMEKF.hpp"

#include <iostream>

namespace serow {

void CoMEKF::init(const State& state, double mass, double rate) {
    I_ = Eigen::Matrix<double, 9, 9>::Identity();
    P_ = Eigen::Matrix<double, 9, 9>::Identity();
    // Initialize state indices
    c_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    v_idx_ = c_idx_ + 3;
    f_idx_ = v_idx_ + 3;
    // Initialize state uncertainty
    P_(c_idx_, c_idx_) = state.getCoMPositionCov();
    P_(v_idx_, v_idx_) = state.getCoMLinearVelocityCov();
    P_(f_idx_, f_idx_) = state.getCoMExternalForcesCov();
    mass_ = mass;
    nominal_dt_ = 1.0 / rate;
    g_ = 9.81;
    std::cout << "Nonlinear CoM Estimator Initialized Successfully" << std::endl;
}

State CoMEKF::predict(const State& state, const KinematicMeasurement& kin,
                      const GroundReactionForceMeasurement& grf) {
    double dt = nominal_dt_;
    if (last_grf_timestamp_.has_value()) {
        dt = grf.timestamp - last_grf_timestamp_.value();
    }
    if (dt < nominal_dt_ / 2) {
        dt = nominal_dt_;
    }
    State predicted_state = state;
    const auto& [Ac, Lc] =
        computePredictionJacobians(state, grf.cop, grf.force, kin.com_angular_momentum_derivative);

    // Discretization
    Eigen::Matrix<double, 9, 9> Qd = Eigen::Matrix<double, 9, 9>::Zero();
    Qd(c_idx_, c_idx_) = kin.com_position_process_cov;
    Qd(v_idx_, v_idx_) = kin.com_linear_velocity_process_cov;
    Qd(f_idx_, f_idx_) = kin.external_forces_process_cov;

    // Euler Discretization - First order Truncation
    Eigen::Matrix<double, 9, 9> Ad = I_;
    Ad += Ac * dt;
    P_ = Ad * P_ * Ad.transpose() + Lc * Qd * Lc.transpose() * dt;

    // Propagate the mean estimate, forward euler integration of dynamics f
    Eigen::Matrix<double, 9, 1> f =
        computeContinuousDynamics(state, grf.cop, grf.force, kin.com_angular_momentum_derivative);
    predicted_state.com_position_ += f(c_idx_) * dt;
    predicted_state.com_linear_velocity_ += f(v_idx_) * dt;
    predicted_state.external_forces_ += f(f_idx_) * dt;
    last_grf_timestamp_ = grf.timestamp;
    return predicted_state;
}

State CoMEKF::updateWithKinematics(const State& state, const KinematicMeasurement& kin) {
    return updateWithCoMPosition(state, kin.com_position, kin.com_position_cov);
}

State CoMEKF::updateWithImu(const State& state, const KinematicMeasurement& kin,
                            const GroundReactionForceMeasurement& grf) {
    return updateWithCoMAcceleration(state, kin.com_linear_acceleration, grf.cop, grf.force,
                                     kin.com_linear_acceleration_cov,
                                     kin.com_angular_momentum_derivative);
}

Eigen::Matrix<double, 9, 1> CoMEKF::computeContinuousDynamics(
    const State& state, const Eigen::Vector3d& cop_position,
    const Eigen::Vector3d& ground_reaction_force,
    const Eigen::Vector3d& com_angular_momentum_derivative) {
    Eigen::Matrix<double, 9, 1> res = Eigen::Matrix<double, 9, 1>::Zero();
    res.segment<3>(0) = state.com_linear_velocity_;
    double den = state.com_position_.z() - cop_position.z();
    res(3) =
        (state.com_position_.x() - cop_position.x()) / (mass_ * den) * ground_reaction_force.z() +
        state.external_forces_.x() / mass_ - com_angular_momentum_derivative.y() / (mass_ * den);
    res(4) =
        (state.com_position_.y() - cop_position.y()) / (mass_ * den) * ground_reaction_force.z() +
        state.external_forces_.y() / mass_ + com_angular_momentum_derivative.x() / (mass_ * den);
    res(5) = (ground_reaction_force.z() + state.external_forces_.z()) / mass_ - g_;

    return res;
}

std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>>
CoMEKF::computePredictionJacobians(const State& state, const Eigen::Vector3d& cop_position,
                                   const Eigen::Vector3d& ground_reaction_force,
                                   const Eigen::Vector3d& com_angular_momentum_derivative) {
    Eigen::Matrix<double, 9, 9> Ac = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> Lc = Eigen::Matrix<double, 9, 9>::Identity();
    double den = state.com_position_.z() - cop_position.z();
    Ac.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    Ac(3, 0) = ground_reaction_force.z() / (mass_ * den);
    Ac(3, 2) = -(ground_reaction_force.z() * (state.com_position_.x() - cop_position.x())) /
                   (mass_ * den * den) +
               com_angular_momentum_derivative.y() / (mass_ * den * den);
    Ac(3, 6) = 1.0 / mass_;
    Ac(4, 1) = ground_reaction_force.z() / (mass_ * den);
    Ac(4, 2) = -ground_reaction_force.z() * (state.com_position_.y() - cop_position.y()) /
                   (mass_ * den * den) -
               com_angular_momentum_derivative.x() / (mass_ * den * den);
    Ac(4, 7) = 1.0 / mass_;
    Ac(5, 8) = 1.0 / mass_;

    return std::make_tuple(Ac, Lc);
}

State CoMEKF::updateWithCoMAcceleration(
    const State& state, const Eigen::Vector3d& com_linear_acceleration,
    const Eigen::Vector3d& cop_position, const Eigen::Vector3d& ground_reaction_force,
    const Eigen::Matrix3d& com_linear_acceleration_cov,
    const Eigen::Vector3d& com_angular_momentum_derivative) {
    State updated_state = state;

    double den = state.com_position_.z() - cop_position.z();
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    z.x() = com_linear_acceleration(0) - ((state.com_position_.x() - cop_position.x()) /
                                              (mass_ * den) * ground_reaction_force.z() +
                                          state.external_forces_.x() / mass_ -
                                          com_angular_momentum_derivative.y() / (mass_ * den));
    z.y() = com_linear_acceleration(1) - ((state.com_position_.y() - cop_position.y()) /
                                              (mass_ * den) * ground_reaction_force.z() +
                                          state.external_forces_.y() / mass_ +
                                          com_angular_momentum_derivative.x() / (mass_ * den));
    z.z() = com_linear_acceleration(2) -
            ((ground_reaction_force.z() + state.external_forces_.z()) / mass_ - g_);

    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H(0, 0) = ground_reaction_force.z() / (mass_ * den);
    H(0, 2) = -(ground_reaction_force.z() * (state.com_position_.x() - cop_position.x())) /
                  (mass_ * den * den) +
              com_angular_momentum_derivative.y() / (mass_ * den * den);
    H(0, 6) = 1.0 / mass_;
    H(1, 1) = ground_reaction_force.z() / (mass_ * den);
    H(1, 2) = -ground_reaction_force.z() * (state.com_position_.y() - cop_position.y()) /
                  (mass_ * den * den) -
              com_angular_momentum_derivative.x() / (mass_ * den * den);
    H(1, 7) = 1.0 / mass_;
    H(2, 8) = 1.0 / mass_;

    const Eigen::Matrix3d& R = com_linear_acceleration_cov;
    const Eigen::Matrix3d s = R + H * P_ * H.transpose();
    const Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * s.inverse();
    const Eigen::Matrix<double, 9, 1> dx = K * z;
    P_ = (I_ - K * H) * P_ * (I_ - K * H).transpose() + K * R * K.transpose();
    updateState(updated_state, dx, P_);
    return updated_state;
}

State CoMEKF::updateWithCoMPosition(const State& state, const Eigen::Vector3d& com_position,
                                    const Eigen::Matrix3d& com_position_cov) {
    State updated_state = state;
    const Eigen::Vector3d z = com_position - state.com_position_;
    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H(c_idx_, c_idx_) = Eigen::Matrix3d::Identity();

    const Eigen::Matrix3d& R = com_position_cov;
    const Eigen::Matrix3d s = R + H * P_ * H.transpose();
    const Eigen::Matrix<double, 9, 3> K = P_ * H.transpose() * s.inverse();
    const Eigen::Matrix<double, 9, 1> dx = K * z;
    P_ = (I_ - K * H) * P_ * (I_ - K * H).transpose() + K * R * K.transpose();
    updateState(updated_state, dx, P_);
    return updated_state;
}

void CoMEKF::updateState(State& state, const Eigen::Matrix<double, 9, 1>& dx,
                         const Eigen::Matrix<double, 9, 9>& P) const {
    state.com_position_ += dx(c_idx_);
    state.com_position_cov_ = P_(c_idx_, c_idx_);
    state.com_linear_velocity_ += dx(v_idx_);
    state.com_linear_velocity_cov_ = P_(v_idx_, v_idx_);
    state.external_forces_ += dx(f_idx_);
    state.external_forces_cov_ = P_(f_idx_, f_idx_);
}

}  // namespace serow
