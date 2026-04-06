/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
 * Serow is free software: you can redistribute it and/or modify it under the terms of the GNU
 * General Public License as published by the Free Software Foundation, version 3.
 *
 * Serow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
 * even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Serow. If not,
 * see <https://www.gnu.org/licenses/>.
 **/
#include "CoMEKF.hpp"
#include "lie.hpp"

#include <iostream>

namespace serow {

void CoMEKF::init(const CentroidalState& state, const double mass, const double g,
                  const double grf_rate, const double kin_rate) {
    I_ = Eigen::Matrix<double, 9, 9>::Identity();
    P_ = Eigen::Matrix<double, 9, 9>::Identity();
    // Initialize state indices
    c_idx_ = Eigen::Array3i::LinSpaced(0, 2);
    v_idx_ = c_idx_ + 3;
    f_idx_ = v_idx_ + 3;
    // Initialize state uncertainty
    P_(c_idx_, c_idx_) = state.com_position_cov;
    P_(v_idx_, v_idx_) = state.com_linear_velocity_cov;
    P_(f_idx_, f_idx_) = state.external_forces_cov;
    mass_ = mass;
    nominal_grf_dt_ = 1.0 / grf_rate;
    nominal_kin_dt_ = 1.0 / kin_rate;
    g_ = g;
    last_grf_timestamp_.reset();
    last_com_update_timestamp_.reset();
    last_acc_update_timestamp_.reset();
}

void CoMEKF::setState(const CentroidalState& state) {
    P_ = I_;
    P_(c_idx_, c_idx_) = state.com_position_cov;
    P_(v_idx_, v_idx_) = state.com_linear_velocity_cov;
    P_(f_idx_, f_idx_) = state.external_forces_cov;
    last_grf_timestamp_ = state.timestamp;
}

void CoMEKF::predict(CentroidalState& state, const KinematicMeasurement& kin,
                     const GroundReactionForceMeasurement& grf) {
    double dt = nominal_grf_dt_;
    if (last_grf_timestamp_.has_value()) {
        dt = grf.timestamp - last_grf_timestamp_.value();
    }
    last_grf_timestamp_ = grf.timestamp;

    if (dt < 0.0) {
        return;
    }

    const auto& [Ac, Lc] =
        computePredictionJacobians(state, grf.cop, grf.force, state.angular_momentum_derivative);

    // Discretization
    Eigen::Matrix<double, 9, 9> Qd = Eigen::Matrix<double, 9, 9>::Zero();
    Qd(c_idx_, c_idx_) = kin.com_position_process_cov;
    Qd(v_idx_, v_idx_) = kin.com_linear_velocity_process_cov;
    Qd(f_idx_, f_idx_) = kin.external_forces_process_cov;

    // Euler Discretization - First order Truncation
    const Eigen::Matrix<double, 9, 9> Ad = I_ + Ac * dt;
    Eigen::Matrix<double, 9, 9> P_new;
    P_new.noalias() = Ad * P_ * Ad.transpose();
    P_new += Lc * Qd * Lc.transpose() * dt;
    P_ = P_new;

    // Propagate the mean estimate, forward euler integration of dynamics f
    const Eigen::Matrix<double, 9, 1> f =
        computeContinuousDynamics(state, grf.cop, grf.force, state.angular_momentum_derivative);
    state.com_position += f(c_idx_) * dt;
    state.com_linear_velocity += f(v_idx_) * dt;
    state.external_forces += f(f_idx_) * dt;
    state.com_position_cov = P_(c_idx_, c_idx_);
    state.com_linear_velocity_cov = P_(v_idx_, v_idx_);
    state.external_forces_cov = P_(f_idx_, f_idx_);
    state.timestamp = grf.timestamp;
}

void CoMEKF::updateWithKinematics(CentroidalState& state, const BaseState& base_state,
                                  const KinematicMeasurement& kin) {
    const Eigen::Matrix3d Rwb = base_state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d& com_position = base_state.base_position + Rwb * kin.com_position;
    const Eigen::Matrix3d& com_position_cov = Rwb * kin.com_position_cov * Rwb.transpose();
    updateWithCoMPosition(state, com_position, com_position_cov, kin.timestamp);
    state.timestamp = kin.timestamp;
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> CoMEKF::computeComLinearAccelerationMeasurement(
    const BaseState& base_state, const KinematicMeasurement& kin) {
    const Eigen::Matrix3d Rwb = base_state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d& w = base_state.base_angular_velocity;
    const Eigen::Vector3d& alpha = base_state.base_angular_acceleration;
    const Eigen::Vector3d& a0 = base_state.base_linear_acceleration;
    const Eigen::Vector3d r = Rwb * kin.com_position;
    const Eigen::Vector3d wxr = w.cross(r);
    const Eigen::Vector3d com_linear_acceleration = a0 + w.cross(wxr) + alpha.cross(r);

    const Eigen::Matrix3d Sigma_r = Rwb * kin.com_position_cov * Rwb.transpose();
    const Eigen::Matrix3d W = lie::so3::wedge(w);
    const Eigen::Matrix3d A = lie::so3::wedge(alpha);
    const Eigen::Matrix3d r_hat = lie::so3::wedge(r);
    const Eigen::Matrix3d J_r = W * W + A;
    const Eigen::Matrix3d J_w = -lie::so3::wedge(wxr) - W * r_hat;

    Eigen::Matrix3d Sigma = base_state.base_linear_acceleration_cov;
    Sigma.noalias() += J_r * Sigma_r * J_r.transpose();
    Sigma.noalias() += J_w * base_state.base_angular_velocity_cov * J_w.transpose();
    Sigma.noalias() += r_hat * base_state.base_angular_acceleration_cov * r_hat.transpose();
    Sigma = 0.5 * (Sigma + Sigma.transpose());
    return {com_linear_acceleration, Sigma};
}

void CoMEKF::updateWithImu(CentroidalState& state, const BaseState& base_state,
                           const KinematicMeasurement& kin,
                           const GroundReactionForceMeasurement& grf) {
    const std::pair<Eigen::Vector3d, Eigen::Matrix3d> com_linear_acceleration_measurement =
        computeComLinearAccelerationMeasurement(base_state, kin);

    updateWithCoMAcceleration(state, com_linear_acceleration_measurement.first,
                              com_linear_acceleration_measurement.second, grf.cop, grf.force,
                              state.angular_momentum_derivative, grf.timestamp);

    state.com_linear_acceleration = com_linear_acceleration_measurement.first;
    state.com_linear_acceleration_cov = com_linear_acceleration_measurement.second;
    state.timestamp = kin.timestamp;
}

Eigen::Matrix<double, 9, 1> CoMEKF::computeContinuousDynamics(
    const CentroidalState& state, const Eigen::Vector3d& cop_position,
    const Eigen::Vector3d& ground_reaction_force,
    const Eigen::Vector3d& com_angular_momentum_derivative) {
    Eigen::Matrix<double, 9, 1> res = Eigen::Matrix<double, 9, 1>::Zero();
    res.segment<3>(0) = state.com_linear_velocity;
    double den = state.com_position.z() - cop_position.z();
    den = std::max(den, 1e-6);

    res(3) =
        (state.com_position.x() - cop_position.x()) / (mass_ * den) * ground_reaction_force.z() +
        state.external_forces.x() / mass_ - com_angular_momentum_derivative.y() / (mass_ * den);
    res(4) =
        (state.com_position.y() - cop_position.y()) / (mass_ * den) * ground_reaction_force.z() +
        state.external_forces.y() / mass_ + com_angular_momentum_derivative.x() / (mass_ * den);
    res(5) = (ground_reaction_force.z() + state.external_forces.z()) / mass_ - g_;

    return res;
}

std::tuple<Eigen::Matrix<double, 9, 9>, Eigen::Matrix<double, 9, 9>>
CoMEKF::computePredictionJacobians(const CentroidalState& state,
                                   const Eigen::Vector3d& cop_position,
                                   const Eigen::Vector3d& ground_reaction_force,
                                   const Eigen::Vector3d& com_angular_momentum_derivative) {
    Eigen::Matrix<double, 9, 9> Ac = Eigen::Matrix<double, 9, 9>::Zero();
    Eigen::Matrix<double, 9, 9> Lc = Eigen::Matrix<double, 9, 9>::Identity();
    double den = state.com_position.z() - cop_position.z();
    den = std::max(den, 1e-6);

    Ac.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    Ac(3, 0) = ground_reaction_force.z() / (mass_ * den);
    Ac(3, 2) = -(ground_reaction_force.z() * (state.com_position.x() - cop_position.x())) /
            (mass_ * den * den) +
        com_angular_momentum_derivative.y() / (mass_ * den * den);
    Ac(3, 6) = 1.0 / mass_;
    Ac(4, 1) = ground_reaction_force.z() / (mass_ * den);
    Ac(4, 2) = -ground_reaction_force.z() * (state.com_position.y() - cop_position.y()) /
            (mass_ * den * den) -
        com_angular_momentum_derivative.x() / (mass_ * den * den);
    Ac(4, 7) = 1.0 / mass_;
    Ac(5, 8) = 1.0 / mass_;

    return std::make_tuple(Ac, Lc);
}

void CoMEKF::updateWithCoMAcceleration(CentroidalState& state,
                                       const Eigen::Vector3d& com_linear_acceleration,
                                       const Eigen::Matrix3d& com_linear_acceleration_cov,
                                       const Eigen::Vector3d& cop_position,
                                       const Eigen::Vector3d& ground_reaction_force,
                                       const Eigen::Vector3d& com_angular_momentum_derivative,
                                       const double timestamp) {
    double dt = nominal_grf_dt_;
    if (last_acc_update_timestamp_.has_value()) {
        dt = timestamp - last_acc_update_timestamp_.value();
    }
    last_acc_update_timestamp_ = timestamp;

    if (dt < 0.0) {
        return;
    }

    double den = state.com_position.z() - cop_position.z();
    den = std::max(den, 1e-6);

    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    z.x() = com_linear_acceleration(0) -
        ((state.com_position.x() - cop_position.x()) / (mass_ * den) * ground_reaction_force.z() +
         state.external_forces.x() / mass_ - com_angular_momentum_derivative.y() / (mass_ * den));
    z.y() = com_linear_acceleration(1) -
        ((state.com_position.y() - cop_position.y()) / (mass_ * den) * ground_reaction_force.z() +
         state.external_forces.y() / mass_ + com_angular_momentum_derivative.x() / (mass_ * den));
    z.z() = com_linear_acceleration(2) -
        ((ground_reaction_force.z() + state.external_forces.z()) / mass_ - g_);

    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H(0, 0) = ground_reaction_force.z() / (mass_ * den);
    H(0, 2) = -(ground_reaction_force.z() * (state.com_position.x() - cop_position.x())) /
            (mass_ * den * den) +
        com_angular_momentum_derivative.y() / (mass_ * den * den);
    H(0, 6) = 1.0 / mass_;
    H(1, 1) = ground_reaction_force.z() / (mass_ * den);
    H(1, 2) = -ground_reaction_force.z() * (state.com_position.y() - cop_position.y()) /
            (mass_ * den * den) -
        com_angular_momentum_derivative.x() / (mass_ * den * den);
    H(1, 7) = 1.0 / mass_;
    H(2, 8) = 1.0 / mass_;

    const Eigen::Matrix3d R = com_linear_acceleration_cov / dt;
    const Eigen::Matrix3d s = R + H * P_ * H.transpose();
    const Eigen::Matrix<double, 9, 3> K =
        s.ldlt().solve((P_ * H.transpose()).transpose()).transpose();
    const Eigen::Matrix<double, 9, 1> dx = K * z;
    const Eigen::Matrix<double, 9, 9> IKH = I_ - K * H;
    Eigen::Matrix<double, 9, 9> P_new;
    P_new.noalias() = IKH * P_ * IKH.transpose();
    P_new += K * R * K.transpose();
    P_ = P_new;
    updateState(state, dx, P_);
}

void CoMEKF::updateWithCoMPosition(CentroidalState& state, const Eigen::Vector3d& com_position,
                                   const Eigen::Matrix3d& com_position_cov,
                                   const double timestamp) {
    double dt = nominal_kin_dt_;
    if (last_com_update_timestamp_.has_value()) {
        dt = timestamp - last_com_update_timestamp_.value();
    }
    last_com_update_timestamp_ = timestamp;

    if (dt < 0.0) {
        return;
    }

    const Eigen::Vector3d z = com_position - state.com_position;
    Eigen::Matrix<double, 3, 9> H = Eigen::Matrix<double, 3, 9>::Zero();
    H(c_idx_, c_idx_) = Eigen::Matrix3d::Identity();

    const Eigen::Matrix3d R = com_position_cov / dt;
    const Eigen::Matrix3d s = R + H * P_ * H.transpose();
    const Eigen::Matrix<double, 9, 3> K =
        s.ldlt().solve((P_ * H.transpose()).transpose()).transpose();
    const Eigen::Matrix<double, 9, 1> dx = K * z;
    const Eigen::Matrix<double, 9, 9> IKH = I_ - K * H;
    Eigen::Matrix<double, 9, 9> P_new;
    P_new.noalias() = IKH * P_ * IKH.transpose();
    P_new += K * R * K.transpose();
    P_ = P_new;
    updateState(state, dx, P_);
}

void CoMEKF::updateState(CentroidalState& state, const Eigen::Matrix<double, 9, 1>& dx,
                         const Eigen::Matrix<double, 9, 9>& P) const {
    state.com_position += dx(c_idx_);
    state.com_position_cov = P(c_idx_, c_idx_);
    state.com_linear_velocity += dx(v_idx_);
    state.com_linear_velocity_cov = P(v_idx_, v_idx_);
    state.external_forces += dx(f_idx_);
    state.external_forces_cov = P(f_idx_, f_idx_);
}

}  // namespace serow
