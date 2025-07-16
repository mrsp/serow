/**
 * Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
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
#include "BaseEKF.hpp"

#include "lie.hpp"

namespace serow {

void BaseEKF::init(const BaseState& state, double g, double imu_rate, bool outlier_detection) {
    g_ = Eigen::Vector3d(0.0, 0.0, -g);
    outlier_detection_ = outlier_detection;
    num_states_ = 15;
    num_inputs_ = 12;
    nominal_dt_ = 1.0 / imu_rate;
    I_.setIdentity(num_states_, num_states_);

    // Initialize state indices
    v_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    r_idx_ = v_idx_ + 3;
    p_idx_ = r_idx_ + 3;
    bg_idx_ = p_idx_ + 3;
    ba_idx_ = bg_idx_ + 3;

    ng_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    na_idx_ = ng_idx_ + 3;
    nbg_idx_ = na_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;

    // Set the initial state
    setState(state);

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), gyro_bias (6), acc_bias (9)
    Lc_.setZero(num_states_, num_inputs_);
    Lc_(v_idx_, na_idx_) = -Eigen::Matrix3d::Identity();
    Lc_(r_idx_, ng_idx_) = -Eigen::Matrix3d::Identity();
    Lc_(bg_idx_, nbg_idx_) = Eigen::Matrix3d::Identity();
    Lc_(ba_idx_, nba_idx_) = Eigen::Matrix3d::Identity();

    std::cout << "Base EKF Initialized Successfully" << std::endl;
}

void BaseEKF::setState(const BaseState& state) {
    // Initialize the error state covariance
    P_ = I_;
    P_(v_idx_, v_idx_) = state.base_linear_velocity_cov;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;
    last_imu_timestamp_ = state.timestamp;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> BaseEKF::computePredictionJacobians(
    const BaseState& state, Eigen::Vector3d angular_velocity) {
    angular_velocity -= state.imu_angular_velocity_bias;
    const Eigen::Vector3d& v = state.base_linear_velocity;
    const Eigen::Matrix3d& R = state.base_orientation.toRotationMatrix();

    Eigen::MatrixXd Ac, Lc;
    Lc = Lc_;
    Lc(v_idx_, ng_idx_) = -lie::so3::wedge(v);

    Ac.setZero(num_states_, num_states_);
    Ac(v_idx_, v_idx_) = -lie::so3::wedge(angular_velocity);
    Ac(v_idx_, r_idx_).noalias() = lie::so3::wedge(R.transpose() * g_);
    Ac(v_idx_, bg_idx_) = -lie::so3::wedge(v);
    Ac(v_idx_, ba_idx_) = -Eigen::Matrix3d::Identity();
    Ac(r_idx_, r_idx_) = -lie::so3::wedge(angular_velocity);
    Ac(r_idx_, bg_idx_) = -Eigen::Matrix3d::Identity();
    Ac(p_idx_, v_idx_) = R;
    Ac(p_idx_, r_idx_).noalias() = -R * lie::so3::wedge(v);

    return std::make_tuple(Ac, Lc);
}

void BaseEKF::predict(BaseState& state, const ImuMeasurement& imu) {
    double dt = nominal_dt_;
    if (last_imu_timestamp_.has_value()) {
        dt = imu.timestamp - last_imu_timestamp_.value();
    }
    if (dt < nominal_dt_ / 2) {
        dt = nominal_dt_;
    }
    // Compute the state and input-state Jacobians
    const auto& [Ac, Lc] = computePredictionJacobians(state, imu.angular_velocity);
    // Euler Discretization - First order Truncation
    const Eigen::MatrixXd Ad = I_ + Ac * dt;

    Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(num_inputs_, num_inputs_);
    // Covariance Q with full state + biases
    Qc(ng_idx_, ng_idx_) = imu.angular_velocity_cov;
    Qc(na_idx_, na_idx_) = imu.linear_acceleration_cov;
    Qc(nbg_idx_, nbg_idx_) = imu.angular_velocity_bias_cov;
    Qc(nba_idx_, nba_idx_) = imu.linear_acceleration_bias_cov;

    // Predict the state error covariance
    const Eigen::MatrixXd Qd = Ad * Lc * Qc * Lc.transpose() * Ad.transpose() * dt;
    P_ = Ad * P_ * Ad.transpose() + Qd;

    // Predict the state
    computeDiscreteDynamics(state, dt, imu.angular_velocity, imu.linear_acceleration);
    last_imu_timestamp_ = imu.timestamp;
}

void BaseEKF::computeDiscreteDynamics(BaseState& state, double dt, Eigen::Vector3d angular_velocity,
                                      Eigen::Vector3d linear_acceleration) {
    angular_velocity -= state.imu_angular_velocity_bias;
    linear_acceleration -= state.imu_linear_acceleration_bias;

    // Nonlinear Process Model
    const Eigen::Vector3d v = state.base_linear_velocity;
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d r = state.base_position;

    // Linear velocity
    state.base_linear_velocity.noalias() =
        (v.cross(angular_velocity) + R.transpose() * g_ + linear_acceleration) * dt + v;

    // Position
    state.base_position.noalias() = (R * v) * dt + r;

    // Orientation
    state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(R, angular_velocity * dt)).normalized();
}

void BaseEKF::updateWithOdometry(BaseState& state, const Eigen::Vector3d& base_position,
                                 const Eigen::Quaterniond& base_orientation,
                                 const Eigen::Matrix3d& base_position_cov,
                                 const Eigen::Matrix3d& base_orientation_cov) {
    Eigen::MatrixXd H;
    H.setZero(6, num_states_);
    Eigen::MatrixXd R = Eigen::Matrix<double, 6, 6>::Zero();

    // Construct the innovation vector z
    const Eigen::Vector3d zp = base_position - state.base_position;
    const Eigen::Vector3d zq = lie::so3::minus(base_orientation, state.base_orientation);
    Eigen::VectorXd z;
    z.setZero(6);
    z.head(3) = zp;
    z.tail(3) = zq;

    // Construct the linearized measurement matrix H
    H.block(0, p_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();
    H.block(3, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    // Construct the measurement noise matrix R
    R.topLeftCorner<3, 3>() = base_position_cov;
    R.bottomRightCorner<3, 3>() = base_orientation_cov;

    const Eigen::Matrix<double, 6, 6> s = R + H * P_ * H.transpose();
    const Eigen::MatrixXd K = P_ * H.transpose() * s.inverse();
    const Eigen::VectorXd dx = K * z;

    P_ = (I_ - K * H) * P_;
    updateState(state, dx, P_);
}

void BaseEKF::updateWithTwist(BaseState& state, const Eigen::Vector3d& base_linear_velocity,
                              const Eigen::Matrix3d& base_linear_velocity_cov,
                              const Eigen::Quaterniond& base_orientation,
                              const Eigen::Matrix3d& base_orientation_cov) {
    const Eigen::Matrix3d R_world_to_base = state.base_orientation.toRotationMatrix();
    // Construct the linearized measurement matrix H
    Eigen::MatrixXd H;
    H.setZero(6, num_states_);
    H.block(0, v_idx_[0], 3, 3) = R_world_to_base;
    H.block(0, r_idx_[0], 3, 3).noalias() =
        -R_world_to_base * lie::so3::wedge(state.base_linear_velocity);
    H.block(3, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    // Construct the innovation vector z
    Eigen::VectorXd z = Eigen::Matrix<double, 6, 1>::Zero();
    z.head(3) = base_linear_velocity - R_world_to_base * state.base_linear_velocity;
    z.tail(3) = lie::so3::minus(base_orientation, state.base_orientation);

    // Construct the measurement noise matrix R
    Eigen::MatrixXd R = Eigen::Matrix<double, 6, 6>::Zero();
    R.topLeftCorner<3, 3>() = base_linear_velocity_cov;
    R.bottomRightCorner<3, 3>() = base_orientation_cov;

    const Eigen::Matrix<double, 6, 6> s = R + H * P_ * H.transpose();
    const Eigen::MatrixXd K = P_ * H.transpose() * s.inverse();
    const Eigen::VectorXd dx = K * z;

    P_ = (I_ - K * H) * P_;
    updateState(state, dx, P_);
}

BaseState BaseEKF::updateStateCopy(const BaseState& state, const Eigen::VectorXd& dx,
                                   const Eigen::MatrixXd& P) const {
    BaseState updated_state = state;
    updated_state.base_position += dx(p_idx_);
    updated_state.base_position_cov = P(p_idx_, p_idx_);
    updated_state.base_linear_velocity += dx(v_idx_);
    updated_state.base_linear_velocity_cov = P(v_idx_, v_idx_);
    updated_state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
            .normalized();
    updated_state.base_orientation_cov = P(r_idx_, r_idx_);
    updated_state.imu_angular_velocity_bias += dx(bg_idx_);
    updated_state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    updated_state.imu_linear_acceleration_bias += dx(ba_idx_);
    updated_state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);
    return updated_state;
}

void BaseEKF::updateState(BaseState& state, const Eigen::VectorXd& dx,
                          const Eigen::MatrixXd& P) const {
    state.base_position += dx(p_idx_);
    state.base_position_cov = P(p_idx_, p_idx_);
    state.base_linear_velocity += dx(v_idx_);
    state.base_linear_velocity_cov = P(v_idx_, v_idx_);
    state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
            .normalized();
    state.base_orientation_cov = P(r_idx_, r_idx_);
    state.imu_angular_velocity_bias += dx(bg_idx_);
    state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    state.imu_linear_acceleration_bias += dx(ba_idx_);
    state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);
}

void BaseEKF::update(BaseState& state, const ImuMeasurement& imu, const KinematicMeasurement& kin,
                     std::optional<OdometryMeasurement> odom) {
    updateWithTwist(state, kin.base_linear_velocity, kin.base_linear_velocity_cov, imu.orientation,
                    imu.orientation_cov);

    if (odom.has_value()) {
        updateWithOdometry(state, odom->base_position, odom->base_orientation,
                           odom->base_position_cov, odom->base_orientation_cov);
    }
}

}  // namespace serow
