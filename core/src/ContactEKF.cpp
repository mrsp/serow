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
#include "ContactEKF.hpp"

#include <cmath>

#include "lie.hpp"

namespace serow {

void ContactEKF::init(const BaseState& state, std::set<std::string> contacts_frame, const double g,
                      const double imu_rate, const double kin_rate, const double eps,
                      const bool point_feet, const bool use_imu_orientation, const bool verbose) {
    num_leg_end_effectors_ = contacts_frame.size();
    contacts_frame_ = std::move(contacts_frame);
    point_feet_ = point_feet;
    eps_ = eps;
    g_ = Eigen::Vector3d(0.0, 0.0, -g);
    num_states_ = 15;
    num_inputs_ = 12;
    nominal_imu_dt_ = 1.0 / imu_rate;
    nominal_kin_dt_ = 1.0 / kin_rate;
    I_.setIdentity(num_states_, num_states_);

    // Initialize state indices
    v_idx_ = Eigen::Array3i::LinSpaced(0, 2);
    r_idx_ = v_idx_ + 3;
    p_idx_ = r_idx_ + 3;
    bg_idx_ = p_idx_ + 3;
    ba_idx_ = bg_idx_ + 3;

    // Initialize input indices
    ng_idx_ = Eigen::Array3i::LinSpaced(0, 2);
    na_idx_ = ng_idx_ + 3;
    nbg_idx_ = na_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;

    // Set the initial state
    P_ = I_;
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    P_(v_idx_, v_idx_).noalias() = R.transpose() * state.base_linear_velocity_cov * R;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), gyro_bias (6), acc_bias (9)
    Lc_.setZero(num_states_, num_inputs_);
    Lc_(v_idx_, na_idx_) = -Eigen::Matrix3d::Identity();
    Lc_(r_idx_, ng_idx_) = -Eigen::Matrix3d::Identity();
    Lc_(bg_idx_, nbg_idx_) = Eigen::Matrix3d::Identity();
    Lc_(ba_idx_, nba_idx_) = Eigen::Matrix3d::Identity();
    last_imu_predict_timestamp_.reset();
    last_kin_update_timestamp_.reset();
    last_imu_update_timestamp_.reset();
    last_terrain_update_timestamp_.reset();

    use_imu_orientation_ = use_imu_orientation;
    base_linear_velocity_z_lpf_ =
        std::make_unique<ButterworthLPF>("Base Linear Velocity Z LPF", imu_rate, 10.0, false);

    verbose_ = verbose;
    if (verbose) {
        std::cout << "[SEROW/ContactEKF]: Initialized successfully" << '\n';
    }
}

void ContactEKF::setState(const BaseState& state) {
    // Set the error state covariance
    P_ = I_;
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    P_(v_idx_, v_idx_).noalias() = R.transpose() * state.base_linear_velocity_cov * R;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;
    last_imu_update_timestamp_ = state.timestamp;
    last_kin_update_timestamp_ = state.timestamp;
    last_imu_update_timestamp_ = state.timestamp;
    last_terrain_update_timestamp_ = state.timestamp;
}

std::tuple<Eigen::Matrix<double, 15, 15>, Eigen::Matrix<double, 15, 12>>
ContactEKF::computePredictionJacobians(const BaseState& state, Eigen::Vector3d angular_velocity) {
    angular_velocity -= state.imu_angular_velocity_bias;
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d v = state.base_local_linear_velocity;

    Eigen::Matrix<double, 15, 15> Ac;
    Eigen::Matrix<double, 15, 12> Lc;
    Lc = Lc_;
    Lc(v_idx_, ng_idx_) = -lie::so3::wedge(v);

    Ac.setZero();
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

void ContactEKF::predict(BaseState& state, const ImuMeasurement& imu) {
    double dt = nominal_imu_dt_;
    if (last_imu_predict_timestamp_.has_value()) {
        dt = imu.timestamp - last_imu_predict_timestamp_.value();
    }
    last_imu_predict_timestamp_ = imu.timestamp;

    // Check if the sample time is abnormal
    if (dt < 0.0) {
        std::cout << "[SEROW/ContactEKF]: Predict step sample time is negative " << dt
                  << " while the nominal sample time is " << nominal_imu_dt_
                  << " returning without updating the state" << '\n';
        return;
    }

    // Compute the state and input-state Jacobians
    const auto& [Ac, Lc] = computePredictionJacobians(state, imu.angular_velocity);
    // Euler Discretization - First order Truncation
    const Eigen::Matrix<double, 15, 15> Ad = I_ + Ac * dt;

    Eigen::Matrix<double, 12, 12> Qc = Eigen::Matrix<double, 12, 12>::Zero();
    // Covariance Q with full state + biases
    Qc(ng_idx_, ng_idx_) = imu.angular_velocity_cov;
    Qc(na_idx_, na_idx_) = imu.linear_acceleration_cov;
    Qc(nbg_idx_, nbg_idx_) = imu.angular_velocity_bias_cov;
    Qc(nba_idx_, nba_idx_) = imu.linear_acceleration_bias_cov;

    // Predict the state error covariance
    // First-order discrete-time process noise with Euler integration
    const Eigen::Matrix<double, 15, 15> Qd = Lc * Qc * Lc.transpose() * dt;
    Eigen::Matrix<double, 15, 15> P_new;
    P_new.noalias() = Ad * P_ * Ad.transpose();
    P_new += Qd;
    P_ = P_new;

    // Predict the state
    computeDiscreteDynamics(state, dt, imu.angular_velocity, imu.linear_acceleration);
    state.base_linear_velocity =
        state.base_orientation.toRotationMatrix() * state.base_local_linear_velocity;
}

void ContactEKF::computeDiscreteDynamics(BaseState& state, double dt,
                                         Eigen::Vector3d angular_velocity,
                                         Eigen::Vector3d linear_acceleration) {
    angular_velocity -= state.imu_angular_velocity_bias;
    linear_acceleration -= state.imu_linear_acceleration_bias;

    // Nonlinear Process Model
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d r = state.base_position;
    const Eigen::Vector3d v = state.base_local_linear_velocity;
    const Eigen::Vector3d v_w = R * state.base_local_linear_velocity;

    // Linear velocity
    state.base_local_linear_velocity.noalias() =
        (v.cross(angular_velocity) + R.transpose() * g_ + linear_acceleration) * dt + v;

    // Position
    state.base_position.noalias() = v_w * dt + r;

    // Orientation
    state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(R, angular_velocity * dt)).normalized();
}

void ContactEKF::updateWithOdometry(BaseState& state, const Eigen::Vector3d& base_position,
                                    const Eigen::Quaterniond& base_orientation,
                                    const Eigen::Matrix3d& base_position_cov,
                                    const Eigen::Matrix3d& base_orientation_cov) {
    if (!first_odometry_position_.has_value() || !first_odometry_orientation_.has_value()) {
        first_odometry_position_ = base_position - state.base_position;
        first_odometry_orientation_ = state.base_orientation * base_orientation.inverse();
        return;
    }

    // Remove the initial offset if any
    const Eigen::Vector3d bp = base_position - first_odometry_position_.value();
    const Eigen::Quaterniond bo = first_odometry_orientation_.value() * base_orientation;

    // Construct the linearized measurement matrix H
    Eigen::Matrix<double, 6, 15> H;
    H.setZero();
    H.block(0, p_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();
    H.block(3, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    // RESKF update
    base_position_outlier_detector.init();
    Eigen::Matrix<double, 15, 15> P_i = P_;
    BaseState updated_state_i = state;
    Eigen::Matrix<double, 6, 1> z;
    z.setZero();
    z.head(3) = bp - state.base_position;
    z.tail(3) = lie::so3::minus(bo, state.base_orientation);

    const Eigen::Matrix<double, 15, 6> PH_transpose = P_ * H.transpose();
    Eigen::Matrix<double, 6, 6> R = Eigen::Matrix<double, 6, 6>::Zero();
    R.bottomRightCorner<3, 3>() = base_orientation_cov;
    const Eigen::Matrix3d bb = bp * bp.transpose();
    for (size_t i = 0; i < base_position_outlier_detector.iters; i++) {
        if (base_position_outlier_detector.zeta > base_position_outlier_detector.threshold) {
            const Eigen::Matrix3d R_z = base_position_cov / base_position_outlier_detector.zeta;
            R.topLeftCorner<3, 3>() = R_z;
            const Eigen::Matrix<double, 6, 6> s = R + H * PH_transpose;
            const Eigen::Matrix<double, 15, 6> K =
                s.ldlt().solve(PH_transpose.transpose()).transpose();
            const Eigen::Matrix<double, 15, 1> dx = K * z;
            const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
            P_i = IKH * P_ * IKH.transpose() + K * R * K.transpose();
            updated_state_i = updateStateCopy(state, dx, P_);

            // Outlier detection with the base position measurement vector
            const Eigen::Vector3d& x_i = updated_state_i.base_position;
            const Eigen::Matrix3d BetaT = bb - 2.0 * bp * x_i.transpose() + x_i * x_i.transpose() +
                H.block(0, p_idx_[0], 3, 3) * P_i * H.block(0, p_idx_[0], 3, 3).transpose();
            base_position_outlier_detector.estimate(BetaT, base_position_cov);
        } else {
            // Measurement is an outlier
            updated_state_i = state;
            P_i = P_;
            break;
        }
    }

    P_ = P_i;
    state = updated_state_i;
}

void ContactEKF::updateWithTerrain(
    BaseState& state, const std::map<std::string, Eigen::Vector3d>& contacts_position,
    const std::map<std::string, Eigen::Matrix3d>& contacts_position_cov,
    const std::map<std::string, double>& contacts_probability, const double timestamp,
    std::shared_ptr<TerrainElevation> terrain_estimator) {
    const Eigen::Vector3d p_world_to_base = state.base_position;
    const Eigen::Matrix3d R_world_to_base = state.base_orientation.toRotationMatrix();

    double dt = nominal_kin_dt_;
    if (last_terrain_update_timestamp_.has_value()) {
        dt = timestamp - last_terrain_update_timestamp_.value();
    }
    last_terrain_update_timestamp_ = timestamp;

    if (dt < 0.0) {
        return;
    }

    Eigen::Matrix<double, 1, 15> H;
    Eigen::Matrix<double, 15, 1> PH_transpose;

    for (const auto& [cf, cp] : contacts_probability) {
        if (cp > 0.5) {
            const Eigen::Vector3d con_pos_world =
                R_world_to_base * contacts_position.at(cf) + p_world_to_base;
            const std::array<float, 2> con_pos_xy = {static_cast<float>(con_pos_world.x()),
                                                     static_cast<float>(con_pos_world.y())};
            const auto elevation = terrain_estimator->getElevation(con_pos_xy);
            if (elevation.has_value() && elevation.value().updated) {
                const Eigen::Matrix3d con_cov_world =
                    R_world_to_base * contacts_position_cov.at(cf) * R_world_to_base.transpose();

                // Construct the linearized measurement matrix H
                H.setZero();
                // dh/d_chi = [0, 0, 1] * (-R * skew(p_contact_body))
                H(0, p_idx_[2]) = 1.0;
                // const Eigen::Matrix3d skew_cp = lie::so3::wedge(contacts_position.at(cf));
                // H.block(0, r_idx_[0], 1, 3) = -(R_world_to_base * skew_cp).row(2);

                const double z = static_cast<double>(elevation.value().height) - con_pos_world.z();

                const double N =
                    (static_cast<double>(elevation.value().variance + con_cov_world(2, 2)) + 1e-6) /
                    (cp * dt);

                PH_transpose.noalias() = P_ * H.transpose();
                const double s = N + (H * PH_transpose)(0, 0);
                const Eigen::Matrix<double, 15, 1> K = PH_transpose / s;
                const Eigen::Matrix<double, 15, 1> dx = K * z;
                const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
                Eigen::Matrix<double, 15, 15> P_new;
                P_new.noalias() = IKH * P_ * IKH.transpose();
                P_new += K * N * K.transpose();
                P_ = P_new;
                updateState(state, dx, P_);
            }
        }
    }
}

BaseState ContactEKF::updateStateCopy(const BaseState& state,
                                      const Eigen::Matrix<double, 15, 1>& dx,
                                      const Eigen::Matrix<double, 15, 15>& P) const {
    BaseState updated_state = state;
    updated_state.base_position += dx(p_idx_);
    updated_state.base_position_cov = P(p_idx_, p_idx_);

    updated_state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
            .normalized();
    const Eigen::Matrix3d R = updated_state.base_orientation.toRotationMatrix();
    updated_state.base_local_linear_velocity += dx(v_idx_);
    updated_state.base_linear_velocity_cov.noalias() = R * P(v_idx_, v_idx_) * R.transpose();
    updated_state.base_orientation_cov = P(r_idx_, r_idx_);
    updated_state.imu_angular_velocity_bias += dx(bg_idx_);
    updated_state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    updated_state.imu_linear_acceleration_bias += dx(ba_idx_);
    updated_state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);
    return updated_state;
}

void ContactEKF::updateState(BaseState& state, const Eigen::Matrix<double, 15, 1>& dx,
                             const Eigen::Matrix<double, 15, 15>& P) const {
    state.base_position += dx(p_idx_);
    state.base_position_cov = P(p_idx_, p_idx_);
    state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
            .normalized();
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    state.base_local_linear_velocity += dx(v_idx_);
    state.base_linear_velocity_cov.noalias() = R * P(v_idx_, v_idx_) * R.transpose();
    state.base_orientation_cov = P(r_idx_, r_idx_);
    state.imu_angular_velocity_bias += dx(bg_idx_);
    state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    state.imu_linear_acceleration_bias += dx(ba_idx_);
    state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);
}

void ContactEKF::update(BaseState& state, const ImuMeasurement& imu,
                        const KinematicMeasurement& kin, std::optional<OdometryMeasurement> odom,
                        std::shared_ptr<TerrainElevation> terrain_estimator) {
    // Use the predicted state to update the terrain estimator
    if (terrain_estimator) {
        double dt = nominal_kin_dt_;
        if (last_terrain_update_timestamp_.has_value()) {
            dt = kin.timestamp - last_terrain_update_timestamp_.value();
        }

        if (dt < 0.0) {
            return;
        }

        Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
        T_world_to_base.translation() = state.base_position;
        T_world_to_base.linear() = state.base_orientation.toRotationMatrix();
        for (const auto& [cf, cp] : kin.contacts_probability) {
            if (cp > terrain_estimator->getMinContactProbability()) {
                const Eigen::Vector3d con_pos_world =
                    T_world_to_base * kin.contacts_position.at(cf);
                const std::array<float, 2> con_pos_xy = {static_cast<float>(con_pos_world.x()),
                                                         static_cast<float>(con_pos_world.y())};
                const float con_pos_z = static_cast<float>(con_pos_world.z());

                // Transform measurement noise to world frame
                const Eigen::Matrix3d con_cov = T_world_to_base.linear() *
                    kin.contacts_position_noise.at(cf) * T_world_to_base.linear().transpose();

                std::optional<std::array<float, 3>> normal = std::nullopt;
                if (!point_feet_ && kin.contacts_orientation.has_value() &&
                    kin.contacts_orientation.value().count(cf) > 0) {
                    const Eigen::Matrix3d& R_world_to_base = T_world_to_base.linear();
                    const Eigen::Vector3d n_contact =
                        (R_world_to_base *
                         kin.contacts_orientation.value().at(cf).toRotationMatrix() *
                         Eigen::Vector3d::UnitZ())
                            .normalized();
                    // Compute the angular velocity of the leg end-effector in contact
                    const Eigen::Vector3d omega_contact =
                        R_world_to_base * kin.base_to_foot_angular_velocities.at(cf) +
                        state.base_angular_velocity;
                    // Compute the linear velocity of the leg end-effector in contact
                    const Eigen::Vector3d p_base_to_leg =
                        R_world_to_base * kin.base_to_foot_positions.at(cf);
                    const Eigen::Vector3d v_contact =
                        R_world_to_base * state.base_local_linear_velocity +
                        state.base_angular_velocity.cross(p_base_to_leg) +
                        R_world_to_base * kin.base_to_foot_linear_velocities.at(cf);
                    // Update the terrain elevation with a plane contact if the leg in contact is
                    // stable
                    if (cp > terrain_estimator->getMinStableContactProbability() &&
                        omega_contact.norm() <
                            terrain_estimator->getMinStableFootAngularVelocity() &&
                        v_contact.norm() < terrain_estimator->getMinStableFootLinearVelocity()) {
                        normal = {static_cast<float>(n_contact.x()),
                                  static_cast<float>(n_contact.y()),
                                  static_cast<float>(n_contact.z())};
                    }
                }
                if (!terrain_estimator->update(
                        con_pos_xy, con_pos_z,
                        static_cast<float>((con_cov(2, 2) + 1e-6) / (cp * dt)), normal)) {
                    std::cout << "Contact for " << cf
                              << " is not inside the terrain elevation map and thus height is not "
                                 "updated "
                              << '\n';
                } else {
                    if (kin.is_new_contact.count(cf) > 0 && kin.is_new_contact.at(cf)) {
                        terrain_estimator->addContactPoint(con_pos_xy);
                    }
                }
            }
        }
    }

    // Update the state with the absolute IMU orientation
    if (use_imu_orientation_) {
        updateWithIMUOrientation(state, imu.orientation, imu.orientation_cov, imu.timestamp);
    }

    // Update the state with the absolute base linear velocity computed with leg kinematics only if
    // there is contact
    double den = 0.0;
    for (const auto& [cf, cp] : kin.contacts_probability) {
        den += cp;
    }
    if (den > eps_) {
        updateWithBaseLinearVelocity(state, kin.base_linear_velocity, kin.base_linear_velocity_cov,
                                     kin.timestamp);
    }

    if (odom.has_value()) {
        updateWithOdometry(state, odom->base_position, odom->base_orientation,
                           odom->base_position_cov, odom->base_orientation_cov);
    }

    // Update the state with the absolute terrain height at each contact location and
    // potentially recenter the terrain mapper
    if (terrain_estimator) {
        updateWithTerrain(state, kin.contacts_position, kin.contacts_position_noise,
                          kin.contacts_probability, kin.timestamp, terrain_estimator);
        // Recenter the map
        const std::array<float, 2> base_pos_xy = {static_cast<float>(state.base_position.x()),
                                                  static_cast<float>(state.base_position.y())};
        const std::array<float, 2>& map_origin_xy = terrain_estimator->getMapOrigin();
        if ((std::fabs(base_pos_xy[0] - map_origin_xy[0]) >
             terrain_estimator->getMaxRecenterDistance()) ||
            (std::fabs(base_pos_xy[1] - map_origin_xy[1]) >
             terrain_estimator->getMaxRecenterDistance())) {
            terrain_estimator->recenter(base_pos_xy);
        }
        terrain_estimator->interpolateContactPoints();
    }

    Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
    T_world_to_base.translation() = state.base_position;
    T_world_to_base.linear() = state.base_orientation.toRotationMatrix();
    state.base_linear_velocity = T_world_to_base.linear() * state.base_local_linear_velocity;

    if (terrain_estimator) {
        state.base_linear_velocity.z() =
            base_linear_velocity_z_lpf_->filter(state.base_linear_velocity.z());
    }

    for (const auto& cf : contacts_frame_) {
        state.contacts_position.at(cf) = T_world_to_base * kin.contacts_position.at(cf);
        if (state.contacts_orientation.has_value() &&
            state.contacts_orientation.value().count(cf) > 0) {
            state.contacts_orientation.value().at(cf) =
                T_world_to_base.linear() * kin.contacts_orientation.value().at(cf);
        }
    }
}

void ContactEKF::updateWithIMUOrientation(BaseState& state,
                                          const Eigen::Quaterniond& imu_orientation,
                                          const Eigen::Matrix3d& imu_orientation_cov,
                                          const double timestamp) {
    double dt = nominal_imu_dt_;
    if (last_imu_update_timestamp_.has_value()) {
        dt = timestamp - last_imu_update_timestamp_.value();
    }
    last_imu_update_timestamp_ = timestamp;

    if (dt < 0.0) {
        return;
    }

    Eigen::Matrix<double, 3, 15> H;
    H.setZero();
    H.block(0, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    const Eigen::Vector3d z = lie::so3::minus(imu_orientation, state.base_orientation);
    const Eigen::Matrix3d R = imu_orientation_cov / dt;

    const Eigen::Matrix<double, 15, 3> PH_transpose = P_ * H.transpose();
    const Eigen::Matrix3d s = R + H * PH_transpose;
    const Eigen::Matrix<double, 15, 3> K = s.ldlt().solve(PH_transpose.transpose()).transpose();
    const Eigen::Matrix<double, 15, 1> dx = K * z;

    const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
    Eigen::Matrix<double, 15, 15> P_new;
    P_new.noalias() = IKH * P_ * IKH.transpose();
    P_new += K * R * K.transpose();
    P_ = P_new;
    updateState(state, dx, P_);
}

void ContactEKF::updateWithBaseLinearVelocity(BaseState& state,
                                              const Eigen::Vector3d& base_linear_velocity,
                                              const Eigen::Matrix3d& base_linear_velocity_cov,
                                              const double timestamp) {
    double dt = nominal_kin_dt_;
    if (last_kin_update_timestamp_.has_value()) {
        dt = timestamp - last_kin_update_timestamp_.value();
    }
    last_kin_update_timestamp_ = timestamp;

    if (dt < 0.0) {
        return;
    }

    const int num_iter = 5;
    Eigen::Matrix<double, 3, 15> H;
    H.setZero();
    Eigen::Matrix<double, 15, 3> K;
    Eigen::Matrix<double, 15, 3> PH_transpose;
    Eigen::Matrix3d s;
    const Eigen::Matrix3d N = base_linear_velocity_cov / dt;

    // Iterative ESKF update
    for (size_t iter = 0; iter < num_iter; iter++) {
        const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
        const Eigen::Vector3d v = state.base_local_linear_velocity;
        H.block(0, v_idx_[0], 3, 3) = R;
        H.block(0, r_idx_[0], 3, 3) = -R * lie::so3::wedge(v);

        const Eigen::Vector3d z = base_linear_velocity - R * state.base_local_linear_velocity;

        PH_transpose.noalias() = P_ * H.transpose();
        s.noalias() = N + H * PH_transpose;
        K = s.ldlt().solve(PH_transpose.transpose()).transpose();
        const Eigen::Matrix<double, 15, 1> dx = K * z;

        state.base_position += dx(p_idx_);
        state.base_orientation =
            Eigen::Quaterniond(
                lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
                .normalized();
        state.base_local_linear_velocity += dx(v_idx_);
        state.imu_angular_velocity_bias += dx(bg_idx_);
        state.imu_linear_acceleration_bias += dx(ba_idx_);
        if (dx.squaredNorm() < 1e-9) {
            break;
        }
    }

    // Joseph form for numerical stability
    const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
    Eigen::Matrix<double, 15, 15> P_new;
    P_new.noalias() = IKH * P_ * IKH.transpose();
    P_new += K * N * K.transpose();
    P_ = P_new;

    // Update state covariances with the final P_
    state.base_position_cov = P_(p_idx_, p_idx_);
    state.base_orientation_cov = P_(r_idx_, r_idx_);
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    state.base_linear_velocity_cov.noalias() = R * P_(v_idx_, v_idx_) * R.transpose();
    state.imu_angular_velocity_bias_cov = P_(bg_idx_, bg_idx_);
    state.imu_linear_acceleration_bias_cov = P_(ba_idx_, ba_idx_);
}

}  // namespace serow
