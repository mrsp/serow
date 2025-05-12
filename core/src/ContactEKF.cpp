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
#include "ContactEKF.hpp"

#include "lie.hpp"

namespace serow {

void ContactEKF::init(const BaseState& state, std::set<std::string> contacts_frame, bool point_feet,
                      double g, double imu_rate, bool outlier_detection) {
    num_leg_end_effectors_ = contacts_frame.size();
    contacts_frame_ = std::move(contacts_frame);
    g_ = Eigen::Vector3d(0.0, 0.0, -g);
    outlier_detection_ = outlier_detection;
    point_feet_ = point_feet;
    contact_dim_ = point_feet_ ? 3 : 6;
    num_states_ = 15 + contact_dim_ * num_leg_end_effectors_;
    num_inputs_ = 12 + contact_dim_ * num_leg_end_effectors_;
    nominal_dt_ = 1.0 / imu_rate;
    I_.setIdentity(num_states_, num_states_);

    // Initialize state indices
    v_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    r_idx_ = v_idx_ + 3;
    p_idx_ = r_idx_ + 3;
    bg_idx_ = p_idx_ + 3;
    ba_idx_ = bg_idx_ + 3;

    Eigen::Array3i pl_idx = ba_idx_;
    for (const auto& contact_frame : contacts_frame_) {
        pl_idx += 3;
        pl_idx_.insert({contact_frame, pl_idx});
    }

    if (!point_feet_) {
        Eigen::Array3i rl_idx = pl_idx;
        for (const auto& contact_frame : contacts_frame_) {
            rl_idx += 3;
            rl_idx_.insert({contact_frame, rl_idx});
        }
    }

    ng_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    na_idx_ = ng_idx_ + 3;
    nbg_idx_ = na_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;

    Eigen::Array3i npl_idx = nba_idx_;
    for (const auto& contact_frame : contacts_frame_) {
        npl_idx += 3;
        npl_idx_.insert({contact_frame, npl_idx});
    }

    if (!point_feet_) {
        Eigen::Array3i nrl_idx = npl_idx;
        for (const auto& contact_frame : contacts_frame_) {
            nrl_idx += 3;
            nrl_idx_.insert({contact_frame, nrl_idx});
        }
    }

    // Set the initial state
    P_ = I_;
    P_(v_idx_, v_idx_) = state.base_linear_velocity_cov;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;

    for (const auto& contact_frame : contacts_frame_) {
        P_(pl_idx_.at(contact_frame), pl_idx_.at(contact_frame)) =
            state.contacts_position_cov.at(contact_frame);
        if (!point_feet_) {
            P_(rl_idx_.at(contact_frame), rl_idx_.at(contact_frame)) =
                state.contacts_orientation_cov.value().at(contact_frame);
        }
    }

    for (const auto& contact_frame : contacts_frame_) {
        position_action_cov_gain_[contact_frame] = 1.0;
        contact_position_action_cov_gain_[contact_frame] = 1.0;
        if (!point_feet_) {
            orientation_action_cov_gain_[contact_frame] = 1.0;
            contact_orientation_action_cov_gain_[contact_frame] = 1.0;
        }
        if (state.contacts_position_cov.count(contact_frame)) {
            P_(pl_idx_.at(contact_frame), pl_idx_.at(contact_frame)) =
                state.contacts_position_cov.at(contact_frame);
        }
        if (!point_feet_ && state.contacts_orientation_cov.has_value() &&
            state.contacts_orientation_cov.value().count(contact_frame)) {
            P_(rl_idx_.at(contact_frame), rl_idx_.at(contact_frame)) =
                state.contacts_orientation_cov.value().at(contact_frame);
        }
    }

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), gyro_bias (6), acc_bias (9), leg end effectors (12 - 12 + contact_dim * N)
    Lc_.setZero(num_states_, num_inputs_);
    Lc_(v_idx_, na_idx_) = -Eigen::Matrix3d::Identity();
    Lc_(r_idx_, ng_idx_) = -Eigen::Matrix3d::Identity();
    Lc_(bg_idx_, nbg_idx_) = Eigen::Matrix3d::Identity();
    Lc_(ba_idx_, nba_idx_) = Eigen::Matrix3d::Identity();

    for (const auto& contact_frame : contacts_frame_) {
        Lc_(pl_idx_.at(contact_frame), npl_idx_.at(contact_frame)) = -Eigen::Matrix3d::Identity();
        if (!point_feet_) {
            Lc_(rl_idx_.at(contact_frame), nrl_idx_.at(contact_frame)) =
                -Eigen::Matrix3d::Identity();
        }
    }

    last_imu_timestamp_.reset();
    std::cout << "Contact EKF Initialized Successfully" << std::endl;
}

void ContactEKF::setState(const BaseState& state) {
    // Set the error state covariance
    P_ = I_;
    P_(v_idx_, v_idx_) = state.base_linear_velocity_cov;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;

    for (const auto& contact_frame : contacts_frame_) {
        P_(pl_idx_.at(contact_frame), pl_idx_.at(contact_frame)) =
            state.contacts_position_cov.at(contact_frame);
        if (!point_feet_) {
            P_(rl_idx_.at(contact_frame), rl_idx_.at(contact_frame)) =
                state.contacts_orientation_cov.value().at(contact_frame);
        }
    }

    for (const auto& contact_frame : contacts_frame_) {
        position_action_cov_gain_[contact_frame] = 1.0;
        contact_position_action_cov_gain_[contact_frame] = 1.0;
        if (!point_feet_) {
            orientation_action_cov_gain_[contact_frame] = 1.0;
            contact_orientation_action_cov_gain_[contact_frame] = 1.0;
        }
        if (state.contacts_position_cov.count(contact_frame)) {
            P_(pl_idx_.at(contact_frame), pl_idx_.at(contact_frame)) =
                state.contacts_position_cov.at(contact_frame);
        }
        if (!point_feet_ && state.contacts_orientation_cov.has_value() &&
            state.contacts_orientation_cov.value().count(contact_frame)) {
            P_(rl_idx_.at(contact_frame), rl_idx_.at(contact_frame)) =
                state.contacts_orientation_cov.value().at(contact_frame);
        }
    }
    last_imu_timestamp_ = state.timestamp;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> ContactEKF::computePredictionJacobians(
    const BaseState& state, Eigen::Vector3d angular_velocity) {
    angular_velocity -= state.imu_angular_velocity_bias;
    const Eigen::Vector3d& v = state.base_linear_velocity;
    const Eigen::Matrix3d& R = state.base_orientation.toRotationMatrix();

    Eigen::MatrixXd Ac(num_states_, num_states_), Lc(num_states_, num_inputs_);
    Lc = Lc_;
    Lc(v_idx_, ng_idx_) = -lie::so3::wedge(v);
    for (const auto& contact_frame : contacts_frame_) {
        Lc_(pl_idx_.at(contact_frame), npl_idx_.at(contact_frame)) = -R;
    }

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

void ContactEKF::predict(BaseState& state, const ImuMeasurement& imu,
                         const KinematicMeasurement& kin) {
    double dt = nominal_dt_;
    if (last_imu_timestamp_.has_value()) {
        dt = imu.timestamp - last_imu_timestamp_.value();
    }
    if (dt < nominal_dt_ / 2) {
        std::cout << "[SEROW/ContactEKF]: Predict step sample time is abnormal " << dt
                  << " while the nominal sample time is " << nominal_dt_ << " setting to nominal"
                  << std::endl;
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

    for (const auto& [cf, cs] : kin.contacts_status) {
        const int contact_status = static_cast<int>(cs);
        Qc(npl_idx_.at(cf), npl_idx_.at(cf)).noalias() =
            kin.position_slip_cov * position_action_cov_gain_.at(cf) +
            (1 - contact_status) * 1e4 * Eigen::Matrix3d::Identity();
        if (!point_feet_) {
            Qc(nrl_idx_.at(cf), nrl_idx_.at(cf)).noalias() =
                kin.orientation_slip_cov * orientation_action_cov_gain_.at(cf) +
                (1 - contact_status) * 1e4 * Eigen::Matrix3d::Identity();
        }
    }

    // Predict the state error covariance
    const Eigen::MatrixXd Qd = Ad * Lc * Qc * Lc.transpose() * Ad.transpose() * dt;
    P_ = Ad * P_ * Ad.transpose() + Qd;

    // Predict the state
    computeDiscreteDynamics(state, dt, imu.angular_velocity, imu.linear_acceleration,
                            kin.contacts_status, kin.contacts_position, kin.contacts_orientation);
    last_imu_timestamp_ = imu.timestamp;
}

void ContactEKF::computeDiscreteDynamics(
    BaseState& state, double dt, Eigen::Vector3d angular_velocity,
    Eigen::Vector3d linear_acceleration, std::optional<std::map<std::string, bool>> contacts_status,
    std::optional<std::map<std::string, Eigen::Vector3d>> contacts_position,
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientations) {
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

    // Predicted contacts positions
    if (contacts_status.has_value() && contacts_position.has_value()) {
        for (const auto& [cf, cs] : contacts_status.value()) {
            if (contacts_position.value().count(cf)) {
                if (!cs) {
                    state.contacts_position[cf].noalias() =
                        (r + R * contacts_position.value().at(cf));
                }
            }
        }
    }

    // Predicted contacts orientations
    if (!point_feet_ && contacts_status.has_value() && contacts_orientations.has_value()) {
        for (const auto& [cf, cs] : contacts_status.value()) {
            if (contacts_orientations.value().count(cf)) {
                if (!cs) {
                    state.contacts_orientation.value().at(cf) =
                        Eigen::Quaterniond(R *
                                           contacts_orientations.value().at(cf).toRotationMatrix())
                            .normalized();
                }
            }
        }
    }
}

void ContactEKF::updateWithContacts(
    BaseState& state, const std::map<std::string, Eigen::Vector3d>& contacts_position,
    std::map<std::string, Eigen::Matrix3d> contacts_position_noise,
    const std::map<std::string, bool>& contacts_status, const Eigen::Matrix3d& position_cov,
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation,
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_noise,
    std::optional<Eigen::Matrix3d> orientation_cov,
    std::shared_ptr<TerrainElevation> terrain_estimator) {
    contact_position_innovation_.clear();
    contact_orientation_innovation_.clear();
    base_position_per_contact_position_update_.clear();
    base_orientation_per_contact_position_update_.clear();
    base_position_per_contact_orientation_update_.clear();
    base_orientation_per_contact_orientation_update_.clear();
    
    // Compute the relative contacts position/orientation measurement noise
    for (const auto& [cf, cp] : contacts_status) {
        const int cs = cp ? 1 : 0;

        // If the terrain estimator is in the loop reduce the effect that kinematics has in the
        // contact height update
        if (terrain_estimator) {
            const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
            contacts_position_noise.at(cf) = R * contacts_position_noise.at(cf) * R.transpose();
            contacts_position_noise.at(cf)(2, 0) = 0.0;
            contacts_position_noise.at(cf)(2, 1) = 0.0;
            contacts_position_noise.at(cf)(2, 2) = 0.01;
            contacts_position_noise.at(cf)(0, 2) = 0.0;
            contacts_position_noise.at(cf)(1, 2) = 0.0;
            contacts_position_noise.at(cf) = R.transpose() * contacts_position_noise.at(cf) * R;
        }

        contacts_position_noise.at(cf) = cs * (contacts_position_noise.at(cf) 
            + position_cov * contact_position_action_cov_gain_.at(cf)) +
            (1 - cs) * Eigen::Matrix3d::Identity() * 1e4;

        if (!point_feet_ && contacts_orientation_noise.has_value() && orientation_cov.has_value()) {
            contacts_orientation_noise.value().at(cf) =
                cs * (contacts_orientation_noise.value().at(cf) 
                + orientation_cov.value() * contact_orientation_action_cov_gain_.at(cf)) +
                (1 - cs) * Eigen::Matrix3d::Identity() * 1e4;
        }
    }

    // Update the state with the relative contacts position
    for (const auto& [cf, cp] : contacts_position) {
        const int num_iter = 5;
        Eigen::MatrixXd H(3, num_states_);
        Eigen::MatrixXd K(num_states_, 3);
        Eigen::Vector3d z;
        Eigen::Matrix3d s;

        // Iterative ESKF update
        for (size_t iter = 0; iter < num_iter; iter++) {
            H.setZero();
            const Eigen::Vector3d x = state.base_orientation.toRotationMatrix().transpose() *
                (state.contacts_position.at(cf) - state.base_position);
            z = cp - x;
            H.block(0, p_idx_[0], 3, 3) = -state.base_orientation.toRotationMatrix().transpose();
            H.block(0, pl_idx_.at(cf)[0], 3, 3) =
                state.base_orientation.toRotationMatrix().transpose();
            H.block(0, r_idx_[0], 3, 3) = lie::so3::wedge(x);

            // Normal ESKF update
            s.noalias() = contacts_position_noise.at(cf) + H * P_ * H.transpose();
            K.noalias() = P_ * H.transpose() * s.inverse();
            const Eigen::VectorXd dx = K * z;
            updateState(state, dx, P_);
            if (dx.norm() < 1e-5) {
                break;
            }
        }

        if (!outlier_detection_) {
            P_ = (I_ - K * H) * P_;
        } else {
            // RESKF update
            contact_outlier_detector.init();
            Eigen::MatrixXd P_i = P_;
            BaseState updated_state_i = state;
            for (size_t i = 0; i < contact_outlier_detector.iters; i++) {
                if (contact_outlier_detector.zeta > contact_outlier_detector.threshold) {
                    const Eigen::Matrix3d R_z =
                        contacts_position_noise.at(cf) / contact_outlier_detector.zeta;
                    s.noalias() = R_z + H * P_ * H.transpose();
                    K.noalias() = P_ * H.transpose() * s.inverse();
                    const Eigen::VectorXd dx = K * z;
                    P_i = (I_ - K * H) * P_;
                    updated_state_i = updateStateCopy(state, dx, P_);

                    // Outlier detection with the relative contact position measurement vector
                    const Eigen::Vector3d x_i =
                        updated_state_i.base_orientation.toRotationMatrix().transpose() *
                        (updated_state_i.contacts_position.at(cf) - updated_state_i.base_position);
                    const Eigen::Matrix3d BetaT = cp * cp.transpose() - 2.0 * cp * x_i.transpose() +
                        x_i * x_i.transpose() + H * P_i * H.transpose();
                    contact_outlier_detector.estimate(BetaT, contacts_position_noise.at(cf));
                } else {
                    // Measurement is an outlier
                    updated_state_i = state;
                    P_i = P_;
                    break;
                }
            }
            P_ = std::move(P_i);
            state = std::move(updated_state_i);
        }
        if (contacts_status.at(cf)) {
            contact_position_innovation_[cf] = {z, s + 1e-6 * Eigen::Matrix3d::Identity()};
            base_position_per_contact_position_update_[cf] = state.base_position;
            base_orientation_per_contact_position_update_[cf] = state.base_orientation;
        }
    }

    // Optionally update the state with the relative contacts orientation
    if (!point_feet_ && contacts_orientation.has_value()) {
        for (const auto& [cf, co] : contacts_orientation.value()) {
            const int num_iter = 5;
            Eigen::MatrixXd H(3, num_states_);
            Eigen::MatrixXd K(num_states_, 3);
            Eigen::Vector3d z;
            Eigen::Matrix3d s;
            // Iterative ESKF update
            for (size_t iter = 0; iter < num_iter; iter++) {
                // Construct the innovation vector z
                const Eigen::Quaterniond x = Eigen::Quaterniond(
                    state.contacts_orientation.value().at(cf).toRotationMatrix().transpose() *
                    state.base_orientation.toRotationMatrix());
                z = lie::so3::minus(co, x);

                // Construct the linearized measurement matrix H
                H.setZero();
                H.block(0, r_idx_[0], 3, 3) = -x.toRotationMatrix();
                H.block(0, rl_idx_.at(cf)[0], 3, 3) = Eigen::Matrix3d::Identity();

                s.noalias() = contacts_position_noise.at(cf) / contact_outlier_detector.zeta +
                    H * P_ * H.transpose();
                K.noalias() = P_ * H.transpose() * s.inverse();
                const Eigen::VectorXd dx = K * z;

                updateState(state, dx, P_);
                if (dx.norm() < 1e-5) {
                    break;
                }
            }
            P_ = (I_ - K * H) * P_;
            if (contacts_status.at(cf)) {
                contact_orientation_innovation_[cf] = {z, s + 1e-6 * Eigen::Matrix3d::Identity()}; 
                base_position_per_contact_orientation_update_[cf] = state.base_position;
                base_orientation_per_contact_orientation_update_[cf] = state.base_orientation;
            }
        }
    }
}

void ContactEKF::updateWithOdometry(BaseState& state, const Eigen::Vector3d& base_position,
                                    const Eigen::Quaterniond& base_orientation,
                                    const Eigen::Matrix3d& base_position_cov,
                                    const Eigen::Matrix3d& base_orientation_cov) {
    Eigen::MatrixXd H(6, num_states_);
    H.setZero();
    Eigen::MatrixXd R = Eigen::Matrix<double, 6, 6>::Zero();

    // Construct the innovation vector z
    const Eigen::Vector3d zp = base_position - state.base_position;
    const Eigen::Vector3d zq = lie::so3::minus(base_orientation, state.base_orientation);
    Eigen::VectorXd z(6);
    z.setZero();
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

void ContactEKF::updateWithTerrain(BaseState& state,
                                   const std::map<std::string, bool>& contacts_status,
                                   std::shared_ptr<TerrainElevation> terrain_estimator) {
    // Construct the innovation vector z, the linearized measurement matrix H, and the measurement
    // noise matrix R
    Eigen::VectorXd z(1);
    Eigen::MatrixXd H(1, num_states_);
    Eigen::MatrixXd R(1, 1);
    z.setZero();
    R.setIdentity();

    for (const auto& [cf, cs] : contacts_status) {
        // Update only when the elevation at the contact points is available and updated in the map
        if (cs) {
            const std::array<float, 2> con_pos_xy = {
                static_cast<float>(state.contacts_position.at(cf).x()),
                static_cast<float>(state.contacts_position.at(cf).y())};
            auto elevation = terrain_estimator->getElevation(con_pos_xy);
            if (elevation.has_value() && elevation.value().updated) {
                // Construct the linearized measurement matrix H
                H.setZero();
                H(0, pl_idx_.at(cf)[2]) = 1.0;

                // Compute innovation
                z(0) = static_cast<double>(elevation.value().height) -
                    state.contacts_position.at(cf).z();

                // Construct the measurement noise matrix R
                R(0, 0) = static_cast<double>(elevation.value().variance);

                const Eigen::MatrixXd s = R + H * P_ * H.transpose();
                const Eigen::MatrixXd K = P_ * H.transpose() * s.inverse();
                const Eigen::VectorXd dx = K * z;
                updateState(state, dx, P_);
                P_ = (I_ - K * H) * P_;
            }
        }
    }
}

BaseState ContactEKF::updateStateCopy(const BaseState& state, const Eigen::VectorXd& dx,
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

    for (const auto& cf : contacts_frame_) {
        updated_state.contacts_position.at(cf) += dx(pl_idx_.at(cf));
        updated_state.contacts_position_cov.at(cf) = P(pl_idx_.at(cf), pl_idx_.at(cf));
    }

    if (!point_feet_) {
        if (state.contacts_orientation.has_value()) {
            for (const auto& cf : contacts_frame_) {
                updated_state.contacts_orientation.value().at(cf) =
                    Eigen::Quaterniond(
                        lie::so3::plus(state.contacts_orientation.value().at(cf).toRotationMatrix(),
                                       dx(rl_idx_.at(cf))))
                        .normalized();
                updated_state.contacts_orientation_cov.value().at(cf) =
                    P(rl_idx_.at(cf), rl_idx_.at(cf));
            }
        } else {
            std::cerr << "Contacts orientations not initialized, skipping in update" << std::endl;
        }
    }
    return updated_state;
}

void ContactEKF::updateState(BaseState& state, const Eigen::VectorXd& dx,
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

    for (const auto& cf : contacts_frame_) {
        state.contacts_position.at(cf) += dx(pl_idx_.at(cf));
        state.contacts_position_cov.at(cf) = P(pl_idx_.at(cf), pl_idx_.at(cf));
    }

    if (!point_feet_) {
        if (state.contacts_orientation.has_value()) {
            for (const auto& cf : contacts_frame_) {
                state.contacts_orientation.value().at(cf) =
                    Eigen::Quaterniond(
                        lie::so3::plus(state.contacts_orientation.value().at(cf).toRotationMatrix(),
                                       dx(rl_idx_.at(cf))))
                        .normalized();
                state.contacts_orientation_cov.value().at(cf) = P(rl_idx_.at(cf), rl_idx_.at(cf));
            }
        } else {
            std::cerr << "Contacts orientations not initialized, skipping in update" << std::endl;
        }
    }
}

void ContactEKF::update(BaseState& state, const KinematicMeasurement& kin,
                        std::optional<OdometryMeasurement> odom,
                        std::shared_ptr<TerrainElevation> terrain_estimator) {
    // Use the predicted state to update the terrain estimator
    if (terrain_estimator) {
        for (const auto& [cf, cp] : kin.contacts_probability) {
            if (cp > 0.15) {
                Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
                T_world_to_base.translation() = state.base_position;
                T_world_to_base.linear() = state.base_orientation.toRotationMatrix();
                const Eigen::Vector3d con_pos_world =
                    T_world_to_base * kin.contacts_position.at(cf);
                const std::array<float, 2> con_pos_xy = {static_cast<float>(con_pos_world.x()),
                                                         static_cast<float>(con_pos_world.y())};
                const float con_pos_z = static_cast<float>(con_pos_world.z());

                // Transform measurement noise to world frame
                const Eigen::Matrix3d con_cov = T_world_to_base.linear() *
                    kin.contacts_position_noise.at(cf) * T_world_to_base.linear().transpose();

                if (!terrain_estimator->update(con_pos_xy, con_pos_z,
                                               static_cast<float>(con_cov(2, 2)))) {
                    std::cout << "Contact for " << cf
                              << " is not inside the terrain elevation map and thus height is not "
                                 "updated "
                              << std::endl;
                }
            }
        }
    }

    // Update the state with the relative to base contacts
    updateWithContacts(state, kin.contacts_position, kin.contacts_position_noise,
                       kin.contacts_status, kin.position_cov, kin.contacts_orientation,
                       kin.contacts_orientation_noise, kin.orientation_cov, terrain_estimator);

    if (odom.has_value()) {
        updateWithOdometry(state, odom->base_position, odom->base_orientation,
                           odom->base_position_cov, odom->base_orientation_cov);
    }

    // Update the state with the absolute terrain height at each contact location and potentially
    // recenter the terrain mapper
    if (terrain_estimator) {
        updateWithTerrain(state, kin.contacts_status, terrain_estimator);
        // Recenter the map
        const std::array<float, 2> base_pos_xy = {static_cast<float>(state.base_position.x()),
                                                  static_cast<float>(state.base_position.y())};
        const std::array<float, 2>& map_origin_xy = terrain_estimator->getMapOrigin();
        if ((abs(base_pos_xy[0] - map_origin_xy[0]) > 0.35) ||
            (abs(base_pos_xy[1] - map_origin_xy[1]) > 0.35)) {
            terrain_estimator->recenter(base_pos_xy);
        }
    }
}

void ContactEKF::setAction(const std::string& cf, const Eigen::VectorXd& action) {
    const size_t num_actions = 1 + 1 * !point_feet_;
    if (action.size() != static_cast<Eigen::Index>(num_actions)) {
        throw std::invalid_argument("Action size must be 1 + 1 * !point_feet_");
    }

    contact_position_action_cov_gain_.at(cf) = action(0);
    if (!point_feet_ && orientation_action_cov_gain_.count(cf) > 0 &&
        contact_orientation_action_cov_gain_.count(cf) > 0) {
        contact_orientation_action_cov_gain_.at(cf) = action(1);
    }
}

bool ContactEKF::getContactPositionInnovation(const std::string& contact_frame,
                                              Eigen::Vector3d& base_position,
                                              Eigen::Quaterniond& base_orientation,
                                              Eigen::Vector3d& innovation,
                                              Eigen::Matrix3d& covariance) const {
    if (contact_position_innovation_.find(contact_frame) != contact_position_innovation_.end()) {
        innovation = contact_position_innovation_.at(contact_frame).first;
        covariance = contact_position_innovation_.at(contact_frame).second;
        base_position = base_position_per_contact_position_update_.at(contact_frame);
        base_orientation = base_orientation_per_contact_position_update_.at(contact_frame);
        return true;
    }
    return false;
}

bool ContactEKF::getContactOrientationInnovation(const std::string& contact_frame,
                                                 Eigen::Vector3d& base_position,
                                                 Eigen::Quaterniond& base_orientation,
                                                 Eigen::Vector3d& innovation,
                                                 Eigen::Matrix3d& covariance) const {
    if (point_feet_) {
        return false;
    }

    if (contact_orientation_innovation_.find(contact_frame) !=
        contact_orientation_innovation_.end()) {
        innovation = contact_orientation_innovation_.at(contact_frame).first;
        covariance = contact_orientation_innovation_.at(contact_frame).second;
        base_position = base_position_per_contact_orientation_update_.at(contact_frame);
        base_orientation = base_orientation_per_contact_orientation_update_.at(contact_frame);
        return true;
    }
    return false;
}

}  // namespace serow
