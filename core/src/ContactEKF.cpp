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

    // Initialize the error state covariance
    P_ = I_;
    P_(v_idx_, v_idx_) = state.base_linear_velocity_cov;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;

    for (const auto& contact_frame : contacts_frame_) {
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

    std::cout << "Contact EKF Initialized Successfully" << std::endl;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> ContactEKF::computePredictionJacobians(
    const BaseState& state, Eigen::Vector3d angular_velocity) {
    angular_velocity -= state.imu_angular_velocity_bias;
    const Eigen::Vector3d& v = state.base_linear_velocity;
    const Eigen::Matrix3d& R = state.base_orientation.toRotationMatrix();

    Eigen::MatrixXd Ac, Lc;
    Lc = Lc_;
    Lc(v_idx_, ng_idx_).noalias() = -lie::so3::wedge(v);
    for (const auto& contact_frame : contacts_frame_) {
        Lc_(pl_idx_.at(contact_frame), npl_idx_.at(contact_frame)) = -R;
    }

    Ac.setZero(num_states_, num_states_);
    Ac(v_idx_, v_idx_).noalias() = -lie::so3::wedge(angular_velocity);
    Ac(v_idx_, r_idx_).noalias() = lie::so3::wedge(R.transpose() * g_);
    Ac(v_idx_, bg_idx_).noalias() = -lie::so3::wedge(v);
    Ac(v_idx_, ba_idx_).noalias() = -Eigen::Matrix3d::Identity();
    Ac(r_idx_, r_idx_).noalias() = -lie::so3::wedge(angular_velocity);
    Ac(r_idx_, bg_idx_).noalias() = -Eigen::Matrix3d::Identity();
    Ac(p_idx_, v_idx_).noalias() = R;
    Ac(p_idx_, r_idx_).noalias() = -R * lie::so3::wedge(v);

    return std::make_tuple(Ac, Lc);
}

BaseState ContactEKF::predict(const BaseState& state, const ImuMeasurement& imu,
                              const KinematicMeasurement& kin) {
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
    Eigen::MatrixXd Ad = I_;
    Ad += Ac * dt;

    Eigen::MatrixXd Qc = Eigen::MatrixXd::Zero(num_inputs_, num_inputs_);
    // Covariance Q with full state + biases
    Qc(ng_idx_, ng_idx_) = imu.angular_velocity_cov;
    Qc(na_idx_, na_idx_) = imu.linear_acceleration_cov;
    Qc(nbg_idx_, nbg_idx_) = imu.angular_velocity_bias_cov;
    Qc(nba_idx_, nba_idx_) = imu.linear_acceleration_bias_cov;

    for (const auto& [cf, cs] : kin.contacts_status) {
        int contact_status = static_cast<int>(cs);
        Qc(npl_idx_.at(cf), npl_idx_.at(cf)) =
            kin.position_slip_cov + (1 - contact_status) * 1e4 * Eigen::Matrix3d::Identity();
        if (!point_feet_) {
            Qc(nrl_idx_.at(cf), nrl_idx_.at(cf)) =
                kin.orientation_slip_cov + (1 - contact_status) * 1e4 * Eigen::Matrix3d::Identity();
        }
    }

    // Predict the state error covariance
    Eigen::MatrixXd Qd = Ad * Lc * Qc * Lc.transpose() * Ad.transpose() * dt;
    P_ = Ad * P_ * Ad.transpose() + Qd;

    // Predict the state
    const BaseState& predicted_state = computeDiscreteDynamics(
        state, dt, imu.angular_velocity, imu.linear_acceleration, kin.contacts_status,
        kin.contacts_position, kin.contacts_orientation);
    last_imu_timestamp_ = imu.timestamp;
    return predicted_state;
}

BaseState ContactEKF::computeDiscreteDynamics(
    const BaseState& state, double dt, Eigen::Vector3d angular_velocity,
    Eigen::Vector3d linear_acceleration, std::optional<std::map<std::string, bool>> contacts_status,
    std::optional<std::map<std::string, Eigen::Vector3d>> contacts_position,
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientations) {
    BaseState predicted_state = state;
    angular_velocity -= state.imu_angular_velocity_bias;
    linear_acceleration -= state.imu_linear_acceleration_bias;

    // Nonlinear Process Model
    // Linear velocity
    const Eigen::Vector3d& v = state.base_linear_velocity;
    const Eigen::Matrix3d& R = state.base_orientation.toRotationMatrix();
    predicted_state.base_linear_velocity = v.cross(angular_velocity);
    predicted_state.base_linear_velocity += R.transpose() * g_;
    predicted_state.base_linear_velocity += linear_acceleration;
    predicted_state.base_linear_velocity *= dt;
    predicted_state.base_linear_velocity += v;
    // Position
    const Eigen::Vector3d& r = state.base_position;
    predicted_state.base_position = R * v;
    predicted_state.base_position *= dt;
    predicted_state.base_position += r;

    // Biases
    predicted_state.imu_angular_velocity_bias = state.imu_angular_velocity_bias;
    predicted_state.imu_linear_acceleration_bias = state.imu_linear_acceleration_bias;

    // Orientation
    predicted_state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(R, angular_velocity * dt)).normalized();

    // Predicted contacts positions
    if (contacts_status.has_value() && contacts_position.has_value()) {
        for (auto [cf, cs] : contacts_status.value()) {
            if (contacts_position.value().count(cf)) {
                if (!cs) {
                    predicted_state.contacts_position[cf] =
                        (r + R * contacts_position.value().at(cf));
                }
            }
        }
    }

    // Predicted contacts orientations
    if (!point_feet_ && contacts_status.has_value() && contacts_orientations.has_value()) {
        for (auto [cf, cs] : contacts_status.value()) {
            if (contacts_orientations.value().count(cf)) {
                if (!cs) {
                    predicted_state.contacts_orientation.value().at(cf) =
                        Eigen::Quaterniond(R *
                                           contacts_orientations.value().at(cf).toRotationMatrix())
                            .normalized();
                }
            }
        }
    }

    return predicted_state;
}

BaseState ContactEKF::updateWithContacts(
    const BaseState& state, const std::map<std::string, Eigen::Vector3d>& contacts_position,
    std::map<std::string, Eigen::Matrix3d> contacts_position_noise,
    const std::map<std::string, bool>& contacts_status, const Eigen::Matrix3d& position_cov,
    std::optional<std::map<std::string, Eigen::Quaterniond>> contacts_orientation,
    std::optional<std::map<std::string, Eigen::Matrix3d>> contacts_orientation_noise,
    std::optional<Eigen::Matrix3d> orientation_cov,
    std::shared_ptr<NaiveTerrainElevation> terrain_estimator) {
    BaseState updated_state = state;

    // Compute the relative contacts position/orientation measurement noise
    for (const auto& [cf, cp] : contacts_status) {
        const int cs = cp ? 1 : 0;
        if (terrain_estimator) {
            const Eigen::Matrix3d& R = state.base_orientation.toRotationMatrix();
            contacts_position_noise.at(cf) = R * contacts_position_noise.at(cf) * R.transpose();
            contacts_position_noise.at(cf)(2, 0) = 0.0;
            contacts_position_noise.at(cf)(2, 1) = 0.0;
            contacts_position_noise.at(cf)(2, 2) = 0.01;
            contacts_position_noise.at(cf)(0, 2) = 0.0;
            contacts_position_noise.at(cf)(1, 2) = 0.0;
            contacts_position_noise.at(cf) = R.transpose() * contacts_position_noise.at(cf) * R;
        }

        contacts_position_noise.at(cf) = cs * contacts_position_noise.at(cf) +
                                         (1 - cs) * Eigen::Matrix3d::Identity() * 1e4 +
                                         position_cov;

        if (contacts_orientation_noise.has_value() && orientation_cov.has_value()) {
            contacts_orientation_noise.value().at(cf) =
                cs * contacts_orientation_noise.value().at(cf) +
                (1 - cs) * Eigen::Matrix3d::Identity() * 1e4 + orientation_cov.value();
        }
    }

    // Update the state with the relative contacts position
    for (const auto& [cf, cp] : contacts_position) {
        const int num_iter = 5;
        Eigen::MatrixXd H;
        Eigen::MatrixXd K;
        for (size_t i = 0; i < num_iter; i++) {
            H.setZero(3, num_states_);
            const Eigen::Vector3d x =
                updated_state.base_orientation.toRotationMatrix().transpose() *
                (updated_state.contacts_position.at(cf) - updated_state.base_position);
            const Eigen::Vector3d z = cp - x;
            H.block(0, p_idx_[0], 3, 3) =
                -updated_state.base_orientation.toRotationMatrix().transpose();
            H.block(0, pl_idx_.at(cf)[0], 3, 3) =
                updated_state.base_orientation.toRotationMatrix().transpose();
            H.block(0, r_idx_[0], 3, 3) = lie::so3::wedge(x);

            if (outlier_detection_) {
                // RESKF update
                contact_outlier_detector.init();
                Eigen::MatrixXd P_i = P_;
                BaseState updated_state_i = updated_state;
                for (size_t i = 0; i < contact_outlier_detector.iters; i++) {
                    if (contact_outlier_detector.zeta > contact_outlier_detector.threshold) {
                        const Eigen::Matrix3d R_z =
                            contacts_position_noise.at(cf) / contact_outlier_detector.zeta;
                        const Eigen::Matrix3d s = R_z + H * P_ * H.transpose();
                        const Eigen::MatrixXd K = P_ * H.transpose() * s.inverse();
                        const Eigen::VectorXd dx = K * z;
                        P_i = (I_ - K * H) * P_;
                        updated_state_i = updateStateCopy(updated_state, dx, P_);

                        // Outlier detection with the relative contact position measurement vector
                        const Eigen::Vector3d x_i =
                            updated_state_i.base_orientation.toRotationMatrix().transpose() *
                            (updated_state_i.contacts_position.at(cf) -
                             updated_state_i.base_position);
                        Eigen::Matrix3d BetaT = cp * cp.transpose();
                        BetaT.noalias() -= 2.0 * cp * x_i.transpose();
                        BetaT.noalias() += x_i * x_i.transpose();
                        BetaT.noalias() += H * P_i * H.transpose();
                        contact_outlier_detector.estimate(BetaT, contacts_position_noise.at(cf));
                    } else {
                        // Measurement is an outlier
                        updated_state_i = updated_state;
                        P_i = P_;
                        break;
                    }
                }
                P_ = std::move(P_i);
                updated_state = std::move(updated_state_i);
            } else {
                // Normal ESKF update
                const Eigen::Matrix3d s = contacts_position_noise.at(cf) + H * P_ * H.transpose();
                K = P_ * H.transpose() * s.inverse();
                const Eigen::VectorXd dx = K * z;
                updateState(updated_state, dx, P_);
                if (dx.norm() < 1e-6) {
                    break;
                }
            }
        }
        P_ = (I_ - K * H) * P_;
    }

    // Optionally update the state with the relative contacts orientation
    if (!point_feet_ && contacts_orientation.has_value()) {
        for (const auto& [cf, co] : contacts_orientation.value()) {
            Eigen::MatrixXd H;
            H.setZero(3, num_states_);
            const Eigen::Quaterniond x = Eigen::Quaterniond(
                state.contacts_orientation.value().at(cf).toRotationMatrix().transpose() *
                state.base_orientation.toRotationMatrix());

            const Eigen::Vector3d z = lie::so3::minus(co, x);
            H.block(0, r_idx_[0], 3, 3) = -x.toRotationMatrix();
            H.block(0, rl_idx_.at(cf)[0], 3, 3) = Eigen::Matrix3d::Identity();

            const Eigen::Matrix3d s =
                contacts_position_noise.at(cf) / contact_outlier_detector.zeta +
                H * P_ * H.transpose();
            const Eigen::MatrixXd K = P_ * H.transpose() * s.inverse();
            const Eigen::VectorXd dx = K * z;

            P_ = (I_ - K * H) * P_;
            updateState(updated_state, dx, P_);
        }
    }

    return updated_state;
}

BaseState ContactEKF::updateWithOdometry(const BaseState& state,
                                         const Eigen::Vector3d& base_position,
                                         const Eigen::Quaterniond& base_orientation,
                                         const Eigen::Matrix3d& base_position_cov,
                                         const Eigen::Matrix3d& base_orientation_cov) {
    BaseState updated_state = state;

    Eigen::MatrixXd H;
    H.setZero(6, num_states_);
    Eigen::MatrixXd R = Eigen::Matrix<double, 6, 6>::Zero();

    // Construct the innovation vector z
    const Eigen::Vector3d& zp = base_position - state.base_position;
    const Eigen::Vector3d& zq = lie::so3::minus(base_orientation, state.base_orientation);
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
    updateState(updated_state, dx, P_);

    return updated_state;
}

BaseState ContactEKF::updateWithTerrain(const BaseState& state,
                                        const std::map<std::string, bool>& contacts_status,
                                        const NaiveTerrainElevation& terrain_estimator) {
    BaseState updated_state = state;
    // Construct the innovation vector z, the linearized measurement matrix H, and the measurement
    // noise matrix R
    Eigen::VectorXd z;
    z.setZero(1);
    Eigen::MatrixXd H;
    Eigen::MatrixXd R;
    R.setIdentity(1, 1);
    for (const auto& [cf, cs] : contacts_status) {
        if (cs) {
            auto elevation = terrain_estimator.getElevation(
                {static_cast<float>(updated_state.contacts_position.at(cf).x()),
                 static_cast<float>(updated_state.contacts_position.at(cf).y())});
            if (elevation.has_value() && elevation.value().updated) {
                H.setZero(1, num_states_);
                H(0, pl_idx_.at(cf)[2]) = 1.0;
                std::cout << "map height at [x, y]: " << updated_state.contacts_position.at(cf).x()
                          << " " << updated_state.contacts_position.at(cf).x()
                          << " is: " << elevation.value().height
                          << " with variance: " << elevation.value().variance << std::endl;
                z(0) = static_cast<double>(elevation.value().height) -
                       updated_state.contacts_position.at(cf).z();
                R(0, 0) = static_cast<double>(elevation.value().variance);
                const Eigen::MatrixXd s = R + H * P_ * H.transpose();
                const Eigen::MatrixXd K = P_ * H.transpose() * s.inverse();
                const Eigen::VectorXd dx = K * z;
                updateState(updated_state, dx, P_);
                P_ = (I_ - K * H) * P_;
            }
        }
    }
    return updated_state;
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

// TODO: @sp Maybe consider passing the state by reference instead of having to copy
BaseState ContactEKF::update(const BaseState& state, const KinematicMeasurement& kin,
                             std::optional<OdometryMeasurement> odom,
                             std::shared_ptr<NaiveTerrainElevation> terrain_estimator) {
    // Use the predicted state to update the terrain estimator
    if (terrain_estimator) {
        std::vector<std::array<float, 2>> con_locs;
        for (const auto& [cf, cp] : kin.contacts_probability) {
            if (cp > 0.15) {
                const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
                const Eigen::Vector3d r = state.base_position;
                const Eigen::Vector3d con_pos = r + R * kin.base_to_foot_positions.at(cf);
                const std::array<float, 2> con_pos_xy = {static_cast<float>(con_pos.x()),
                                                         static_cast<float>(con_pos.y())};
                const float con_pos_z = static_cast<float>(con_pos.z());

                // Transform measurement noise to world frame
                const Eigen::Matrix3d con_cov =
                    R * kin.contacts_position_noise.at(cf) * R.transpose();

                // TODO: @sp make this a const parameter
                if (!terrain_estimator->update(con_pos_xy, con_pos_z,
                                               static_cast<float>(con_cov(2, 2)) + 1e-3)) {
                    std::cout
                        << "Contact for " << cf
                        << "is not inside the terrain elevation map and thus height is not updated "
                        << std::endl;
                } else {
                    con_locs.push_back(con_pos_xy);
                }
            }
        }

        // TODO @sp: Interpolation reduces accuracy?
        // if (!terrain_estimator->interpolate(con_locs,
        //                                     {static_cast<float>(state.base_linear_velocity.x()),
        //                                      static_cast<float>(state.base_linear_velocity.y())},
        //                                     0.25)) {
        //     std::cout << "Interpolation failed " << std::endl;
        // }
    }

    // Update the state with the relative to base contacts
    BaseState updated_state =
        updateWithContacts(state, kin.contacts_position, kin.contacts_position_noise,
                           kin.contacts_status, kin.position_cov, kin.contacts_orientation,
                           kin.contacts_orientation_noise, kin.orientation_cov, terrain_estimator);

    if (odom.has_value()) {
        updated_state =
            updateWithOdometry(updated_state, odom->base_position, odom->base_orientation,
                               odom->base_position_cov, odom->base_orientation_cov);
    }

    // Update the state with the absolute terrain height at each contact location and potentially
    // recenter the terrain mapper
    if (terrain_estimator) {
        updated_state = updateWithTerrain(updated_state, kin.contacts_status, *terrain_estimator);
        // TODO: @sp make this a const parameter
        // Recenter the map
        // const std::array<float, 2> base_pos_xy = {
        //     static_cast<float>(updated_state.base_position.x()),
        //     static_cast<float>(updated_state.base_position.y())};
        // const std::array<float, 2>& map_origin_xy = terrain_estimator->getMapOrigin();
        // if ((abs(base_pos_xy[0] - map_origin_xy[0]) > 0.5) ||
        //     (abs(base_pos_xy[1] - map_origin_xy[1]) > 0.5)) {
        //     terrain_estimator->recenter(base_pos_xy);
        // }
    }

    return updated_state;
}

}  // namespace serow
