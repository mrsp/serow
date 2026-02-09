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

#include "lie.hpp"

namespace serow {

void ContactEKF::init(const BaseState& state, std::set<std::string> contacts_frame,
                      double g, double imu_rate, bool use_imu_orientation, bool verbose) {
    num_leg_end_effectors_ = contacts_frame.size();
    contacts_frame_ = std::move(contacts_frame);
    g_ = Eigen::Vector3d(0.0, 0.0, -g);
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
    
    // Initialize input indices
    ng_idx_ = Eigen::Array3i::LinSpaced(0, 3);
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
    last_imu_timestamp_.reset();

    use_imu_orientation_ = use_imu_orientation;
    verbose_ = verbose;
    if (verbose) {
        std::cout << "[SEROW/ContactEKF]: Initialized successfully" << std::endl;
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
    last_imu_timestamp_ = state.timestamp;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> ContactEKF::computePredictionJacobians(
    const BaseState& state, Eigen::Vector3d angular_velocity) {
    angular_velocity -= state.imu_angular_velocity_bias;
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d v = R.transpose() * state.base_linear_velocity;

    Eigen::MatrixXd Ac(num_states_, num_states_), Lc(num_states_, num_inputs_);
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
    double dt = nominal_dt_;
    if (last_imu_timestamp_.has_value()) {
        dt = imu.timestamp - last_imu_timestamp_.value();
    }

    // Check if the sample time is abnormal
    if (dt < 0.0) {
        std::cout << "[SEROW/ContactEKF]: Predict step sample time is negative " << dt
                  << " while the nominal sample time is " << nominal_dt_
                  << " returning without updating the state" << std::endl;
        return;
    }
    if (dt < nominal_dt_ / 2) {
        if (verbose_) {
            std::cout << "[SEROW/ContactEKF]: Predict step sample time is abnormal " << dt
                      << " while the nominal sample time is " << nominal_dt_
                      << " setting to nominal" << std::endl;
        }
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
    // First-order discrete-time process noise with Euler integration
    const Eigen::MatrixXd Qd = Lc * Qc * Lc.transpose() * dt;
    P_ = Ad * P_ * Ad.transpose() + Qd;

    // Predict the state
    computeDiscreteDynamics(state, dt, imu.angular_velocity, imu.linear_acceleration);
    last_imu_timestamp_ = imu.timestamp;
}

void ContactEKF::computeDiscreteDynamics(
    BaseState& state, double dt, Eigen::Vector3d angular_velocity,
    Eigen::Vector3d linear_acceleration) {
    angular_velocity -= state.imu_angular_velocity_bias;
    linear_acceleration -= state.imu_linear_acceleration_bias;

    // Nonlinear Process Model
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d r = state.base_position;
    const Eigen::Vector3d v = R.transpose() * state.base_linear_velocity;
    const Eigen::Vector3d v_w = state.base_linear_velocity;

    // Linear velocity
    state.base_linear_velocity.noalias() = R *
        ((v.cross(angular_velocity) + R.transpose() * g_ + linear_acceleration) * dt) + v_w;

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
    Eigen::MatrixXd H(6, num_states_);
    Eigen::MatrixXd PH_transpose(num_states_, 6);
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

    PH_transpose.noalias() = P_ * H.transpose();
    const Eigen::Matrix<double, 6, 6> s = R + H * PH_transpose;
    const Eigen::MatrixXd K = PH_transpose * s.inverse();
    const Eigen::VectorXd dx = K * z;

    P_ = (I_ - K * H) * P_;
    updateState(state, dx, P_);
}

void ContactEKF::updateWithTerrain(BaseState& state,
                                   const std::map<std::string, Eigen::Vector3d>& contacts_position,
                                   const std::map<std::string, Eigen::Matrix3d>& contacts_position_cov,
                                   const std::map<std::string, double>& contacts_probability,
                                   std::shared_ptr<TerrainElevation> terrain_estimator) {

    const Eigen::Vector3d p_world_to_base = state.base_position;
    const Eigen::Matrix3d R_world_to_base = state.base_orientation.toRotationMatrix();

    // Initialize the innovation vector z, the linearized measurement matrix H, and the measurement noise 
    // matrix R
    Eigen::VectorXd z(1);
    Eigen::MatrixXd H(1, num_states_);
    Eigen::MatrixXd R(1, 1);
    Eigen::MatrixXd PH_transpose(num_states_, 1);

    for (const auto& [cf, cp] : contacts_probability) {
        // Update only when the elevation at the contact points is available and updated in the
        // map
        if (cp > 0.5) {
            const Eigen::Vector3d con_pos_world = R_world_to_base * contacts_position.at(cf) + p_world_to_base;
            const std::array<float, 2> con_pos_xy = {
                static_cast<float>(con_pos_world.x()),
                static_cast<float>(con_pos_world.y())};
            const auto elevation = terrain_estimator->getElevation(con_pos_xy);
            if (elevation.has_value() && elevation.value().updated) {
                const Eigen::Matrix3d con_cov_world = R_world_to_base * contacts_position_cov.at(cf) * R_world_to_base.transpose();

                // Construct the linearized measurement matrix H
                H.setZero();
                // dh/dp_z
                H(0, p_idx_[2]) = 1.0;

                // dh/d_chi = [0, 0, 1] * (-R * skew(p_contact_body))
                // const Eigen::Matrix3d skew_cp = lie::so3::wedge(contacts_position.at(cf));
                // H.block(0, r_idx_[0], 1, 3) = -(R_world_to_base * skew_cp).row(2);

                // Compute innovation
                z(0) = static_cast<double>(elevation.value().height) - con_pos_world.z();

                // Construct the measurement noise matrix R
                R(0, 0) = (static_cast<double>(elevation.value().variance + con_cov_world(2, 2)) + 1e-6) / cp; 

                PH_transpose.noalias() = P_ * H.transpose();
                const Eigen::MatrixXd s = R + H * PH_transpose;
                const Eigen::MatrixXd K = PH_transpose * s.inverse();
                const Eigen::VectorXd dx = K * z;
                P_ = (I_ - K * H) * P_;
                updateState(state, dx, P_);
            }
        }
    }  
}

BaseState ContactEKF::updateStateCopy(const BaseState& state, const Eigen::VectorXd& dx,
                                      const Eigen::MatrixXd& P) const {
    BaseState updated_state = state;
    updated_state.base_position += dx(p_idx_);
    updated_state.base_position_cov = P(p_idx_, p_idx_);

    updated_state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
            .normalized();
    const Eigen::Matrix3d R = updated_state.base_orientation.toRotationMatrix();
    updated_state.base_linear_velocity += R * dx(v_idx_);
    updated_state.base_linear_velocity_cov.noalias() = R * P(v_idx_, v_idx_) * R.transpose();
    updated_state.base_orientation_cov = P(r_idx_, r_idx_);
    updated_state.imu_angular_velocity_bias += dx(bg_idx_);
    updated_state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    updated_state.imu_linear_acceleration_bias += dx(ba_idx_);
    updated_state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);
    return updated_state;
}

void ContactEKF::updateState(BaseState& state, const Eigen::VectorXd& dx,
                             const Eigen::MatrixXd& P) const {
    state.base_position += dx(p_idx_);
    state.base_position_cov = P(p_idx_, p_idx_);
    state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
            .normalized();
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();    
    state.base_linear_velocity += R * dx(v_idx_);
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
        for (const auto& [cf, cp] : kin.contacts_probability) {
            if (cp > terrain_estimator->getMinContactProbability()) {
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
                                               static_cast<float>((con_cov(2, 2) + 1e-6) / cp))) {
                    std::cout << "Contact for " << cf
                              << " is not inside the terrain elevation map and thus height is not "
                                 "updated "
                              << std::endl;
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
        updateWithIMUOrientation(state, imu.orientation, imu.orientation_cov);
    }

    updateWithBaseLinearVelocity(state, kin.base_linear_velocity, kin.base_linear_velocity_cov);

    if (odom.has_value()) {
        updateWithOdometry(state, odom->base_position, odom->base_orientation,
                           odom->base_position_cov, odom->base_orientation_cov);
    }

    // Update the state with the absolute terrain height at each contact location and
    // potentially recenter the terrain mapper
    if (terrain_estimator) {
        updateWithTerrain(state, kin.contacts_position, kin.contacts_position_noise, kin.contacts_probability, terrain_estimator);
        // Recenter the map
        const std::array<float, 2> base_pos_xy = {static_cast<float>(state.base_position.x()),
                                                  static_cast<float>(state.base_position.y())};
        const std::array<float, 2>& map_origin_xy = terrain_estimator->getMapOrigin();
        if ((abs(base_pos_xy[0] - map_origin_xy[0]) >
             terrain_estimator->getMaxRecenterDistance()) ||
            (abs(base_pos_xy[1] - map_origin_xy[1]) >
             terrain_estimator->getMaxRecenterDistance())) {
            terrain_estimator->recenter(base_pos_xy);
        }
        terrain_estimator->interpolateContactPoints();
    }

    Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
    T_world_to_base.translation() = state.base_position;
    T_world_to_base.linear() = state.base_orientation.toRotationMatrix();
    for (const auto& cf : contacts_frame_) {
        state.contacts_position.at(cf) = T_world_to_base * kin.contacts_position.at(cf);
        if (state.contacts_orientation.has_value() && state.contacts_orientation.value().count(cf) > 0) {
            state.contacts_orientation.value().at(cf) = T_world_to_base.linear() * kin.contacts_orientation.value().at(cf);
        }
    }
}

void ContactEKF::updateWithIMUOrientation(BaseState& state,
                                          const Eigen::Quaterniond& imu_orientation,
                                          const Eigen::Matrix3d& imu_orientation_cov) {
    // Construct the linearized measurement matrix H
    Eigen::MatrixXd H;
    H.setZero(3, num_states_);
    H.block(0, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    // Construct the innovation vector z
    const Eigen::Vector3d z = lie::so3::minus(imu_orientation, state.base_orientation);

    // Construct the measurement noise matrix R
    const Eigen::Matrix3d& R = imu_orientation_cov;

    const Eigen::MatrixXd PH_transpose = P_ * H.transpose();
    const Eigen::Matrix3d s = R + H * PH_transpose;
    const Eigen::MatrixXd K = PH_transpose * s.inverse();
    const Eigen::VectorXd dx = K * z;

    P_ = (I_ - K * H) * P_;
    updateState(state, dx, P_);
}

void ContactEKF::updateWithBaseLinearVelocity(BaseState& state, const Eigen::Vector3d& base_linear_velocity,
                                              const Eigen::Matrix3d& base_linear_velocity_cov) {
    const int num_iter = 5;
    Eigen::MatrixXd H(3, num_states_);
    H.setZero();
    Eigen::MatrixXd K(num_states_, 3);
    Eigen::MatrixXd PH_transpose(num_states_, 3);
    Eigen::Vector3d z;
    Eigen::Matrix3d s;

    // Iterative ESKF update
    for (size_t iter = 0; iter < num_iter; iter++) {
        // Construct the linearized measurement matrix H around the current state estimate
        const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
        const Eigen::Vector3d v = R.transpose() * state.base_linear_velocity;
        H.block(0, v_idx_[0], 3, 3) = R;
        H.block(0, r_idx_[0], 3, 3) = -R * lie::so3::wedge(v);
            
        // Construct the innovation vector z
        const Eigen::Vector3d z = base_linear_velocity - state.base_linear_velocity;

        // Compute the Kalman gain with the current linearization
        PH_transpose.noalias() = P_ * H.transpose();
        s.noalias() = base_linear_velocity_cov + H * PH_transpose;
        K.noalias() = PH_transpose * s.inverse();
        const Eigen::VectorXd dx = K * z;
        
        // Update only the state estimate (not the covariance) during iterations
        state.base_position += dx(p_idx_);
        state.base_orientation =
            Eigen::Quaterniond(lie::so3::plus(state.base_orientation.toRotationMatrix(), dx(r_idx_)))
                .normalized();
        const Eigen::Matrix3d R_updated = state.base_orientation.toRotationMatrix();
        state.base_linear_velocity += R_updated * dx(v_idx_);
        state.imu_angular_velocity_bias += dx(bg_idx_);
        state.imu_linear_acceleration_bias += dx(ba_idx_);
        if (dx.squaredNorm() < 1e-9) {
            break;
        }
    }
    
    // Update the covariance once after convergence
    P_ = (I_ - K * H) * P_;
    
    // Update state covariances with the final P_
    state.base_position_cov = P_(p_idx_, p_idx_);
    state.base_orientation_cov = P_(r_idx_, r_idx_);
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    state.base_linear_velocity_cov.noalias() = R * P_(v_idx_, v_idx_) * R.transpose();
    state.imu_angular_velocity_bias_cov = P_(bg_idx_, bg_idx_);
    state.imu_linear_acceleration_bias_cov = P_(ba_idx_, ba_idx_);
}

}  // namespace serow
