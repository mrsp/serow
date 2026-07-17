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
#include "LeftInvariantEKF.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "lie.hpp"

namespace {

static constexpr double kMinTerrainDt = 1e-6;
static constexpr double kStableContactThreshold = 0.5;

}  // namespace

namespace serow {

void LeftInvariantEKF::init(const BaseState& state, std::set<std::string> contacts_frame,
                            const double g, const double imu_rate, const double kin_rate,
                            const double eps, const bool point_feet, const bool use_imu_orientation,
                            const bool verbose) {
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

    // State error indices: [xi_R, xi_v, xi_p, db_g, db_a]
    r_idx_ = Eigen::Array3i::LinSpaced(0, 2);
    v_idx_ = r_idx_ + 3;
    p_idx_ = v_idx_ + 3;
    bg_idx_ = p_idx_ + 3;
    ba_idx_ = bg_idx_ + 3;

    // Input noise indices
    ng_idx_ = Eigen::Array3i::LinSpaced(0, 2);
    na_idx_ = ng_idx_ + 3;
    nz_idx_ = na_idx_ + 3;
    nbg_idx_ = nz_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;

    // Error covariance
    P_ = I_;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(v_idx_, v_idx_) = state.base_linear_velocity_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;

    Ac_.setZero(num_states_, num_states_);
    Ac_(p_idx_, v_idx_) = Eigen::Matrix3d::Identity();
    Ac_(r_idx_, bg_idx_) = -Eigen::Matrix3d::Identity();
    Ac_(v_idx_, ba_idx_) = -Eigen::Matrix3d::Identity();

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), zero (6), gyro_bias (9), acc_bias (12)
    // Lc is Identity matrix for Left-Invariant EKF
    Lc_.setIdentity(num_states_, num_states_);
    last_imu_predict_timestamp_.reset();
    last_kin_update_timestamp_.reset();
    last_imu_update_timestamp_.reset();
    last_terrain_update_timestamp_.reset();

    use_imu_orientation_ = use_imu_orientation;
    verbose_ = verbose;
    if (verbose) {
        std::cout << "[SEROW/LeftInvariantEKF]: Initialized successfully" << '\n';
    }
}

void LeftInvariantEKF::setState(const BaseState& state) {
    P_ = I_;
    P_(r_idx_, r_idx_) = state.base_orientation_cov;
    P_(v_idx_, v_idx_) = state.base_linear_velocity_cov;
    P_(p_idx_, p_idx_) = state.base_position_cov;
    P_(bg_idx_, bg_idx_) = state.imu_angular_velocity_bias_cov;
    P_(ba_idx_, ba_idx_) = state.imu_linear_acceleration_bias_cov;
    last_imu_update_timestamp_ = state.timestamp;
    last_kin_update_timestamp_ = state.timestamp;
    last_imu_predict_timestamp_ = state.timestamp;
    last_terrain_update_timestamp_ = state.timestamp;
    first_position_.reset();
    first_odometry_position_.reset();
    first_odometry_orientation_.reset();
    terrain_contact_filter_.reset();
}

// ---------------------------------------------------------------------------
// Prediction Jacobians
// ---------------------------------------------------------------------------
//  Ac (15x15) =  [ A_X   | A_Xtheta ]
//                [-------|--------- ]
//                [ 0_6x9 | 0_6x6    ]
//
//  A_X (9x9, state-independent):
//    [ -[w]x    0      0  ]
//    [ -[a]x  -[w]x    0  ]
//    [  0       I   -[w]x ]
//
//  A_Xtheta (9x6, state-independent):
//    [   -I       0  ]
//    [    0      -I  ]
//    [    0       0  ]
//
//  Lc (15x15): Identity matrix
// ---------------------------------------------------------------------------
std::tuple<Eigen::Matrix<double, 15, 15>, Eigen::Matrix<double, 15, 15>>
LeftInvariantEKF::computePredictionJacobians(const BaseState& state,
                                             Eigen::Vector3d angular_velocity,
                                             Eigen::Vector3d linear_acceleration) {
    angular_velocity -= state.imu_angular_velocity_bias;
    linear_acceleration -= state.imu_linear_acceleration_bias;

    // Start from the constant (state-independent) parts
    Eigen::Matrix<double, 15, 15> Ac = Ac_;
    Eigen::Matrix<double, 15, 15> Lc = Lc_;

    // Fill in the dynamic parts
    const Eigen::Matrix3d w_x = -lie::so3::wedge(angular_velocity);
    const Eigen::Matrix3d a_x = -lie::so3::wedge(linear_acceleration);
    Ac(r_idx_, r_idx_) = w_x;
    Ac(v_idx_, r_idx_) = a_x;
    Ac(v_idx_, v_idx_) = w_x;
    Ac(p_idx_, p_idx_) = w_x;

    return std::make_tuple(Ac, Lc);
}

// ---------------------------------------------------------------------------
// Predict step
// ---------------------------------------------------------------------------
void LeftInvariantEKF::predict(BaseState& state, const ImuMeasurement& imu) {
    double dt = nominal_imu_dt_;
    if (last_imu_predict_timestamp_.has_value()) {
        dt = imu.timestamp - last_imu_predict_timestamp_.value();
    }
    last_imu_predict_timestamp_ = imu.timestamp;

    // Check if the sample time is abnormal
    if (dt < 0.0) {
        std::cout << "[SEROW/LeftInvariantEKF]: Predict step sample time is negative " << dt
                  << " while the nominal sample time is " << nominal_imu_dt_
                  << " returning without updating the state" << '\n';
        return;
    }

    // Continuous-time Jacobians
    const auto& [Ac, Lc] =
        computePredictionJacobians(state, imu.angular_velocity, imu.linear_acceleration);

    // Discrete state-transition: Phi = exp(A dt) ~ I + A dt + 0.5 A^2 dt^2
    const Eigen::Matrix<double, 15, 15> Ad = I_ + Ac * dt + 0.5 * Ac * Ac * dt * dt;

    // Continuous noise covariance
    Eigen::Matrix<double, 15, 15> Qc = Eigen::Matrix<double, 15, 15>::Zero();
    Qc(ng_idx_, ng_idx_) = imu.angular_velocity_cov;
    Qc(na_idx_, na_idx_) = imu.linear_acceleration_cov;
    Qc(nz_idx_, nz_idx_) = Eigen::Matrix3d::Zero();
    Qc(nbg_idx_, nbg_idx_) = imu.angular_velocity_bias_cov;
    Qc(nba_idx_, nba_idx_) = imu.linear_acceleration_bias_cov;

    // Propagate covariance
    const Eigen::Matrix<double, 15, 15> P_new = Ad * P_ * Ad.transpose() + Qc * dt;
    P_ = 0.5 * (P_new + P_new.transpose());  // Enforce symmetry

    // Propagate the mean state (world-frame dynamics)
    computeDiscreteDynamics(state, dt, imu.angular_velocity, imu.linear_acceleration);
}

// ---------------------------------------------------------------------------
// Discrete dynamics (world-frame)
//   R_{k+1} = R_k Exp(omega dt)
//   v_{k+1} = v_k + (R_k a + g) dt
//   p_{k+1} = p_k + v_k dt + 0.5 (R_k a + g) dt^2
// ---------------------------------------------------------------------------
void LeftInvariantEKF::computeDiscreteDynamics(BaseState& state, double dt,
                                               Eigen::Vector3d angular_velocity,
                                               Eigen::Vector3d linear_acceleration) {
    angular_velocity -= state.imu_angular_velocity_bias;
    linear_acceleration -= state.imu_linear_acceleration_bias;

    // Nonlinear Process Model
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Vector3d v = state.base_linear_velocity;
    const Eigen::Vector3d p = state.base_position;

    const Eigen::Vector3d v_acc = (R * linear_acceleration + g_) * dt;

    // World-frame velocity
    state.base_linear_velocity = v + v_acc;

    // World-frame position
    state.base_position = p + v * dt + 0.5 * v_acc * dt;

    // Orientation
    state.base_orientation =
        Eigen::Quaterniond(lie::so3::plus(R, angular_velocity * dt)).normalized();

    // Derive body-frame velocity from updated world-frame quantities
    state.base_local_linear_velocity =
        state.base_orientation.toRotationMatrix().transpose() * state.base_linear_velocity;
}

// ---------------------------------------------------------------------------
// Odometry update  (pose measurement in world frame)
//
//   z_R = log(R^T * R_y)   ≈  ξ_R             (left-invariant)
//   z_p = R^T * (p_y − p)  ≈  R^T * ξ_p       (left-invariant)
//   H   = [ I  0  0 | 0 ]   (rotation row)
//         [  0  0 I | 0 ]   (position row)
// ---------------------------------------------------------------------------
void LeftInvariantEKF::updateWithOdometry(BaseState& state, const Eigen::Vector3d& base_position,
                                          const Eigen::Quaterniond& base_orientation,
                                          const Eigen::Matrix3d& base_position_cov,
                                          const Eigen::Matrix3d& base_orientation_cov) {
    if (!first_odometry_position_.has_value() || !first_odometry_orientation_.has_value() ||
        !first_position_.has_value()) {
        first_odometry_position_ = base_position - state.base_position;
        first_odometry_orientation_ = state.base_orientation * base_orientation.inverse();
        first_position_ = state.base_position;
        return;
    }

    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Matrix3d R_z = R.transpose() * base_position_cov * R;
    // Remove the initial offset if any
    const Eigen::Vector3d bp = first_position_.value() +
        first_odometry_orientation_.value() * (base_position - first_odometry_position_.value());
    const Eigen::Quaterniond bo = first_odometry_orientation_.value() * base_orientation;

    // Construct the linearized measurement matrix H
    Eigen::Matrix<double, 6, 15> H = Eigen::Matrix<double, 6, 15>::Zero();
    H.block(0, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();
    H.block(3, p_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    // Compute the innovations
    Eigen::Matrix<double, 6, 1> z;
    z.head(3) = lie::so3::logMap(state.base_orientation.inverse() * bo);
    z.tail(3) = R.transpose() * (bp - state.base_position);

    // RESKF outlier-robust update
    base_position_outlier_detector.init();
    Eigen::Matrix<double, 15, 15> P_i = P_;
    BaseState updated_state_i = state;

    const Eigen::Matrix<double, 15, 6> PH_transpose = P_ * H.transpose();
    Eigen::Matrix<double, 6, 6> N = Eigen::Matrix<double, 6, 6>::Zero();
    N.topLeftCorner<3, 3>() = base_orientation_cov;
    const Eigen::Matrix3d bb = bp * bp.transpose();
    for (size_t i = 0; i < base_position_outlier_detector.iters; i++) {
        if (base_position_outlier_detector.zeta > base_position_outlier_detector.threshold) {
            N.bottomRightCorner<3, 3>() = R_z / base_position_outlier_detector.zeta;
            const Eigen::Matrix<double, 6, 6> s = N + H * PH_transpose;
            const Eigen::Matrix<double, 15, 6> K =
                s.ldlt().solve(PH_transpose.transpose()).transpose();
            const Eigen::Matrix<double, 15, 1> dx = K * z;
            const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
            P_i = IKH * P_ * IKH.transpose() + K * N * K.transpose();
            updated_state_i = updateStateCopy(state, dx, P_);

            // Outlier detection with the base position measurement vector
            const Eigen::Vector3d& x_i = updated_state_i.base_position;
            const Eigen::Matrix3d BetaT = bb - bp * x_i.transpose() - x_i * bp.transpose() +
                x_i * x_i.transpose() +
                H.block(3, p_idx_[0], 3, 3) * P_i.block(p_idx_[0], p_idx_[0], 3, 3) *
                    H.block(3, p_idx_[0], 3, 3).transpose();
            base_position_outlier_detector.estimate(BetaT, R_z);
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

// ---------------------------------------------------------------------------
// Terrain height update  (scalar, terrain-safe version)
//   z = h_map(x_foot,y_foot) - z_foot_world
//
// The full first-order model can couple terrain height into orientation and
// horizontal states.  In SBEE that coupling can create a feedback loop.
// ---------------------------------------------------------------------------
void LeftInvariantEKF::updateWithTerrain(
    BaseState& state, const std::map<std::string, Eigen::Vector3d>& contacts_position,
    const std::map<std::string, double>& contacts_probability, const double timestamp,
    std::shared_ptr<TerrainElevation> terrain_estimator) {
    if (!terrain_estimator) {
        return;
    }

    double dt = nominal_kin_dt_;
    if (last_terrain_update_timestamp_.has_value()) {
        dt = timestamp - last_terrain_update_timestamp_.value();
    }
    last_terrain_update_timestamp_ = timestamp;

    if (dt <= 0.0 || !std::isfinite(dt)) {
        return;
    }
    dt = std::max(dt, kMinTerrainDt);

    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block(0, p_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();
    Eigen::Vector3d z = Eigen::Vector3d::Zero();
    Eigen::Matrix3d N = Eigen::Matrix3d::Zero();

    for (const auto& [cf, cp] : contacts_probability) {
        if (cp < kStableContactThreshold) {
            continue;
        }

        if (contacts_position.count(cf) == 0) {
            continue;
        }

        const Eigen::Vector3d p_world_to_base = state.base_position;
        const Eigen::Matrix3d R_world_to_base = state.base_orientation.toRotationMatrix();
        const Eigen::Matrix3d R_world_to_base_transpose = R_world_to_base.transpose();

        Eigen::Vector3d con_pos_world =
            R_world_to_base * contacts_position.at(cf) + p_world_to_base;
        if (!(con_pos_world).allFinite()) {
            continue;
        }

        const std::array<float, 2> con_pos_xy = {static_cast<float>(con_pos_world.x()),
                                                 static_cast<float>(con_pos_world.y())};

        const auto elevation = terrain_estimator->getElevation(con_pos_xy);
        if (!elevation.has_value() || !elevation.value().updated) {
            continue;
        }

        // Construct the corrected contact position
        con_pos_world.z() = static_cast<double>(elevation.value().height);

        // Reconstruct the base position from the contact position
        const Eigen::Vector3d p_world_to_base_measured =
            con_pos_world - R_world_to_base * contacts_position.at(cf);

        // Compute the innovation
        z = R_world_to_base_transpose * (p_world_to_base_measured - p_world_to_base);

        // Compute the measurement covariance
        N = Eigen::Matrix3d::Identity() *
            std::max(static_cast<double>(elevation.value().variance),
                     static_cast<double>(terrain_estimator->getMinVariance())) /
            (cp * dt);
        N(0, 0) = 1e6;
        N(1, 1) = 1e6;
        N = R_world_to_base_transpose * N * R_world_to_base;

        const Eigen::Matrix<double, 15, 3> PH_transpose = P_ * H.transpose();
        const Eigen::Matrix3d s = N + H * PH_transpose;
        const Eigen::Matrix<double, 15, 3> K = s.ldlt().solve(PH_transpose.transpose()).transpose();
        const Eigen::Matrix<double, 15, 1> dx = K * z;

        const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
        Eigen::Matrix<double, 15, 15> P_new;
        P_new.noalias() = IKH * P_ * IKH.transpose();
        P_new += K * N * K.transpose();
        P_ = P_new;
        updateState(state, dx, P_);
    }
}

// ---------------------------------------------------------------------------
// State retraction
//   R+    = R * Exp(xi_R)     (RIGHT multiply)
//   v+    = v + R * xi_v      (world-frame)
//   p+    = p + R * xi_p      (world-frame)
//   b+    = b + db            (linear)
//   v_body = R+^T  v+         (derived)
// ---------------------------------------------------------------------------
BaseState LeftInvariantEKF::updateStateCopy(const BaseState& state,
                                            const Eigen::Matrix<double, 15, 1>& dx,
                                            const Eigen::Matrix<double, 15, 15>& P) const {
    BaseState updated_state = state;
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();

    // World-frame orientation (right multiply)
    updated_state.base_orientation =
        Eigen::Quaterniond(R * lie::so3::expMap(dx(r_idx_))).normalized();

    // World-frame velocity and position
    updated_state.base_linear_velocity += R * dx(v_idx_);
    updated_state.base_position += R * dx(p_idx_);

    // Biases
    updated_state.imu_angular_velocity_bias += dx(bg_idx_);
    updated_state.imu_linear_acceleration_bias += dx(ba_idx_);

    // Derive body-frame velocity
    updated_state.base_local_linear_velocity =
        updated_state.base_orientation.toRotationMatrix().transpose() *
        updated_state.base_linear_velocity;

    // Covariances (world-frame)
    updated_state.base_orientation_cov = P(r_idx_, r_idx_);
    updated_state.base_linear_velocity_cov = P(v_idx_, v_idx_);
    updated_state.base_position_cov = P(p_idx_, p_idx_);
    updated_state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    updated_state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);

    return updated_state;
}

void LeftInvariantEKF::updateState(BaseState& state, const Eigen::Matrix<double, 15, 1>& dx,
                                   const Eigen::Matrix<double, 15, 15>& P) const {
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    // World-frame orientation (right multiply)
    state.base_orientation = Eigen::Quaterniond(R * lie::so3::expMap(dx(r_idx_))).normalized();

    // World-frame velocity and position
    state.base_linear_velocity += R * dx(v_idx_);
    state.base_position += R * dx(p_idx_);

    // Biases
    state.imu_angular_velocity_bias += dx(bg_idx_);
    state.imu_linear_acceleration_bias += dx(ba_idx_);

    // Derive body-frame velocity
    state.base_local_linear_velocity =
        state.base_orientation.toRotationMatrix().transpose() * state.base_linear_velocity;

    // Covariances (world-frame)
    state.base_orientation_cov = P(r_idx_, r_idx_);
    state.base_linear_velocity_cov = P(v_idx_, v_idx_);
    state.base_position_cov = P(p_idx_, p_idx_);
    state.imu_angular_velocity_bias_cov = P(bg_idx_, bg_idx_);
    state.imu_linear_acceleration_bias_cov = P(ba_idx_, ba_idx_);
}

// ---------------------------------------------------------------------------
// Full update step
// ---------------------------------------------------------------------------
void LeftInvariantEKF::update(BaseState& state, const ImuMeasurement& imu,
                              const KinematicMeasurement& kin,
                              std::optional<OdometryMeasurement> odom,
                              std::shared_ptr<TerrainElevation> terrain_estimator) {
    // Use raw contacts for the leg-velocity update, but use a debounced terrain-only
    // contact stream for terrain-map updates and terrain EKF corrections.
    std::map<std::string, double> terrain_contacts_probability;
    double terrain_dt = nominal_kin_dt_;
    bool terrain_dt_valid = true;

    if (terrain_estimator) {
        terrain_contacts_probability = terrain_contact_filter_.filter(kin.contacts_probability);

        if (last_terrain_update_timestamp_.has_value()) {
            terrain_dt = kin.timestamp - last_terrain_update_timestamp_.value();
        }
        if (terrain_dt <= 0.0 || !std::isfinite(terrain_dt)) {
            terrain_dt_valid = false;
            terrain_dt = nominal_kin_dt_;
        }
        terrain_dt = std::max(terrain_dt, kMinTerrainDt);
    }

    // 1) Optional absolute IMU orientation update.
    if (use_imu_orientation_) {
        updateWithIMUOrientation(state, imu.orientation, imu.orientation_cov, imu.timestamp);
    }

    // 2) Base linear velocity update from leg odometry, only if at least one raw
    // contact is available.
    double den = 0.0;
    for (const auto& [cf, cp] : kin.contacts_probability) {
        den += cp;
    }
    if (den > eps_) {
        updateWithBaseLinearVelocity(state, kin.base_linear_velocity, kin.base_linear_velocity_cov,
                                     kin.timestamp);
    }

    // 3) Optional external odometry update.
    if (odom.has_value()) {
        updateWithOdometry(state, odom->base_position, odom->base_orientation,
                           odom->base_position_cov, odom->base_orientation_cov);
    }

    if (terrain_estimator && terrain_dt_valid) {
        // 4) Update the terrain map using the corrected state.
        Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
        T_world_to_base.translation() = state.base_position;
        T_world_to_base.linear() = state.base_orientation.toRotationMatrix();

        for (const auto& [cf, cp] : terrain_contacts_probability) {
            if (cp <= terrain_estimator->getMinContactProbability()) {
                continue;
            }
            if (kin.contacts_position.count(cf) == 0 ||
                kin.contacts_position_noise.count(cf) == 0 ||
                kin.base_to_foot_positions.count(cf) == 0 ||
                kin.base_to_foot_linear_velocities.count(cf) == 0) {
                continue;
            }

            const Eigen::Vector3d con_pos_world = T_world_to_base * kin.contacts_position.at(cf);
            if (!(con_pos_world).allFinite()) {
                continue;
            }

            const std::array<float, 2> con_pos_xy = {static_cast<float>(con_pos_world.x()),
                                                     static_cast<float>(con_pos_world.y())};
            const float con_pos_z = static_cast<float>(con_pos_world.z());

            const Eigen::Matrix3d con_cov = T_world_to_base.linear() *
                kin.contacts_position_noise.at(cf) * T_world_to_base.linear().transpose();

            std::optional<std::array<float, 3>> normal = std::nullopt;
            if (!point_feet_ && kin.contacts_orientation.has_value() &&
                kin.contacts_orientation.value().count(cf) > 0 &&
                kin.base_to_foot_angular_velocities.count(cf) > 0) {
                const Eigen::Matrix3d& R_world_to_base = T_world_to_base.linear();
                const Eigen::Vector3d n_contact =
                    (R_world_to_base * kin.contacts_orientation.value().at(cf).toRotationMatrix() *
                     Eigen::Vector3d::UnitZ())
                        .normalized();

                const Eigen::Vector3d omega_contact =
                    R_world_to_base * kin.base_to_foot_angular_velocities.at(cf) +
                    state.base_angular_velocity;

                const Eigen::Vector3d p_base_to_leg =
                    R_world_to_base * kin.base_to_foot_positions.at(cf);
                const Eigen::Vector3d v_contact = state.base_linear_velocity +
                    state.base_angular_velocity.cross(p_base_to_leg) +
                    R_world_to_base * kin.base_to_foot_linear_velocities.at(cf);

                if (cp > terrain_estimator->getMinStableContactProbability() &&
                    omega_contact.norm() < terrain_estimator->getMinStableFootAngularVelocity() &&
                    v_contact.norm() < terrain_estimator->getMinStableFootLinearVelocity() &&
                    (n_contact).allFinite() && std::fabs(n_contact.z()) > 1e-3) {
                    normal = {static_cast<float>(n_contact.x()), static_cast<float>(n_contact.y()),
                              static_cast<float>(n_contact.z())};
                }
            } else {
                const Eigen::Matrix3d& R_world_to_base = T_world_to_base.linear();
                const Eigen::Vector3d n_contact =
                    (R_world_to_base * Eigen::Vector3d::UnitZ()).normalized();
                normal = {static_cast<float>(n_contact.x()), static_cast<float>(n_contact.y()),
                          static_cast<float>(n_contact.z())};
            }

            const float terrain_meas_variance = static_cast<float>(
                std::max((con_cov(2, 2) + 1e-6) / (cp * terrain_dt),
                         static_cast<double>(terrain_estimator->getMinVariance())));

            if (!terrain_estimator->update(con_pos_xy, con_pos_z, terrain_meas_variance, normal)) {
                if (verbose_) {
                    std::cout << "[SEROW/LeftInvariantEKF]: Contact for " << cf
                              << " is outside the terrain elevation map; height not updated.\n";
                }
            } else if (terrain_contact_filter_.becameStable(cf)) {
                terrain_estimator->addContactPoint(con_pos_xy);
            }
        }

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

        // 5) Now apply a terrain correction from the new map.
        updateWithTerrain(state, kin.contacts_position, terrain_contacts_probability, kin.timestamp,
                          terrain_estimator);
    }

    Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
    T_world_to_base.translation() = state.base_position;
    T_world_to_base.linear() = state.base_orientation.toRotationMatrix();
    for (const auto& cf : contacts_frame_) {
        state.contacts_position.at(cf) = T_world_to_base * kin.contacts_position.at(cf);
        if (state.contacts_orientation.has_value() &&
            state.contacts_orientation.value().count(cf) > 0) {
            state.contacts_orientation.value().at(cf) =
                T_world_to_base.linear() * kin.contacts_orientation.value().at(cf);
        }
    }
}

// ---------------------------------------------------------------------------
// IMU orientation update  (world-measurement)
//
//   z = log(R_hat^T * R_y) ≈ ξ_R  (left-invariant)
//   H = [ I  0  0 | 0 ]
// ---------------------------------------------------------------------------
void LeftInvariantEKF::updateWithIMUOrientation(BaseState& state,
                                                const Eigen::Quaterniond& imu_orientation,
                                                const Eigen::Matrix3d& imu_orientation_cov,
                                                const double timestamp) {
    double dt = nominal_imu_dt_;
    if (last_imu_update_timestamp_.has_value()) {
        dt = timestamp - last_imu_update_timestamp_.value();
    }
    last_imu_update_timestamp_ = timestamp;

    if (dt <= 0.0 || !std::isfinite(dt)) {
        return;
    }

    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block(0, r_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    const Eigen::Vector3d z = lie::so3::logMap(state.base_orientation.inverse() * imu_orientation);
    const Eigen::Matrix3d N = imu_orientation_cov / dt;

    const Eigen::Matrix<double, 15, 3> PH_transpose = P_ * H.transpose();
    const Eigen::Matrix3d s = N + H * PH_transpose;
    const Eigen::Matrix<double, 15, 3> K = s.ldlt().solve(PH_transpose.transpose()).transpose();
    const Eigen::Matrix<double, 15, 1> dx = K * z;

    const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
    Eigen::Matrix<double, 15, 15> P_new;
    P_new.noalias() = IKH * P_ * IKH.transpose();
    P_new += K * N * K.transpose();
    P_ = P_new;
    updateState(state, dx, P_);
}

// ---------------------------------------------------------------------------
// Base linear velocity update  (world-frame measurement)
//
//   z = v_meas - v_hat ≈ R^T * ξ_v      (left-invariant)
//   H = [ 0   I   0 | 0   0 ]
//
// ---------------------------------------------------------------------------
void LeftInvariantEKF::updateWithBaseLinearVelocity(BaseState& state,
                                                    const Eigen::Vector3d& base_linear_velocity,
                                                    const Eigen::Matrix3d& base_linear_velocity_cov,
                                                    const double timestamp) {
    double dt = nominal_kin_dt_;
    if (last_kin_update_timestamp_.has_value()) {
        dt = timestamp - last_kin_update_timestamp_.value();
    }
    last_kin_update_timestamp_ = timestamp;

    if (dt <= 0.0 || !std::isfinite(dt)) {
        return;
    }
    const Eigen::Matrix3d R = state.base_orientation.toRotationMatrix();
    const Eigen::Matrix3d R_transpose = R.transpose();
    Eigen::Matrix<double, 3, 15> H = Eigen::Matrix<double, 3, 15>::Zero();
    H.block(0, v_idx_[0], 3, 3) = Eigen::Matrix3d::Identity();

    const Eigen::Vector3d z = R_transpose * (base_linear_velocity - state.base_linear_velocity);

    const Eigen::Matrix3d N = (R_transpose * base_linear_velocity_cov * R) / dt;
    const Eigen::Matrix<double, 15, 3> PH_transpose = P_ * H.transpose();
    const Eigen::Matrix3d s = N + H * PH_transpose;
    const Eigen::Matrix<double, 15, 3> K = s.ldlt().solve(PH_transpose.transpose()).transpose();
    const Eigen::Matrix<double, 15, 1> dx = K * z;

    const Eigen::Matrix<double, 15, 15> IKH = I_ - K * H;
    Eigen::Matrix<double, 15, 15> P_new;
    P_new.noalias() = IKH * P_ * IKH.transpose();
    P_new += K * N * K.transpose();
    P_ = P_new;
    updateState(state, dx, P_);
}

}  // namespace serow
