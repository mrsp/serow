/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "ContactEKF.hpp"
#include "lie.hpp"

ContactEKF::ContactEKF() {
    // Gravity Vector
    g_ = Eigen::Vector3d::Zero();
    g_(2) = -9.80;
}

void ContactEKF::init(State state) {
    num_leg_end_effectors_ = state.num_leg_ee_;
    contact_dim_ = state.point_feet_ ? 3 : 6;
    num_states_ = 15 + contact_dim_ * num_leg_end_effectors_;
    num_inputs_ = 12 + contact_dim_ * num_leg_end_effectors_;
    I_.setIdentity(num_states_, num_states_);

    // Initialize state indices
    v_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    r_idx_ = v_idx_ + 3;
    p_idx_ = r_idx_ + 3;
    bg_idx_ = p_idx_ + 3;
    ba_idx_ = bg_idx_ + 3;

    Eigen::Array3i pl_idx = ba_idx_ + 3;
    for (const auto& contact_frame : state.getContactsFrame()) {
        pl_idx_.insert({contact_frame, pl_idx});
        pl_idx += 3;
    }

    if (!state.point_feet_) {
        Eigen::Array3i rl_idx = pl_idx + 3;
        for (const auto& contact_frame : state.getContactsFrame()) {
            rl_idx_.insert({contact_frame, rl_idx});
            rl_idx += 3;
        }
    }

    ng_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    na_idx_ = ng_idx_ + 3;
    nbg_idx_ = na_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;

    Eigen::Array3i npl_idx = nba_idx_ + 3;
    for (const auto& contact_frame : state.getContactsFrame()) {
        npl_idx_.insert({contact_frame, npl_idx});
        npl_idx += 3;
    }
    
    if (!state.point_feet_) {
        Eigen::Array3i nrl_idx = npl_idx + 3;
        for (const auto& contact_frame : state.getContactsFrame()) {
            nrl_idx_.insert({contact_frame, nrl_idx});
            nrl_idx += 3;
        }
    }

    // Initialize the error state covariance
    P_ = I_;
    P_(v_idx_, v_idx_) = state.getBaseLinearVelocityCov();
    P_(r_idx_, r_idx_) = state.getBaseOrientationCov();
    P_(p_idx_, p_idx_) = state.getBasePositionCov();
    P_(bg_idx_, bg_idx_) = state.getImuAngularVelocityBiasCov();
    P_(ba_idx_, ba_idx_) = state.getImuLinearAccelerationBiasCov();

    for (const auto& contact_frame : state.getContactsFrame()) {
        if (state.getContactPositionCov(contact_frame)) {
            P_(pl_idx_.at(contact_frame), pl_idx_.at(contact_frame)) =
                state.getContactPositionCov(contact_frame).value();
        }
        if (!state.point_feet_ && state.getContactOrientationCov(contact_frame)) {
            P_(rl_idx_.at(contact_frame), rl_idx_.at(contact_frame)) =
                state.getContactOrientationCov(contact_frame).value();
        }
    }

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), gyro_bias (6), acc_bias (9), leg end effectors (12 - 12 + contact_dim * N)
    Lc_.setZero(num_states_, num_inputs_);
    Lc_(v_idx_, na_idx_) = -Matrix3d::Identity();
    Lc_(r_idx_, ng_idx_) = -Matrix3d::Identity();
    Lc_(bg_idx_, nbg_idx_) = Matrix3d::Identity();
    Lc_(ba_idx_, nba_idx_) = Matrix3d::Identity();

    for (const auto& contact_frame : state.getContactsFrame()) {
        Lc_(pl_idx_.at(contact_frame), npl_idx_.at(contact_frame)) = Matrix3d::Identity();
        if (!state.point_feet_) {
            Lc_(rl_idx_.at(contact_frame), nrl_idx_.at(contact_frame)) = Matrix3d::Identity();
        }
    }

    std::cout << "Contact EKF Initialized Successfully" << std::endl;
}

MatrixXd ContactEKF::computeDiscreteTransitionMatrix(State state, Vector3d angular_velocity,
                                                     Vector3d linear_acceleration, double dt) {
    angular_velocity -= state.getImuAngularVelocityBias();
    const Eigen::Vector3d& v = state.getBaseLinearVelocity();
    const Eigen::Matrix3d& R = state.getBaseOrientation().toRotationMatrix();

    Eigen::MatrixXd Ac, Ad;
    Ac.setZero(num_states_, num_states_);
    Ac.block<3, 3>(0, 0).noalias() = -lie::so3::wedge(angular_velocity);
    Ac.block<3, 3>(0, 3).noalias() = lie::so3::wedge(R.transpose() * g_);
    Ac.block<3, 3>(0, 12).noalias() = -Eigen::Matrix3d::Identity();
    Ac.block<3, 3>(0, 9).noalias() = -lie::so3::wedge(v);
    Ac.block<3, 3>(3, 3).noalias() = -lie::so3::wedge(angular_velocity);
    Ac.block<3, 3>(3, 9).noalias() = -Eigen::Matrix3d::Identity();
    Ac.block<3, 3>(6, 0) = R;
    Ac.block<3, 3>(6, 3).noalias() = -R * lie::so3::wedge(v);
    // Euler Discretization - First order Truncation
    Ad = I_;
    Ad += Ac * dt;
    return Ad;
}

State ContactEKF::predict(State state, ImuMeasurement imu, KinematicMeasurement kin) {
    double dt = nominal_dt_ + 0.1;
    if (last_timestamp_.has_value()) {
        dt = imu.timestamp - last_timestamp_.value();
    }

    const Eigen::MatrixXd& Ad =
        computeDiscreteTransitionMatrix(state, imu.angular_velocity, imu.linear_acceleration, dt);

    const State& predicted_state = computeDiscreteDynamics(
        state, dt, imu.angular_velocity, imu.linear_acceleration, kin.contacts_status,
        kin.contacts_position, kin.contacts_orientation);
    last_timestamp_ = imu.timestamp;
    return predicted_state;
}

State ContactEKF::computeDiscreteDynamics(
    State state, double dt, Eigen::Vector3d angular_velocity, Eigen::Vector3d linear_acceleration,
    std::optional<std::unordered_map<std::string, bool>> contacts_status,
    std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contacts_position,
    std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientations) {
    
    State predicted_state = state;
    angular_velocity -= state.getImuAngularVelocityBias();
    linear_acceleration -= state.getImuLinearAccelarationBias();

    // Nonlinear Process Model
    // Compute \dot{v}_b @ k
    const Eigen::Vector3d& v = state.getBaseLinearVelocity();
    const Eigen::Matrix3d& R = state.getBaseOrientation().toRotationMatrix();

    Eigen::Vector3d a = v.cross(angular_velocity);
    a += R.transpose() * g_;
    a += linear_acceleration;

    // Position
    const Eigen::Vector3d& r = state.getBasePosition();

    predicted_state.base_position_.noalias() = R * a * dt * dt / 2.00;
    predicted_state.base_position_ += R * v * dt;
    predicted_state.base_position_ += r;

    // Velocity
    a *= dt;
    a += v;
    predicted_state.base_linear_velocity_ = a;

    // Biases
    predicted_state.imu_angular_velocity_bias_ = state.imu_angular_velocity_bias_;
    predicted_state.imu_linear_acceleration_bias_ = state.imu_linear_acceleration_bias_;

    
    if (contacts_status.has_value() && contacts_position.has_value()) {
        for (auto [cf, cs] : contacts_status.value()) {
            if (contacts_position.value().count(cf)) {
                int contact_status = static_cast<int>(cs);
                predicted_state.contacts_position_[cf] =
                    contact_status * state.contacts_position_.at(cf) +
                    (1 - contact_status) * R * contacts_position.value().at(cf);
            }
        }
    }

    if (contacts_status.has_value() && contacts_orientations.has_value()) {
        for (auto [cf, cs] : contacts_status.value()) {
            if (contacts_orientations.value().count(cf)) {
                if (cs) {
                    predicted_state.contacts_orientation_.value().at(cf) =
                        state.contacts_orientation_.value().at(cf);
                } else {
                    predicted_state.contacts_orientation_.value().at(cf) = Eigen::Quaterniond(
                        R * contacts_orientations.value().at(cf).toRotationMatrix());
                }
            }
        }
    }

    predicted_state.base_orientation_ = Eigen::Quaterniond(
        lie::so3::plus(state.base_orientation_.toRotationMatrix(), angular_velocity));
    return predicted_state;
}

// State ContactEKF::updateContacts(
//     State state,
//     const std::unordered_map<std::string, Eigen::Vector3d>& contacts_position,
//     std::unordered_map<std::string, Eigen::Matrix3d> contacts_position_noise,
//     const std::unordered_map<std::string, double>& contacts_probability,
//     std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientation
//     std::optional<std::unordered_map<std::string, Eigen::Matrix3d>> contacts_orientation_noise) {
    
//     State updated_state = state;
//     Eigen::Matrix3d Rp = Eigen::Matrix3d::Zero();
//     Rp(0, 0) = lp_px * lp_px;
//     Rp(1, 1) = lp_py * lp_py;
//     Rp(2, 2) = lp_pz * lp_pz;
    
//     Eigen::Matrix3d Ro = Eigen::Matrix3d::Zero();
//     if (contacts_orientation_noise) {
//         Ro(0, 0) = lo_px * lo_px;
//         Ro(1, 1) = lo_py * lo_py;
//         Ro(2, 2) = lo_pz * lo_pz;
//     }

//     for (const auto& [cf, cp] : contacts_probability) {
//         int cs = cp > 0.5 ? 1 : 0;
//         contacts_position_noise.at(cf) =
//             cp * contacts_position_noise.at(cf) + (1 - cs) * Eigen::Matrix3d::Identity() * 1e4 + Rp;
//         if (contacts_orientation_noise) {
//             contacts_orientation_noise.at(cf) = cp * contacts_orientation_noise.at(cf) +
//                                                 (1 - cs) * Eigen::Matrix3d::Identity() * 1e4 + Ro;
//         }
//     }

//     for (const auto& [cf, cp] : contacts_position) {
//         Eigen::MatrixXd H;
//         H.setZero(3, num_states_);
//         const Eigen::Vector3d& x = state.getBaseOrientation.toRotationMatrix().transpose() *
//                                     (state.getFootPose(cf) - state.getBasePosition());
//         const Eigen::Vector3d& z = cp - x;
//         H.block(0, p_idx_) = -state.getBaseOrientation.toRotationMatrix().transpose();
//         H.block(0, pl_idx_.at(cf)) = state.getBaseOrientation.toRotationMatrix().transpose();
//         H.block(0, r_idx_) = lie::so3::wedge(x);
//         const Eigen::Matrix3d& s = contacts_position_noise.at(cf) + H * P_ * H.transpose();
//         const Eigen::MatrixXd& K = P * H.transpose() * s.inverse();
//         const Eigen::VectorXd& dx = K * z;

//         updated_state.base_position_ += dx(p_idx_);
//         updated_state.base_velocity += dx(v_idx_);
//         updated_state.base_orientation = Eigen::Quaternion(
//             lie::so3::plus(updated_state.base_orientation.toRotationMatrix(), dx(r_idx_)));
//         for (const auto& cf : state.contact_frames) {
//             // TODO: set orientation seperately
//             updated_state.foot_pose_.at(cf).translation() += dx(pl_idx_.at(cf));
//         }

//         P_ = (I_ - K * H) * P_ * (I_ - H.transpose() * K.transpose());
//         P_.noalias() += K * contacts_position_noise.at(cf) * K.transpose();
//     }

//     if (contacts_orientations.has_value()) {
//         for (const auto& [cf, co] : contacts_orientations.value()) {
//             Eigen::MatrixXd H;
//             H.setZero(3, num_states_);
//             const Eigen::Quaternion& x =
//                 Eigen::Quaternion(state.getFootPose(cf).value().linear().transpose() *
//                                   state.getBaseOrientation.toRotationMatrix());

//             const Eigen::Vector3d& z = so3::minus(co, x);
//             H.block(0, r_idx_, 3, 3) = -x;
//             H.block(0, plo_idx_, 3, 3) = Eigen::Matrix3d::Identity();
            
//             const Eigen::Matrix3d& s = contacts_position_noise.at(cf) + H * P_ * H.transpose();
//             const Eigen::MatrixXd& K = P * H.transpose() * s.inverse();
//             const Eigen::VectorXd& dx = K * z;

//             updated_state.base_position_ += dx(p_idx_);
//             updated_state.base_velocity += dx(v_idx_);
//             updated_state.base_orientation = Eigen::Quaternion(
//                 lie::so3::plus(updated_state.base_orientation.toRotationMatrix(), dx(r_idx_)));
//             for (const auto& cf : state.contact_frames) {
//                 // TODO: set orientation seperately
//                 updated_state.foot_pose_.at(cf).translation() += dx(pl_idx_.at(cf));
//             }

//             P_ = (I_ - K * H) * P_ * (I_ - H.transpose() * K.transpose());
//             P_.noalias() += K * contacts_position_noise.at(cf) * K.transpose();
//         }
//     }

//    return updated_state;
// }
