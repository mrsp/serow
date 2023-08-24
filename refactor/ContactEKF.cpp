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
    num_leg_end_effectors_ = state.getFootFrames().size();
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

    Eigen::ArrayXi pl_idx =
        Eigen::ArrayXi::LinSpaced(contact_dim_, ba_idx_(2), ba_idx_(2) + contact_dim_);
    for (const auto& foot_frame : state.getFootFrames()) {
        pl_idx_.insert({foot_frame, pl_idx});
        pl_idx += contact_dim_;
    }

    ng_idx_ = Eigen::Array3i::LinSpaced(0, 3);
    na_idx_ = ng_idx_ + 3;
    nbg_idx_ = na_idx_ + 3;
    nba_idx_ = nbg_idx_ + 3;
    Eigen::ArrayXi npl_idx =
        Eigen::ArrayXi::LinSpaced(contact_dim_, nba_idx_(2), nba_idx_(2) + contact_dim_);
    for (const auto& foot_frame : state.getFootFrames()) {
        npl_idx_.insert({foot_frame, npl_idx});
        npl_idx += contact_dim_;
    }

    // Initialize the error state covariance
    P_ = I_;
    P_(v_idx_, v_idx_) = state.getBaseLinearVelocityCov();
    P_(r_idx_, r_idx_) = state.getBaseOrientationCov();
    P_(p_idx_, p_idx_) = state.getBasePositionCov();
    P_(bg_idx_, bg_idx_) = state.getImuAngularVelocityBiasCov();
    P_(ba_idx_, ba_idx_) = state.getImuLinearAccelerationBiasCov();
    
    for (const auto& foot_frame : state.getFootFrames()) {
        if (state.getFootPoseCov(foot_frame)) {
            P_(pl_idx_.at(foot_frame), pl_idx_.at(foot_frame)) =
                state.getFootPoseCov(foot_frame).value();
        } else {
            std::cout << "Cannot read foot pose covariance for frame " << foot_frame << std::endl;
        }
    }

    // Compute some parts of the Input-Noise Jacobian once since they are constants
    // gyro (0), acc (3), gyro_bias (6), acc_bias (9), leg end effectors (12 - 12 + contact_dim * N)
    Lc_.setZero(num_states_, num_inputs_);
    Lc_(v_idx_, na_idx_) = -Matrix3d::Identity();
    Lc_(r_idx_, ng_idx_) = -Matrix3d::Identity();
    Lc_(bg_idx_, nbg_idx_) = Matrix3d::Identity();
    Lc_(ba_idx_, nba_idx_) = Matrix3d::Identity();

    for (const auto& foot_frame : state.getFootFrames()) {
        Lc_(pl_idx_.at(foot_frame), npl_idx_.at(foot_frame)) =
            MatrixXd::Identity(contact_dim_, contact_dim_);
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
    State predicted_state;
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
                predicted_state.foot_pose_[cf].translation() =
                    contact_status * state.foot_pose_.at(cf).translation() +
                    (1 - contact_status) * R * contacts_position.value().at(cf);
            }
        }
    }

    if (contacts_status.has_value() && contacts_orientations.has_value()) {
        for (auto [cf, cs] : contacts_status.value()) {
            if (contacts_orientations.value().count(cf)) {
                if (cs) {
                    predicted_state.foot_pose_.at(cf).linear() = state.foot_pose_.at(cf).linear();
                } else {
                    predicted_state.foot_pose_.at(cf).linear() =
                        R * contacts_orientations.value().at(cf).toRotationMatrix();
                }
            }
        }
    }

    predicted_state.base_orientation_ = Eigen::Quaterniond(
        lie::so3::plus(state.base_orientation_.toRotationMatrix(), angular_velocity));
    return predicted_state;
}
