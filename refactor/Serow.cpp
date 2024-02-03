/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "Serow.hpp"

#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace serow {


Serow::Serow(std::string config_file) {
    auto config = json::parse(std::ifstream(config_file));

    // Initialize the state
    std::unordered_set<std::string> contacts_frame;
    for (size_t i = 0; i < config["foot_frames"].size(); i++) {
        contacts_frame.insert({config["foot_frames"][std::to_string(i)]});
    }
    bool point_feet = config["point_feet"];
    State state(contacts_frame, point_feet);
    state_ = std::move(state);

    // Initialize the attitude estimator
    params_.imu_rate = config["imu_rate"];
    double Kp = config["attitude_estimator_proportional_gain"];
    double Ki = config["attitude_estimator_integral_gain"];
    attitude_estimator_ = std::make_unique<Mahony>(params_.imu_rate, Kp, Ki);

    // Initialize the kinematic estimator
    kinematic_estimator_ = std::make_unique<RobotKinematics>(config["model_path"]);
    params_.calibrate_imu = config["calibrate_imu"];
    if (!params_.calibrate_imu) {
        params_.bias_acc << config["bias_acc"][0], config["bias_acc"][1], config["bias_acc"][2];
        params_.bias_gyro << config["bias_gyro"][0], config["bias_gyro"][1], config["bias_gyro"][2];
    } else {
        params_.max_imu_calibration_cycles = config["max_imu_calibration_cycles"];
    }

    params_.force_torque_rate = config["force_torque_rate"];
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            params_.R_base_to_gyro(i, j) = config["R_base_to_gyro"][3 * i + j];
            params_.R_base_to_acc(i, j) = config["R_base_to_acc"][3 * i + j];
            size_t k = 0;
            for (const auto& frame : state_.getContactsFrame()) {
                params_.R_foot_to_force[frame](i, j) =
                    config["R_foot_to_force"][std::to_string(k)][3 * i + j];
                params_.R_foot_to_torque[frame](i, j) =
                    config["R_foot_to_torque"][std::to_string(k)][3 * i + j];
                k++;
            }
        }
    }

    // Joint parameters
    params_.joint_rate = config["joint_rate"];
    params_.estimate_joint_velocity = config["estimate_joint_velocity"];
    params_.joint_cutoff_frequency = config["joint_cutoff_frequency"];
    params_.joint_position_variance = config["joint_position_variance"];

    // Leg odometry perameters
    params_.mass = config["mass"];
    params_.g = config["g"];
    params_.tau_0 = config["tau_0"];
    params_.tau_1 = config["tau_1"];

    // Contact estimation parameters
    params_.estimate_contact_status = config["estimate_contact_status"];
    params_.high_threshold = config["high_threshold"];
    params_.low_threshold = config["low_threshold"];
    params_.median_window = config["median_window"];

    // Base/CoM estimation parameters
    for (size_t i = 0; i < 3; i++) {
        params_.angular_velocity_cov[i] = config["imu_angular_velocity_covariance"][i];
        params_.angular_velocity_bias_cov[i] = config["imu_angular_velocity_bias_covariance"][i];
        params_.linear_acceleration_cov[i] = config["imu_linear_acceleration_covariance"][i];
        params_.linear_acceleration_cov[i] = config["imu_linear_acceleration_bias_covariance"][i];
        params_.contact_position_cov[i] = config["contact_position_covariance"][i];
        params_.contact_orientation_cov[i] = config["contact_orientation_covariance"][i];
        params_.contact_position_slip_cov[i] = config["contact_position_slip_covariance"][i];
        params_.contact_orientation_slip_cov[i] = config["contact_orientation_slip_covariance"][i];
        params_.com_position_process_cov[i] = config["com_position_process_covariance"][i];
        params_.com_linear_velocity_process_cov[i] =
            config["com_linear_velocity_process_covariance"][i];
        params_.external_forces_process_cov[i] = config["external_forces_process_covariance"][i];
        params_.com_position_cov[i] = config["com_position_covariance"][i];
        params_.com_linear_acceleration_cov[i] = config["com_linear_acceleration_covariance"][i];
    }
}

void Serow::filter(ImuMeasurement imu, std::unordered_map<std::string, JointMeasurement> joints,
                   std::optional<std::unordered_map<std::string, ForceTorqueMeasurement>> ft,
                   std::optional<std::unordered_map<std::string, double>> contact_probabilities) {
    if (!is_initialized && ft.has_value()) {
        is_initialized = true;
    } else {
        return;
    }

    // Estimate the joint velocities
    std::unordered_map<std::string, double> joint_positions;
    std::unordered_map<std::string, double> joint_velocities;
    for (const auto& [key, value] : joints) {
        if (params_.estimate_joint_velocity && !joint_estimators_.count(key)) {
            joint_estimators_[key].init(key, params_.joint_rate, params_.joint_cutoff_frequency);
        }
        joint_positions[key] = value.position;
        if (params_.estimate_joint_velocity && value.velocity.has_value()) {
            joint_velocities[key] = value.velocity.value();
        } else {
            joint_velocities[key] = joint_estimators_.at(key).filter(value.position);
        }
    }

    // Estimate the base frame attitude
    Eigen::Vector3d wbb = params_.R_base_to_gyro * imu.angular_velocity;
    Eigen::Vector3d abb = params_.R_base_to_acc * imu.linear_acceleration;

    attitude_estimator_->filter(wbb, abb);
    const Eigen::Matrix3d& R_world_to_base = attitude_estimator_->getR();
    static int imu_calibration_cycles = 0;
    if (params_.calibrate_imu && imu_calibration_cycles < params_.max_imu_calibration_cycles) {
        params_.bias_gyro += wbb;
        params_.bias_acc += abb - R_world_to_base.transpose() * Eigen::Vector3d(0, 0, params_.g);
        imu_calibration_cycles++;
        return;
    } else if (params_.calibrate_imu) {
        params_.bias_acc /= imu_calibration_cycles;
        params_.bias_gyro /= imu_calibration_cycles;
        params_.calibrate_imu = false;
        std::cout << "Calibration finished at " << imu_calibration_cycles << std::endl;
        std::cout << "Gyro biases " << params_.bias_gyro.transpose() << std::endl;
        std::cout << "Accelerometer biases " << params_.bias_acc.transpose() << std::endl;
    }

    // Update the Kinematic Structure
    kinematic_estimator_->updateJointConfig(joint_positions, joint_velocities,
                                            params_.joint_position_variance);
    // Get the CoM w.r.t the base frame
    Eigen::Vector3d base_to_com_position = kinematic_estimator_->comPosition();
    
    // Get the leg end-effector kinematics w.r.t the base frame
    std::unordered_map<std::string, Eigen::Quaterniond> base_to_foot_orientations;
    std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_positions;
    std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_linear_velocities;
    std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_angular_velocities;

    for (const auto& contact_frame : state_.getContactsFrame()) {
        base_to_foot_orientations[contact_frame] =
            kinematic_estimator_->linkOrientation(contact_frame);
        base_to_foot_positions[contact_frame] = kinematic_estimator_->linkPosition(contact_frame);
        base_to_foot_angular_velocities[contact_frame] =
            kinematic_estimator_->getAngularVelocity(contact_frame);
        base_to_foot_linear_velocities[contact_frame] =
            kinematic_estimator_->getLinearVelocity(contact_frame);
    }

    if (ft.has_value()) {
        std::unordered_map<std::string, Eigen::Vector3d> contact_forces;
        std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contact_torques;
        double den = 2.0 * params_.eps;

        for (const auto& frame : state_.getContactsFrame()) {
            if (params_.estimate_contact_status && !contact_estimators_.count(frame)) {
                ContactDetector cd(frame, params_.high_threshold, params_.low_threshold,
                                   params_.mass, params_.g, params_.median_window);
                contact_estimators_[frame] = std::move(cd);
            }

            // Transform F/T to base frame
            contact_forces[frame] = base_to_foot_orientations.at(frame).toRotationMatrix() *
                                    params_.R_foot_to_force.at(frame) * ft->at(frame).force;
            if (!state_.point_feet_ && ft->at(frame).torque.has_value()) {
                contact_torques.value()[frame] = base_to_foot_orientations.at(frame).toRotationMatrix() *
                                         params_.R_foot_to_torque.at(frame) *
                                         ft->at(frame).torque.value();
            }

            if (params_.estimate_contact_status) {
                contact_estimators_.at(frame).SchmittTrigger(contact_forces.at(frame).z());
                state_.contacts_status_[frame] =  contact_estimators_.at(frame).getContactStatus();
                den += contact_estimators_.at(frame).getContactForce();
            }
        }

        if (params_.estimate_contact_status && !contact_probabilities.has_value()) {
            for (const auto& frame : state_.getContactsFrame()) {
                state_.contacts_probability_[frame] =
                    (contact_estimators_.at(frame).getContactForce() + params_.eps) / den;
            }
        } else if (contact_probabilities) {
            state_.contacts_probability_ = std::move(contact_probabilities.value());
            for (const auto& frame : state_.getContactsFrame()) {
                state_.contacts_status_[frame] = false;
                if (state_.contacts_probability_.at(frame) > 0.5) {
                    state_.contacts_status_[frame] = true;
                }
            }
        }

        if (!state_.point_feet_ && contact_torques.has_value()) {
            for (const auto& frame : state_.getContactsFrame()) {
                ft->at(frame).cop = Eigen::Vector3d(0.0, 0.0, 0.0);
                if (state_.contacts_probability_.at(frame) > 0.0) {
                    ft->at(frame).cop = Eigen::Vector3d(
                        -contact_torques.value().at(frame).y() / contact_forces.at(frame).z(),
                        contact_torques.value().at(frame).x() / contact_forces.at(frame).z(), 0.0);
                }
            }
            state_.contact_torques = std::move(contact_torques);
        }
        state_.contact_forces = std::move(contact_forces);
    }

    // Compute the leg odometry and the contact points
    if (!leg_odometry_) {
        state_.base_orientation_ = attitude_estimator_->getQ();
        state_.com_position_ = base_to_com_position;
        state_.contacts_position_ = base_to_foot_positions;
        if (!state_.point_feet_) {
            state_.contacts_orientation_ = base_to_foot_orientations;
        }
        leg_odometry_ = std::make_unique<LegOdometry>(
            base_to_foot_positions, base_to_foot_orientations, params_.mass, params_.tau_0,
            params_.tau_1, params_.joint_rate, params_.g);
        base_estimator_.init(state_, params_.imu_rate);
        com_estimator_.init(params_.mass, params_.force_torque_rate, 0.0, 0.0);
    }
    if (state_.contacts_probability_.empty()) {
        return;
    }

    leg_odometry_->estimate(
        attitude_estimator_->getQ(),
        attitude_estimator_->getGyro() - attitude_estimator_->getR() * params_.bias_gyro,
        base_to_foot_orientations, base_to_foot_positions, base_to_foot_linear_velocities,
        base_to_foot_angular_velocities, state_.contacts_probability_, state_.contact_forces,
        state_.contact_torques);

    // Create the base estimation measurements
    imu.angular_velocity_cov = params_.angular_velocity_cov.asDiagonal();
    imu.angular_velocity_bias_cov = params_.angular_velocity_bias_cov.asDiagonal();
    imu.linear_acceleration_cov = params_.linear_acceleration_cov.asDiagonal();
    imu.linear_acceleration_bias_cov = params_.linear_acceleration_bias_cov.asDiagonal();

    KinematicMeasurement kin;
    kin.contacts_status = state_.contacts_status_;
    kin.contacts_probability = state_.contacts_probability_;
    kin.contacts_position = leg_odometry_->getContactPositions();
    kin.contacts_orientation = leg_odometry_->getContactOrientations();
    kin.position_cov = params_.contact_position_cov.asDiagonal();
    kin.position_slip_cov = params_.contact_position_slip_cov.asDiagonal();
    if (!state_.point_feet_) {
        kin.orientation_cov = params_.contact_orientation_cov.asDiagonal();
        kin.orientation_slip_cov = params_.contact_orientation_slip_cov.asDiagonal();
    }

    std::unordered_map<std::string, Eigen::Matrix3d>  kin_contacts_orientation_noise;
    for (const auto& frame : state_.getContactsFrame()) {
        kin.contacts_position_noise[frame] =
            kinematic_estimator_->getLinearVelocityNoise(frame) *
            kinematic_estimator_->getLinearVelocityNoise(frame).transpose();
        if (!state_.point_feet_) {
            kin_contacts_orientation_noise.insert(
                {frame, kinematic_estimator_->getAngularVelocityNoise(frame) *
                            kinematic_estimator_->getAngularVelocityNoise(frame).transpose()});
        }
    }
    if (!state_.point_feet_) {
        kin.contacts_orientation_noise.emplace(std::move(kin_contacts_orientation_noise));
    }

    // Call the base estimator predict step utilizing imu and contact status measurements
    state_ = base_estimator_.predict(state_, imu, kin);
    // Call the base estimator update step by employing relative contact pose measurements
    state_ = base_estimator_.update(state_, kin);

    if (ft) {
        // Create the CoM estimation measurements
        kin.com_position = state_.getBasePose() * base_to_com_position;
        kin.com_position_cov = params_.com_position_cov.asDiagonal();
        kin.com_position_process_cov = params_.com_position_process_cov.asDiagonal();
        kin.com_linear_velocity_process_cov = params_.com_linear_velocity_process_cov.asDiagonal();
        kin.external_forces_process_cov = params_.external_forces_process_cov.asDiagonal();
        kin.com_linear_acceleration_cov = params_.com_linear_acceleration_cov.asDiagonal();
        GroundReactionForceMeasurement grf;
        double den = 0;
        for (const auto& frame : state_.getContactsFrame()) {
            grf.timestamp = ft->at(frame).timestamp;
            grf.force += state_.getBasePose() * ft->at(frame).force;
            grf.cop +=
                state_.contacts_probability_.at(frame) * (state_.getBasePose() * ft->at(frame).cop);
            den += state_.contacts_probability_.at(frame);
        }
        grf.cop /= den;

        // Call the CoM estimator predict step utilizing ground reaction measurements
        state_ = com_estimator_.predict(state_, kin, grf);
        // Call the CoM estimator update step by employing kinematic and imu measurements
        state_ = com_estimator_.updateWithImu(state_, kin, grf, imu);
    }
    // Call the CoM estimator update step by employing kinematic measurements
    state_ = com_estimator_.updateWithKinematics(state_, kin);
}


}  // namespace serow
