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
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            params_.R_base_to_gyro(i, j) = config["R_base_to_gyro"][3 * i + j];
            params_.R_base_to_acc(i, j) = config["R_base_to_acc"][3 * i + j];
            size_t k = 0;
            for (const auto& frame : state_.getContactsFrame()) {
                params_.R_base_to_force[frame](i, j) =
                    config["R_base_to_force"][std::to_string(k)][3 * i + j];
                params_.R_base_to_torque[frame](i, j) =
                    config["R_base_to_torque"][std::to_string(k)][3 * i + j];
                k++;
            }
        }
    }

    params_.joint_rate = config["joint_rate"];
    params_.joint_cutoff_frequency = config["joint_cutoff_frequency"];

    params_.mass = config["mass"];
    params_.g = config["g"];
    params_.tau0 = config["tau_0"];
    params_.tau1 = config["tau_1"];
}

void Serow::filter(ImuMeasurement imu, std::unordered_map<std::string, JointMeasurement> joints,
                   std::unordered_map<std::string, ForceTorqueMeasurement> ft,
                   std::optional<std::unordered_map<std::string, double>> contact_probabilities) {
    // Estimate the joint velocities
    std::unordered_map<std::string, double> joint_positions;
    std::unordered_map<std::string, double> joint_velocities;
    for (const auto& [key, value] : joints) {
        if (!joint_estimators_.count(key)) {
            joint_estimators_[key].init(key, params_.joint_rate, params_.joint_cutoff_frequency);
        }
        joint_positions[key] = value.position;
        joint_velocities[key] = joint_estimators_.at(key).filter(value.position);
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

    std::unordered_map<std::string, Eigen::Vector3d> contact_forces;
    std::unordered_map<std::string, Eigen::Vector3d> contact_torques;

    if (params_.estimate_contact && contact_estimators_.empty()) {
        for (const auto& frame : state_.getContactsFrame()) {
            ContactDetector cd(frame, params_.high_threshold, params_.low_threshold,
                               params_.median_window);
            contact_estimators_[frame] = std::move(cd);
        }
    }

    // Transform F/T to base frame
    for (const auto& frame : state_.getContactsFrame()) {
        contact_forces[frame] = params_.R_base_to_force.at(frame) * ft.at(frame).force;
        contact_torques[frame] = params_.R_base_to_torque.at(frame) * ft.at(frame).torque;
        if (params_.estimate_contact) {
            contact_estimators_.at(frame).SchmittTrigger(contact_forces.at(frame).z());
            state_.contacts_status_.at(frame) = contact_estimators_.at(frame).getContactStatus();
        }
    }

    // Update the Kinematic Structure
    kinematic_estimator_->updateJointConfig(joint_positions, joint_velocities, 0.1);

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
        contact_forces[contact_frame] = ft.at(contact_frame).force;
        contact_torques[contact_frame] = ft.at(contact_frame).torque;
    }

    // Compute the leg odometry and the contact points
    if (!leg_odometry_) {
        // Initialize the state
        state_.base_orientation_ = attitude_estimator_->getQ();
        state_.com_position_ = base_to_com_position;
        leg_odometry_ = std::make_unique<LegOdometry>(
            base_to_foot_positions, base_to_foot_orientations, params_.mass, params_.tau0,
            params_.tau1, params_.joint_rate, params_.g);
        base_estimator_.init(state_, params_.imu_rate);
    }
    leg_odometry_->estimate(
        attitude_estimator_->getQ(),
        attitude_estimator_->getGyro() - attitude_estimator_->getR() * params_.bias_gyro,
        base_to_foot_orientations, base_to_foot_positions, base_to_foot_linear_velocities,
        base_to_foot_angular_velocities, *contact_probabilities, contact_forces, contact_torques);

    // Create the measurements
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
    for (const auto& frame : state_.getContactsFrame()) {
        kin.contacts_position_noise[frame] =
            kinematic_estimator_->getLinearVelocityNoise(frame) *
            kinematic_estimator_->getLinearVelocityNoise(frame).transpose();
        if (!state_.point_feet_) {
            kin.contacts_orientation_noise->emplace(
                frame, kinematic_estimator_->getAngularVelocityNoise(frame) *
                           kinematic_estimator_->getAngularVelocityNoise(frame).transpose());
        }
    }
    // Call the Base Estimator predict step utilizing imu and contact status measurements
    state_ = base_estimator_.predict(state_, imu, kin);
    // Call the Base Estimator update step by employing relative contact pose measurements
    state_ = base_estimator_.update(state_, kin);

}

}  // namespace serow
