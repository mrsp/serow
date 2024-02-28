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
    State state(contacts_frame, config["point_feet"]);
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
    params_.gyro_cutoff_frequency = config["gyro_cutoff_frequency"];

    params_.force_torque_rate = config["force_torque_rate"];
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            params_.R_base_to_gyro(i, j) = config["R_base_to_gyro"][3 * i + j];
            params_.R_base_to_acc(i, j) = config["R_base_to_acc"][3 * i + j];
            size_t k = 0;
            for (const auto& frame : state_.getContactsFrame()) {
                params_.R_foot_to_force[frame](i, j) =
                    config["R_foot_to_force"][std::to_string(k)][3 * i + j];
                if (!config["R_foot_to_torque"].is_null()) {
                    params_.R_foot_to_torque[frame](i, j) =
                        config["R_foot_to_torque"][std::to_string(k)][3 * i + j];
                }
                k++;
            }
        }
    }

    // Joint parameters
    params_.joint_rate = config["joint_rate"];
    params_.estimate_joint_velocity = config["estimate_joint_velocity"];
    params_.joint_cutoff_frequency = config["joint_cutoff_frequency"];
    params_.joint_position_variance = config["joint_position_variance"];
    params_.angular_momentum_cutoff_frequency = config["angular_momentum_cutoff_frequency"];

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
    params_.outlier_detection = config["outlier_detection"];

    // Base/CoM estimation parameters
    params_.convergence_cycles = config["convergence_cycles"];
    for (size_t i = 0; i < 3; i++) {
        state.base_state_.imu_angular_velocity_bias[i] = config["bias_gyro"][i];
        state.base_state_.imu_linear_acceleration_bias[i] = config["bias_acc"][i];
        params_.angular_velocity_cov[i] = config["imu_angular_velocity_covariance"][i];
        params_.angular_velocity_bias_cov[i] = config["imu_angular_velocity_bias_covariance"][i];
        params_.linear_acceleration_cov[i] = config["imu_linear_acceleration_covariance"][i];
        params_.linear_acceleration_cov[i] = config["imu_linear_acceleration_bias_covariance"][i];
        params_.contact_position_cov[i] = config["contact_position_covariance"][i];
        if (!config["contact_orientation_covariance"].is_null()) {
            params_.contact_orientation_cov[i] = config["contact_orientation_covariance"][i];
        }
        params_.contact_position_slip_cov[i] = config["contact_position_slip_covariance"][i];
        if (!config["contact_orientation_slip_covariance"].is_null()) {
            params_.contact_orientation_slip_cov[i] =
                config["contact_orientation_slip_covariance"][i];
        }
        params_.com_position_process_cov[i] = config["com_position_process_covariance"][i];
        params_.com_linear_velocity_process_cov[i] =
            config["com_linear_velocity_process_covariance"][i];
        params_.external_forces_process_cov[i] = config["external_forces_process_covariance"][i];
        params_.com_position_cov[i] = config["com_position_covariance"][i];
        params_.com_linear_acceleration_cov[i] = config["com_linear_acceleration_covariance"][i];
        params_.initial_base_position_cov[i] = config["initial_base_position_covariance"][i];
        params_.initial_base_orientation_cov[i] = config["initial_base_orientation_covariance"][i];
        params_.initial_base_linear_velocity_cov[i] =
            config["initial_base_linear_velocity_covariance"][i];
        params_.initial_contact_position_cov[i] = config["initial_contact_position_covariance"][i];
        if (!config["initial_contact_orientation_covariance"].is_null()) {
            params_.initial_contact_orientation_cov[i] =
                config["initial_contact_orientation_covariance"][i];
        }
        params_.initial_imu_linear_acceleration_bias_cov[i] =
            config["initial_imu_linear_acceleration_bias_covariance"][i];
        params_.initial_imu_angular_velocity_bias_cov[i] =
            config["initial_imu_angular_velocity_bias_covariance"][i];
        params_.initial_com_position_cov[i] = config["initial_com_position_covariance"][i];
        params_.initial_com_linear_velocity_cov[i] =
            config["initial_com_linear_velocity_covariance"][i];
        params_.initial_external_forces_cov[i] = config["initial_external_forces_covariance"][i];
    }

    // External odometry extrinsics
    if (!config["T_base_to_odom"].is_null()) {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                params_.T_base_to_odom(i, j) = config["T_base_to_odom"][4 * i + j];
            }
        }
    }

    // Terrain height parameter
    params_.is_flat_terrain = config["is_flat_terrain"];
    params_.terrain_height_covariance = config["terrain_height_covariance"];

    // Initialize state uncertainty
    state_.base_state_.base_position_cov = params_.initial_base_position_cov.asDiagonal();
    state_.base_state_.base_orientation_cov = params_.initial_base_orientation_cov.asDiagonal();
    state_.base_state_.base_linear_velocity_cov =
        params_.initial_base_linear_velocity_cov.asDiagonal();
    state_.base_state_.imu_angular_velocity_bias_cov =
        params_.initial_imu_angular_velocity_bias_cov.asDiagonal();
    state_.base_state_.imu_linear_acceleration_bias_cov =
        params_.initial_imu_linear_acceleration_bias_cov.asDiagonal();
    std::unordered_map<std::string, Eigen::Matrix3d> contacts_orientation_cov;
    for (const auto& cf : state_.getContactsFrame()) {
        state_.base_state_.contacts_position_cov[cf] =
            params_.initial_contact_position_cov.asDiagonal();
        if (!state_.isPointFeet()) {
            contacts_orientation_cov[cf] = params_.initial_contact_orientation_cov.asDiagonal();
        }
    }
    if (!contacts_orientation_cov.empty()) {
        state_.base_state_.contacts_orientation_cov = std::move(contacts_orientation_cov);
    }
    state_.centroidal_state_.com_position_cov = params_.initial_com_position_cov.asDiagonal();
    state_.centroidal_state_.com_linear_velocity_cov =
        params_.initial_com_linear_velocity_cov.asDiagonal();
    state_.centroidal_state_.external_forces_cov = params_.initial_external_forces_cov.asDiagonal();
}

void Serow::filter(
    ImuMeasurement imu, std::unordered_map<std::string, JointMeasurement> joints,
    std::optional<std::unordered_map<std::string, ForceTorqueMeasurement>> ft,
    std::optional<OdometryMeasurement> odom,
    std::optional<std::unordered_map<std::string, ContactMeasurement>> contacts_probability) {
    if (!is_initialized_ && ft.has_value()) {
        is_initialized_ = true;
    } else if (!is_initialized_) {
        return;
    }

    // Safely copy the state prior to filtering
    State state(state_);

    // Estimate the joint velocities
    std::unordered_map<std::string, double> joints_position;
    std::unordered_map<std::string, double> joints_velocity;
    double joint_timestamp;
    for (const auto& [key, value] : joints) {
        joint_timestamp = value.timestamp;
        if (params_.estimate_joint_velocity && !joint_estimators_.count(key)) {
            joint_estimators_[key].init(key, params_.joint_rate, params_.joint_cutoff_frequency);
        }
        joints_position[key] = value.position;
        if (params_.estimate_joint_velocity && value.velocity.has_value()) {
            joints_velocity[key] = value.velocity.value();
        } else {
            joints_velocity[key] = joint_estimators_.at(key).filter(value.position);
        }
    }

    // Estimate the base frame attitude
    imu.angular_velocity = params_.R_base_to_gyro * imu.angular_velocity;
    imu.linear_acceleration = params_.R_base_to_acc * imu.linear_acceleration;
    attitude_estimator_->filter(imu.angular_velocity, imu.linear_acceleration);
    const Eigen::Matrix3d& R_world_to_base = attitude_estimator_->getR();

    // Estimate the imu bias in the base frame assumming that the robot is standing still
    if (params_.calibrate_imu && imu_calibration_cycles_ < params_.max_imu_calibration_cycles) {
        params_.bias_gyro += imu.angular_velocity;
        params_.bias_acc += imu.linear_acceleration -
                            R_world_to_base.transpose() * Eigen::Vector3d(0.0, 0.0, params_.g);
        imu_calibration_cycles_++;
        return;
    } else if (params_.calibrate_imu) {
        params_.bias_acc /= imu_calibration_cycles_;
        params_.bias_gyro /= imu_calibration_cycles_;
        params_.calibrate_imu = false;
        std::cout << "Calibration finished at " << imu_calibration_cycles_ << std::endl;
        std::cout << "Gyro biases " << params_.bias_gyro.transpose() << std::endl;
        std::cout << "Accelerometer biases " << params_.bias_acc.transpose() << std::endl;
        state.base_state_.imu_angular_velocity_bias = params_.bias_gyro;
        state.base_state_.imu_linear_acceleration_bias = params_.bias_acc;
    }

    // Update the kinematic structure
    kinematic_estimator_->updateJointConfig(joints_position, joints_velocity,
                                            params_.joint_position_variance);

    state.joint_state_.timestamp = joint_timestamp;
    state.joint_state_.joints_position = std::move(joints_position);
    state.joint_state_.joints_velocity = std::move(joints_velocity);

    // Get the CoM w.r.t the base frame
    const Eigen::Vector3d& base_to_com_position = kinematic_estimator_->comPosition();
    // Get the angular momentum around the CoM in base frame coordinates
    const Eigen::Vector3d& com_angular_momentum = kinematic_estimator_->comAngularMomentum();
    if (!angular_momentum_derivative_estimator) {
        angular_momentum_derivative_estimator = std::make_unique<DerivativeEstimator>();
        angular_momentum_derivative_estimator->init("CoM Angular Momentum Derivative",
                                                    params_.joint_rate,
                                                    params_.angular_momentum_cutoff_frequency);
    }
    // Estimate the angular momentum derivative
    const Eigen::Vector3d& com_angular_momentum_derivative =
        angular_momentum_derivative_estimator->filter(com_angular_momentum);

    // Get the leg end-effector kinematics w.r.t the base frame
    std::unordered_map<std::string, Eigen::Quaterniond> base_to_foot_orientations;
    std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_positions;
    std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_linear_velocities;
    std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_angular_velocities;

    for (const auto& contact_frame : state.getContactsFrame()) {
        base_to_foot_orientations[contact_frame] =
            kinematic_estimator_->linkOrientation(contact_frame);
        base_to_foot_positions[contact_frame] = kinematic_estimator_->linkPosition(contact_frame);
        base_to_foot_angular_velocities[contact_frame] =
            kinematic_estimator_->angularVelocity(contact_frame);
        base_to_foot_linear_velocities[contact_frame] =
            kinematic_estimator_->linearVelocity(contact_frame);
    }

    if (ft.has_value()) {
        std::unordered_map<std::string, Eigen::Vector3d> contacts_force;
        std::unordered_map<std::string, Eigen::Vector3d> contacts_torque;
        double den = state.num_leg_ee_ * params_.eps;

        for (const auto& frame : state.getContactsFrame()) {
            if (params_.estimate_contact_status && !contact_estimators_.count(frame)) {
                ContactDetector cd(frame, params_.high_threshold, params_.low_threshold,
                                   params_.mass, params_.g, params_.median_window);
                contact_estimators_[frame] = std::move(cd);
            }

            state.contact_state_.timestamp = ft->at(frame).timestamp;
            // Transform F/T to base frame
            contacts_force[frame] = base_to_foot_orientations.at(frame).toRotationMatrix() *
                                    params_.R_foot_to_force.at(frame) * ft->at(frame).force;
            if (!state.isPointFeet() && ft->at(frame).torque.has_value()) {
                contacts_torque[frame] = base_to_foot_orientations.at(frame).toRotationMatrix() *
                                         params_.R_foot_to_torque.at(frame) *
                                         ft->at(frame).torque.value();
            }

            // Estimate the contact status
            if (params_.estimate_contact_status) {
                contact_estimators_.at(frame).SchmittTrigger(contacts_force.at(frame).z());
                state.contact_state_.contacts_status[frame] =
                    contact_estimators_.at(frame).getContactStatus();
                den += contact_estimators_.at(frame).getContactForce();
            }
        }

        den /= state.num_leg_ee_;
        if (params_.estimate_contact_status && !contacts_probability.has_value()) {
            for (const auto& frame : state.getContactsFrame()) {
                // Estimate the contact quality
                state.contact_state_.contacts_probability[frame] = std::clamp(
                    (contact_estimators_.at(frame).getContactForce() + params_.eps) / den, 0.0,
                    1.0);
            }
        } else if (contacts_probability) {
            state.contact_state_.contacts_probability = std::move(contacts_probability.value());
            for (const auto& frame : state.getContactsFrame()) {
                state.contact_state_.contacts_status[frame] = false;
                if (state.contact_state_.contacts_probability.at(frame) > 0.5) {
                    state.contact_state_.contacts_status[frame] = true;
                }
            }
        }

        // Estimate the COP in the local foot frame
        for (const auto& frame : state.getContactsFrame()) {
            ft->at(frame).cop = Eigen::Vector3d::Zero();
            if (!state.isPointFeet() && contacts_torque.count(frame) &&
                state.contact_state_.contacts_probability.at(frame) > 0.0) {
                ft->at(frame).cop = Eigen::Vector3d(
                    -contacts_torque.at(frame).y() / contacts_force.at(frame).z(),
                    contacts_torque.at(frame).x() / contacts_force.at(frame).z(), 0.0);
            }
        }

        state.contact_state_.contacts_force = std::move(contacts_force);
        if (!contacts_torque.empty()) {
            state.contact_state_.contacts_torque = std::move(contacts_torque);
        }
    }

    // Compute the leg odometry and the contact points
    if (!leg_odometry_) {
        // Initialize the state
        state.base_state_.base_orientation = attitude_estimator_->getQ();
        state.centroidal_state_.com_position = base_to_com_position;
        state.base_state_.contacts_position = base_to_foot_positions;
        // Assuming the terrain is flat and the robot is initialized in a standing posture we can
        // have a measurement of the average terrain height constraining base estimation.
        if (params_.is_flat_terrain) {
            TerrainMeasurement tm{.height = 0.0, .height_cov = params_.terrain_height_covariance};
            terrain_ = std::move(tm);
            for (const auto& frame : state.getContactsFrame()) {
                terrain_->height += base_to_foot_positions.at(frame).z();
            }
            terrain_->height /= state.getContactsFrame().size();
        }
        if (!state.isPointFeet()) {
            state.base_state_.contacts_orientation = base_to_foot_orientations;
        }
        leg_odometry_ = std::make_unique<LegOdometry>(
            base_to_foot_positions, base_to_foot_orientations, params_.mass, params_.tau_0,
            params_.tau_1, params_.joint_rate, params_.g, params_.eps);
        base_estimator_.init(state.base_state_, state.getContactsFrame(), state.isPointFeet(),
                             params_.g, params_.imu_rate, params_.outlier_detection);
        com_estimator_.init(state.centroidal_state_, params_.mass, params_.g,
                            params_.force_torque_rate);
    }

    // Estimate the relative to the base frame contacts
    leg_odometry_->estimate(
        attitude_estimator_->getQ(),
        attitude_estimator_->getGyro() - attitude_estimator_->getR() * params_.bias_gyro,
        base_to_foot_orientations, base_to_foot_positions, base_to_foot_linear_velocities,
        base_to_foot_angular_velocities, state.contact_state_.contacts_force,
        state.contact_state_.contacts_torque);

    // Create the base estimation measurements
    imu.angular_velocity_cov = params_.angular_velocity_cov.asDiagonal();
    imu.angular_velocity_bias_cov = params_.angular_velocity_bias_cov.asDiagonal();
    imu.linear_acceleration_cov = params_.linear_acceleration_cov.asDiagonal();
    imu.linear_acceleration_bias_cov = params_.linear_acceleration_bias_cov.asDiagonal();

    KinematicMeasurement kin;
    kin.contacts_status = state.contact_state_.contacts_status;
    kin.contacts_probability = state.contact_state_.contacts_probability;
    kin.contacts_position = leg_odometry_->getContactPositions();
    kin.position_cov = params_.contact_position_cov.asDiagonal();
    kin.position_slip_cov = params_.contact_position_slip_cov.asDiagonal();

    if (!state.isPointFeet()) {
        kin.contacts_orientation = leg_odometry_->getContactOrientations();
        kin.orientation_cov = params_.contact_orientation_cov.asDiagonal();
        kin.orientation_slip_cov = params_.contact_orientation_slip_cov.asDiagonal();
    }

    std::unordered_map<std::string, Eigen::Matrix3d> kin_contacts_orientation_noise;
    for (const auto& frame : state.getContactsFrame()) {
        kin.contacts_position_noise[frame] =
            kinematic_estimator_->linearVelocity(frame) *
            kinematic_estimator_->linearVelocity(frame).transpose();
        if (!state.isPointFeet()) {
            kin_contacts_orientation_noise[frame] =
                kinematic_estimator_->angularVelocityNoise(frame) *
                kinematic_estimator_->angularVelocityNoise(frame).transpose();
        }
    }
    if (!state.isPointFeet()) {
        kin.contacts_orientation_noise = std::move(kin_contacts_orientation_noise);
    }

    // Call the base estimator predict step utilizing imu and contact status measurements
    state.base_state_.timestamp = imu.timestamp;
    state.base_state_ = base_estimator_.predict(state.base_state_, imu, kin);

    // Call the base estimator update step by employing relative contact pose, odometry and terrain
    // measurements
    if (odom.has_value()) {
        odom->base_position = params_.T_base_to_odom * odom->base_position;
        odom->base_orientation = Eigen::Quaterniond(params_.T_base_to_odom.linear() *
                                                    odom->base_orientation.toRotationMatrix());
        odom->base_position_cov = params_.T_base_to_odom.linear() * odom->base_position_cov *
                                  params_.T_base_to_odom.linear().transpose();
        odom->base_orientation_cov = params_.T_base_to_odom.linear() * odom->base_orientation_cov *
                                     params_.T_base_to_odom.linear().transpose();
    }
    state.base_state_.timestamp = kin.timestamp;
    state.base_state_ = base_estimator_.update(state.base_state_, kin, odom, terrain_);

    // Estimate the angular acceleration using the gyro velocity
    if (!gyro_derivative_estimator) {
        gyro_derivative_estimator = std::make_unique<DerivativeEstimator>();
        gyro_derivative_estimator->init("Gyro Derivative", params_.imu_rate,
                                        params_.gyro_cutoff_frequency);
    }
    imu.angular_acceleration =
        gyro_derivative_estimator->filter(imu.angular_velocity - state.getImuAngularVelocityBias());

    // Create the CoM estimation measurements
    kin.com_position = state.getBasePose() * base_to_com_position;
    kin.com_position_cov = params_.com_position_cov.asDiagonal();
    kin.com_angular_momentum_derivative =
        state.getBasePose().linear() * com_angular_momentum_derivative;

    // Approximate the CoM linear acceleration
    Eigen::Vector3d base_linear_acceleration =
        state.getBasePose().linear() *
            (imu.linear_acceleration - state.getImuLinearAccelerationBias()) -
        Eigen::Vector3d(0.0, 0.0, params_.g);
    Eigen::Vector3d base_angular_velocity =
        state.getBasePose().linear() * (imu.angular_velocity - state.getImuAngularVelocityBias());
    Eigen::Vector3d base_angular_acceleration =
        state.getBasePose().linear() * imu.angular_acceleration;
    kin.com_linear_acceleration = base_linear_acceleration;
    kin.com_linear_acceleration +=
        base_angular_velocity.cross(base_angular_velocity.cross(base_to_com_position)) +
        base_angular_acceleration.cross(base_to_com_position);

    // Update the state
    state.base_state_.base_linear_acceleration = base_linear_acceleration;
    state.base_state_.base_angular_velocity = base_angular_velocity;
    state.base_state_.base_angular_acceleration = base_angular_acceleration;
    state.centroidal_state_.angular_momentum = state.getBasePose().linear() * com_angular_momentum;
    state.centroidal_state_.angular_momentum_derivative = kin.com_angular_momentum_derivative;
    state.centroidal_state_.com_linear_acceleration = kin.com_linear_acceleration;
    if (ft.has_value()) {
        kin.com_position_process_cov = params_.com_position_process_cov.asDiagonal();
        kin.com_linear_velocity_process_cov = params_.com_linear_velocity_process_cov.asDiagonal();
        kin.external_forces_process_cov = params_.external_forces_process_cov.asDiagonal();
        kin.com_linear_acceleration_cov = params_.com_linear_acceleration_cov.asDiagonal();

        // Compute the COP and the total GRF in the world frame
        GroundReactionForceMeasurement grf;
        double den = 0.0;
        for (const auto& frame : state.getContactsFrame()) {
            grf.timestamp = ft->at(frame).timestamp;
            if (state.contact_state_.contacts_probability.at(frame) > 0.0) {
                grf.force += state.getContactPose(frame)->linear() * ft->at(frame).force;
                grf.cop += state.contact_state_.contacts_probability.at(frame) *
                           (*state.getContactPose(frame) * ft->at(frame).cop);
                den += state.contact_state_.contacts_probability.at(frame);
            }
        }
        if (den > 0.0) {
            grf.cop /= den;
        }
        state.centroidal_state_.cop_position = grf.cop;
        state.centroidal_state_.timestamp = grf.timestamp;
        // Call the CoM estimator predict step utilizing ground reaction measurements
        state.centroidal_state_ = com_estimator_.predict(state.centroidal_state_, kin, grf);
        // Call the CoM estimator update step by employing kinematic and imu measurements
        state.centroidal_state_ = com_estimator_.updateWithImu(state.centroidal_state_, kin, grf);
    }

    // Call the CoM estimator update step by employing kinematic measurements
    state.centroidal_state_.timestamp = kin.timestamp;
    state.centroidal_state_ = com_estimator_.updateWithKinematics(state.centroidal_state_, kin);

    // Check if the state has converged
    if (cycle_++ > params_.convergence_cycles) {
        state.is_valid_ = true;
    }

    // Safely copy the state post filtering
    state_ = std::move(state);
}

std::optional<State> Serow::getState(bool allow_invalid) {
    if (state_.is_valid_ || allow_invalid) {
        return state_;
    } else {
        return std::nullopt;
    }
}

}  // namespace serow
