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

#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace {
/// @brief Print color settings
static constexpr const char* RED_COLOR = "\033[31m";
static constexpr const char* WHITE_COLOR = "\033[0m";
}  // namespace

namespace serow {

std::string findFilepath(const std::string& filename) {
    const char* serow_path_env = std::getenv("SEROW_PATH");
    if (serow_path_env == nullptr) {
        throw std::runtime_error("Environmental variable SEROW_PATH is not set.");
    }

    std::filesystem::path serow_path(serow_path_env);
    for (const auto& entry : std::filesystem::recursive_directory_iterator(serow_path)) {
        if (std::filesystem::is_regular_file(entry) && entry.path().filename() == filename) {
            return entry.path().string();
        }
    }

    throw std::runtime_error("File '" + filename + "' not found.");
}

Serow::Serow() {
    threadpool_ = std::make_unique<ThreadPool>();
}

bool Serow::initialize(const std::string& config_file) {
    // Load configuration JSON
    auto config = json::parse(std::ifstream(findFilepath(config_file)));

    // Helper function to check and extract configuration values
    auto checkConfigParam = [&config](const std::string& param_name, auto& param_value) -> bool {
        using ValueType = std::decay_t<decltype(param_value)>;

        if constexpr (std::is_same_v<ValueType, bool>) {
            if (!config[param_name].is_boolean()) {
                std::cerr << RED_COLOR << "Configuration: " << param_name << " must be boolean\n"
                          << WHITE_COLOR;
                return false;
            }
        } else if constexpr (std::is_same_v<ValueType, double>) {
            if (!config[param_name].is_number_float()) {
                std::cerr << RED_COLOR << "Configuration: " << param_name << " must be float\n"
                          << WHITE_COLOR;
                return false;
            }
        } else if constexpr (std::is_same_v<ValueType, size_t> ||
                             std::is_same_v<ValueType, unsigned int>) {
            if (!config[param_name].is_number_unsigned()) {
                std::cerr << RED_COLOR << "Configuration: " << param_name << " must be integer\n"
                          << WHITE_COLOR;
                return false;
            }
        } else if constexpr (std::is_same_v<ValueType, std::string>) {
            if (!config[param_name].is_string()) {
                std::cerr << RED_COLOR << "Configuration: " << param_name << " must be string\n"
                          << WHITE_COLOR;
                return false;
            }
        }

        param_value = config[param_name];
        
        // Create a threadpool job to log data to mcap file
        
        return true;
    };

    auto checkConfigArray = [&config](const std::string& param_name, size_t expected_size) -> bool {
        if (!config[param_name].is_array() || config[param_name].size() != expected_size) {
            std::cerr << RED_COLOR << "Configuration: " << param_name << " must be an array of "
                      << expected_size << " elements\n"
                      << WHITE_COLOR;
            return false;
        }

        for (size_t i = 0; i < expected_size; i++) {
            if (!config[param_name][i].is_number_float()) {
                std::cerr << RED_COLOR << "Configuration: " << param_name << "[" << i
                          << "] must be float\n"
                          << WHITE_COLOR;
                return false;
            }
        }

        return true;
    };

    // Initialize contact frames
    std::set<std::string> contacts_frame;
    if (!config["foot_frames"].is_object()) {
        std::cerr << RED_COLOR << "Configuration: foot_frames must be an object\n" << WHITE_COLOR;
        return false;
    }

    for (size_t i = 0; i < config["foot_frames"].size(); i++) {
        std::string idx = std::to_string(i);
        if (!config["foot_frames"][idx].is_string()) {
            std::cerr << RED_COLOR << "Configuration: foot_frames[" << idx << "] must be a string\n"
                      << WHITE_COLOR;
            return false;
        }
        contacts_frame.insert(config["foot_frames"][idx]);
    }

    bool point_feet;
    if (!checkConfigParam("point_feet", point_feet))
        return false;

    // Initialize state
    State state(contacts_frame, point_feet);
    state_ = std::move(state);

    // Initialize attitude estimator
    if (!checkConfigParam("imu_rate", params_.imu_rate))
        return false;

    double Kp, Ki;
    if (!checkConfigParam("attitude_estimator_proportional_gain", Kp))
        return false;
    if (!checkConfigParam("attitude_estimator_integral_gain", Ki))
        return false;

    attitude_estimator_ = std::make_unique<Mahony>(params_.imu_rate, Kp, Ki);

    // Initialize kinematic estimator
    std::string model_path;
    if (!checkConfigParam("model_path", model_path))
        return false;
    kinematic_estimator_ = std::make_unique<RobotKinematics>(findFilepath(model_path));

    // IMU bias calibration
    if (!checkConfigParam("calibrate_initial_imu_bias", params_.calibrate_initial_imu_bias))
        return false;

    if (!params_.calibrate_initial_imu_bias) {
        if (!checkConfigArray("bias_acc", 3) || !checkConfigArray("bias_gyro", 3))
            return false;

        params_.bias_acc << config["bias_acc"][0], config["bias_acc"][1], config["bias_acc"][2];
        params_.bias_gyro << config["bias_gyro"][0], config["bias_gyro"][1], config["bias_gyro"][2];
    } else {
        if (!checkConfigParam("max_imu_calibration_cycles", params_.max_imu_calibration_cycles))
            return false;
    }

    // Load other scalar parameters
    if (!checkConfigParam("gyro_cutoff_frequency", params_.gyro_cutoff_frequency))
        return false;
    if (!checkConfigParam("force_torque_rate", params_.force_torque_rate))
        return false;
    if (!checkConfigParam("joint_rate", params_.joint_rate))
        return false;
    if (!checkConfigParam("estimate_joint_velocity", params_.estimate_joint_velocity))
        return false;
    if (!checkConfigParam("joint_cutoff_frequency", params_.joint_cutoff_frequency))
        return false;
    if (!checkConfigParam("joint_position_variance", params_.joint_position_variance))
        return false;
    if (!checkConfigParam("angular_momentum_cutoff_frequency",
                          params_.angular_momentum_cutoff_frequency))
        return false;
    if (!checkConfigParam("mass", params_.mass))
        return false;
    if (!checkConfigParam("g", params_.g))
        return false;
    if (!checkConfigParam("tau_0", params_.tau_0))
        return false;
    if (!checkConfigParam("tau_1", params_.tau_1))
        return false;
    if (!checkConfigParam("estimate_contact_status", params_.estimate_contact_status))
        return false;
    if (!checkConfigParam("high_threshold", params_.high_threshold))
        return false;
    if (!checkConfigParam("low_threshold", params_.low_threshold))
        return false;
    if (!checkConfigParam("median_window", params_.median_window))
        return false;
    if (!checkConfigParam("outlier_detection", params_.outlier_detection))
        return false;
    if (!checkConfigParam("convergence_cycles", params_.convergence_cycles))
        return false;
    if (!checkConfigParam("use_contacts_in_base_estimation", params_.is_contact_ekf))
        return false;
    if (!checkConfigParam("enable_terrain_estimation", params_.enable_terrain_estimation))
        return false;
    if (!checkConfigParam("minimum_terrain_height_variance",
                          params_.minimum_terrain_height_variance))
        return false;

    // Check rotation matrices
    if (!checkConfigArray("R_base_to_gyro", 9))
        return false;
    if (!checkConfigArray("R_base_to_acc", 9))
        return false;

    if (!config["R_foot_to_force"].is_object()) {
        std::cerr << RED_COLOR << "Configuration: R_foot_to_force must be an object\n"
                  << WHITE_COLOR;
        return false;
    }

    bool has_torque = !config["R_foot_to_torque"].is_null();
    if (has_torque && !config["R_foot_to_torque"].is_object()) {
        std::cerr << RED_COLOR << "Configuration: R_foot_to_torque must be an object\n"
                  << WHITE_COLOR;
        return false;
    }

    // Load matrices
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            params_.R_base_to_gyro(i, j) = config["R_base_to_gyro"][3 * i + j];
            params_.R_base_to_acc(i, j) = config["R_base_to_acc"][3 * i + j];

            size_t k = 0;
            for (const auto& frame : state_.getContactsFrame()) {
                std::string k_str = std::to_string(k);

                if (!config["R_foot_to_force"][k_str][3 * i + j].is_number_float()) {
                    std::cerr << RED_COLOR
                              << "Configuration: R_foot_to_force must be an array of floats\n"
                              << WHITE_COLOR;
                    return false;
                }
                params_.R_foot_to_force[frame](i, j) = config["R_foot_to_force"][k_str][3 * i + j];

                if (has_torque) {
                    if (!config["R_foot_to_torque"][k_str][3 * i + j].is_number_float()) {
                        std::cerr << RED_COLOR
                                  << "Configuration: R_foot_to_torque must be an array of floats\n"
                                  << WHITE_COLOR;
                        return false;
                    }
                    params_.R_foot_to_torque[frame](i, j) =
                        config["R_foot_to_torque"][k_str][3 * i + j];
                }
                k++;
            }
        }
    }

    // Check and load covariance arrays
    const std::vector<std::string> cov_arrays = {"imu_angular_velocity_covariance",
                                                 "imu_angular_velocity_bias_covariance",
                                                 "imu_linear_acceleration_covariance",
                                                 "imu_linear_acceleration_bias_covariance",
                                                 "base_linear_velocity_covariance",
                                                 "base_orientation_covariance",
                                                 "contact_position_covariance",
                                                 "com_position_process_covariance",
                                                 "com_linear_velocity_process_covariance",
                                                 "external_forces_process_covariance",
                                                 "com_position_covariance",
                                                 "com_linear_acceleration_covariance",
                                                 "initial_base_position_covariance",
                                                 "initial_base_orientation_covariance",
                                                 "initial_base_linear_velocity_covariance",
                                                 "initial_contact_position_covariance",
                                                 "initial_imu_linear_acceleration_bias_covariance",
                                                 "initial_imu_angular_velocity_bias_covariance",
                                                 "initial_com_position_covariance",
                                                 "initial_com_linear_velocity_covariance",
                                                 "initial_external_forces_covariance",
                                                 "contact_position_slip_covariance"};

    for (const auto& cov_name : cov_arrays) {
        if (!checkConfigArray(cov_name, 3))
            return false;
    }

    // Optional covariance arrays
    const std::vector<std::string> optional_cov_arrays = {"contact_orientation_covariance",
                                                          "contact_orientation_slip_covariance",
                                                          "initial_contact_orientation_covariance"};

    for (const auto& cov_name : optional_cov_arrays) {
        if (!config[cov_name].is_null() && !checkConfigArray(cov_name, 3))
            return false;
    }

    // Load covariance values
    auto loadCovarianceVector = [&config](const std::string& param_name,
                                          Eigen::Vector3d& cov_vector) {
        for (size_t i = 0; i < 3; i++) {
            cov_vector[i] = config[param_name][i];
        }
    };

    // Load bias values from configuration
    for (size_t i = 0; i < 3; i++) {
        state_.base_state_.imu_angular_velocity_bias[i] = config["bias_gyro"][i];
        state_.base_state_.imu_linear_acceleration_bias[i] = config["bias_acc"][i];
    }

    // Load all covariance vectors
    loadCovarianceVector("imu_angular_velocity_covariance", params_.angular_velocity_cov);
    loadCovarianceVector("imu_angular_velocity_bias_covariance", params_.angular_velocity_bias_cov);
    loadCovarianceVector("imu_linear_acceleration_covariance", params_.linear_acceleration_cov);
    loadCovarianceVector("imu_linear_acceleration_bias_covariance",
                         params_.linear_acceleration_bias_cov);
    loadCovarianceVector("base_linear_velocity_covariance", params_.base_linear_velocity_cov);
    loadCovarianceVector("base_orientation_covariance", params_.base_orientation_cov);
    loadCovarianceVector("contact_position_covariance", params_.contact_position_cov);
    loadCovarianceVector("contact_position_slip_covariance", params_.contact_position_slip_cov);
    loadCovarianceVector("com_position_process_covariance", params_.com_position_process_cov);
    loadCovarianceVector("com_linear_velocity_process_covariance",
                         params_.com_linear_velocity_process_cov);
    loadCovarianceVector("external_forces_process_covariance", params_.external_forces_process_cov);
    loadCovarianceVector("com_position_covariance", params_.com_position_cov);
    loadCovarianceVector("com_linear_acceleration_covariance", params_.com_linear_acceleration_cov);
    loadCovarianceVector("initial_base_position_covariance", params_.initial_base_position_cov);
    loadCovarianceVector("initial_base_orientation_covariance",
                         params_.initial_base_orientation_cov);
    loadCovarianceVector("initial_base_linear_velocity_covariance",
                         params_.initial_base_linear_velocity_cov);
    loadCovarianceVector("initial_contact_position_covariance",
                         params_.initial_contact_position_cov);
    loadCovarianceVector("initial_imu_linear_acceleration_bias_covariance",
                         params_.initial_imu_linear_acceleration_bias_cov);
    loadCovarianceVector("initial_imu_angular_velocity_bias_covariance",
                         params_.initial_imu_angular_velocity_bias_cov);
    loadCovarianceVector("initial_com_position_covariance", params_.initial_com_position_cov);
    loadCovarianceVector("initial_com_linear_velocity_covariance",
                         params_.initial_com_linear_velocity_cov);
    loadCovarianceVector("initial_external_forces_covariance", params_.initial_external_forces_cov);

    // Load optional orientation covariances if not null
    if (!config["contact_orientation_covariance"].is_null()) {
        loadCovarianceVector("contact_orientation_covariance", params_.contact_orientation_cov);
    }

    if (!config["contact_orientation_slip_covariance"].is_null()) {
        loadCovarianceVector("contact_orientation_slip_covariance",
                             params_.contact_orientation_slip_cov);
    }

    if (!config["initial_contact_orientation_covariance"].is_null()) {
        loadCovarianceVector("initial_contact_orientation_covariance",
                             params_.initial_contact_orientation_cov);
    }

    // External odometry extrinsics
    if (!config["T_base_to_odom"].is_null()) {
        if (!config["T_base_to_odom"].is_array() || config["T_base_to_odom"].size() != 16) {
            std::cerr << RED_COLOR
                      << "Configuration: T_base_to_odom must be an array of 16 elements\n"
                      << WHITE_COLOR;
            return false;
        }

        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                if (!config["T_base_to_odom"][4 * i + j].is_number_float()) {
                    std::cerr << RED_COLOR << "Configuration: T_base_to_odom[" << 4 * i + j
                              << "] must be float\n"
                              << WHITE_COLOR;
                    return false;
                }
                params_.T_base_to_odom(i, j) = config["T_base_to_odom"][4 * i + j];
            }
        }
    }

    // Initialize state uncertainty
    state_.mass_ = params_.mass;
    state_.base_state_.base_position_cov = params_.initial_base_position_cov.asDiagonal();
    state_.base_state_.base_orientation_cov = params_.initial_base_orientation_cov.asDiagonal();
    state_.base_state_.base_linear_velocity_cov =
        params_.initial_base_linear_velocity_cov.asDiagonal();
    state_.base_state_.imu_angular_velocity_bias_cov =
        params_.initial_imu_angular_velocity_bias_cov.asDiagonal();
    state_.base_state_.imu_linear_acceleration_bias_cov =
        params_.initial_imu_linear_acceleration_bias_cov.asDiagonal();

    // Initialize contact covariances
    std::map<std::string, Eigen::Matrix3d> contacts_orientation_cov;
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

    // Initialize centroidal state
    state_.centroidal_state_.com_position_cov = params_.initial_com_position_cov.asDiagonal();
    state_.centroidal_state_.com_linear_velocity_cov =
        params_.initial_com_linear_velocity_cov.asDiagonal();
    state_.centroidal_state_.external_forces_cov = params_.initial_external_forces_cov.asDiagonal();

    std::cout << "Configuration initialized" << std::endl;
    return true;
}

void Serow::filter(ImuMeasurement imu, std::map<std::string, JointMeasurement> joints,
                   std::optional<std::map<std::string, ForceTorqueMeasurement>> ft,
                   std::optional<OdometryMeasurement> odom,
                   std::optional<std::map<std::string, ContactMeasurement>> contacts_probability) {
    // Early return if not initialized and no FT measurement
    if (!is_initialized_) {
        if (!ft.has_value())
            return;
        is_initialized_ = true;
    }

    // Use move semantics for state copy
    State state = std::move(State(state_));

    // Check if foot frames exist on the F/T measurement
    if (ft.has_value()) {
        for (const auto& frame : state.contacts_frame_) {
            if (ft.value().count(frame) == 0) {
                throw std::runtime_error("Foot frame <" + frame +
                                         "> does not exist in the force measurements");
            }
        }
    }

    // Estimate joint velocities
    std::map<std::string, double> joints_position;
    std::map<std::string, double> joints_velocity;

    double joint_timestamp{};
    for (const auto& [key, value] : joints) {
        joints_position[key] = value.position;
        joint_timestamp = value.timestamp;

        if (!params_.estimate_joint_velocity) {
            if (!value.velocity.has_value()) {
                throw std::runtime_error(
                    "No joint velocities found, either provide them or enable the "
                    "estimate_joint_velocities parameter");
            }
            joints_velocity[key] = value.velocity.value();
        } else {
            if (joint_estimators_.count(key) == 0) {
                joint_estimators_.emplace(key,
                                          DerivativeEstimator(key, params_.joint_rate,
                                                              params_.joint_cutoff_frequency, 1));
            }
            joints_velocity[key] =
                joint_estimators_.at(key).filter(Eigen::Matrix<double, 1, 1>(value.position))(0);
        }
    }

    // Transform IMU measurements to base frame
    imu.angular_velocity = params_.R_base_to_gyro * imu.angular_velocity;
    imu.linear_acceleration = params_.R_base_to_acc * imu.linear_acceleration;

    // Estimate the base frame attitude
    attitude_estimator_->filter(imu.angular_velocity, imu.linear_acceleration);
    const Eigen::Matrix3d& R_world_to_base = attitude_estimator_->getR();

    // IMU bias calibration
    if (params_.calibrate_initial_imu_bias) {
        if (imu_calibration_cycles_ < params_.max_imu_calibration_cycles) {
            params_.bias_gyro += imu.angular_velocity;
            params_.bias_acc.noalias() += imu.linear_acceleration -
                R_world_to_base.transpose() * Eigen::Vector3d(0.0, 0.0, params_.g);
            imu_calibration_cycles_++;
            return;
        } else {
            // Only divide if we've actually accumulated samples
            if (imu_calibration_cycles_ > 0) {
                params_.bias_acc /= imu_calibration_cycles_;
                params_.bias_gyro /= imu_calibration_cycles_;
                params_.calibrate_initial_imu_bias = false;

                // Update state with calibrated biases
                state.base_state_.imu_angular_velocity_bias = params_.bias_gyro;
                state.base_state_.imu_linear_acceleration_bias = params_.bias_acc;

                std::cout << "Calibration finished at " << imu_calibration_cycles_ << std::endl;
                std::cout << "Gyrometer biases " << params_.bias_gyro.transpose() << std::endl;
                std::cout << "Accelerometer biases " << params_.bias_acc.transpose() << std::endl;
            }
        }
    }

    // Update the kinematic structure
    kinematic_estimator_->updateJointConfig(joints_position, joints_velocity,
                                            params_.joint_position_variance);

    // Update the joint state estimate - use std::move to avoid copies
    state.joint_state_.timestamp = joint_timestamp;
    state.joint_state_.joints_position = std::move(joints_position);
    state.joint_state_.joints_velocity = std::move(joints_velocity);

    // Get the CoM w.r.t the base frame as computed with rigid-body kinematics
    const Eigen::Vector3d& base_to_com_position = kinematic_estimator_->comPosition();
    // Get the angular momentum around the CoM in base frame coordinates as compute with rigid-body
    // kinematics
    const Eigen::Vector3d& com_angular_momentum = kinematic_estimator_->comAngularMomentum();

    // Initialize angular momentum derivative estimator if needed
    if (!angular_momentum_derivative_estimator) {
        angular_momentum_derivative_estimator = std::make_unique<DerivativeEstimator>(
            "CoM Angular Momentum Derivative", params_.joint_rate,
            params_.angular_momentum_cutoff_frequency, 3);
    }

    // Estimate the angular momentum derivative around the CoM in base frame
    const Eigen::Vector3d& com_angular_momentum_derivative =
        angular_momentum_derivative_estimator->filter(com_angular_momentum);

    // Preallocate maps for leg end-effector kinematics
    std::map<std::string, Eigen::Vector3d> base_to_foot_positions;
    std::map<std::string, Eigen::Quaterniond> base_to_foot_orientations;
    std::map<std::string, Eigen::Vector3d> base_to_foot_linear_velocities;
    std::map<std::string, Eigen::Vector3d> base_to_foot_angular_velocities;

    // Get the leg end-effector kinematics - reuse keys for efficiency
    for (const auto& contact_frame : state.getContactsFrame()) {
        base_to_foot_orientations[contact_frame] =
            kinematic_estimator_->linkOrientation(contact_frame);
        base_to_foot_positions[contact_frame] = kinematic_estimator_->linkPosition(contact_frame);
        base_to_foot_angular_velocities[contact_frame] =
            kinematic_estimator_->angularVelocity(contact_frame);
        base_to_foot_linear_velocities[contact_frame] =
            kinematic_estimator_->linearVelocity(contact_frame);
    }

    // Estimate the leg end-effector contact state
    if (ft.has_value()) {
        std::map<std::string, Eigen::Vector3d> contacts_force;
        std::map<std::string, Eigen::Vector3d> contacts_torque;

        double den = state.num_leg_ee_ * params_.eps;

        for (const auto& frame : state.getContactsFrame()) {
            // Create contact estimators if needed
            if (params_.estimate_contact_status) {
                if (contact_estimators_.count(frame) == 0) {
                    contact_estimators_.emplace(
                        frame,
                        ContactDetector(frame, params_.high_threshold, params_.low_threshold,
                                        params_.mass, params_.g, params_.median_window));
                }
            }

            state.contact_state_.timestamp = ft->at(frame).timestamp;

            // Transform F/T to base frame
            const auto& foot_orientation = base_to_foot_orientations.at(frame);
            const auto& frame_force = ft->at(frame).force;
            const auto& R_foot_to_force = params_.R_foot_to_force.at(frame);

            // Cache rotation matrix calculations
            const Eigen::Matrix3d combined_rotation =
                R_world_to_base * foot_orientation.toRotationMatrix() * R_foot_to_force;
            contacts_force[frame].noalias() = combined_rotation * frame_force;

            // Process torque if not point feet
            if (!state.isPointFeet() && ft->at(frame).torque.has_value()) {
                contacts_torque[frame].noalias() = foot_orientation.toRotationMatrix() *
                    params_.R_foot_to_torque.at(frame) * ft->at(frame).torque.value();
            }

            // Estimate the contact status
            if (params_.estimate_contact_status) {
                contact_estimators_.at(frame).SchmittTrigger(contacts_force.at(frame).z());
                state.contact_state_.contacts_status[frame] =
                    contact_estimators_.at(frame).getContactStatus();
                den += contact_estimators_.at(frame).getContactForce();
            }
        }

        // Process contact probability
        if (params_.estimate_contact_status && !contacts_probability.has_value()) {
            den /= state.num_leg_ee_;
            for (const auto& frame : state.getContactsFrame()) {
                // Use std::clamp for bounds checking
                state.contact_state_.contacts_probability[frame] = std::clamp(
                    (contact_estimators_.at(frame).getContactForce() + params_.eps) / den, 0.0,
                    1.0);
            }
        } else if (contacts_probability) {
            state.contact_state_.contacts_probability = std::move(contacts_probability.value());

            // Compute binary contact status
            for (const auto& frame : state.getContactsFrame()) {
                state.contact_state_.contacts_status[frame] =
                    state.contact_state_.contacts_probability.at(frame) > 0.5;
            }
        }

        // Estimate the COP in the local foot frame
        for (const auto& frame : state.getContactsFrame()) {
            ft->at(frame).cop = Eigen::Vector3d::Zero();

            // Calculate COP
            if (!state.isPointFeet() && contacts_torque.count(frame) &&
                state.contact_state_.contacts_probability.at(frame) > 0.0) {
                const double z_force = contacts_force.at(frame).z();
                if (std::abs(z_force) > 1e-6) {  // Avoid division by near-zero
                    ft->at(frame).cop =
                        Eigen::Vector3d(-contacts_torque.at(frame).y() / z_force,
                                        contacts_torque.at(frame).x() / z_force, 0.0);
                }
            }
        }

        // Move contacts data to state
        state.contact_state_.contacts_force = std::move(contacts_force);
        if (!contacts_torque.empty()) {
            state.contact_state_.contacts_torque = std::move(contacts_torque);
        }
    }


    // Cache frequently used values
    const Eigen::Quaterniond& attitude_q = attitude_estimator_->getQ();
    const Eigen::Vector3d& attitude_gyro = attitude_estimator_->getGyro();
    const Eigen::Matrix3d& attitude_R = attitude_estimator_->getR();

    // Initialize leg odometry if needed
    if (!leg_odometry_) {
        // Initialize the state
        state.base_state_.base_orientation = attitude_q;
        state.centroidal_state_.com_position = base_to_com_position;
        state.base_state_.contacts_position = base_to_foot_positions;

        if (!state.isPointFeet()) {
            state.base_state_.contacts_orientation = base_to_foot_orientations;
        }

        // Create leg odometry
        leg_odometry_ = std::make_unique<LegOdometry>(
            base_to_foot_positions, base_to_foot_orientations, params_.mass, params_.tau_0,
            params_.tau_1, params_.joint_rate, params_.g, params_.eps);

        // Initialize the base and CoM estimators
        if (params_.is_contact_ekf) {
            base_estimator_con_.init(state.base_state_, state.getContactsFrame(),
                                     state.isPointFeet(), params_.g, params_.imu_rate,
                                     params_.outlier_detection);
        } else {
            base_estimator_.init(state.base_state_, params_.g, params_.imu_rate,
                                 params_.outlier_detection);
        }

        com_estimator_.init(state.centroidal_state_, params_.mass, params_.g,
                            params_.force_torque_rate);
    }

    // Perform leg odometry estimation
    leg_odometry_->estimate(
        attitude_q, attitude_gyro - attitude_R * params_.bias_gyro, base_to_foot_orientations,
        base_to_foot_positions, base_to_foot_linear_velocities, base_to_foot_angular_velocities,
        state.contact_state_.contacts_force, state.contact_state_.contacts_probability,
        state.contact_state_.contacts_torque);

    // Create the base estimation measurements
    imu.angular_velocity_cov = params_.angular_velocity_cov.asDiagonal();
    imu.angular_velocity_bias_cov = params_.angular_velocity_bias_cov.asDiagonal();
    imu.linear_acceleration_cov = params_.linear_acceleration_cov.asDiagonal();
    imu.linear_acceleration_bias_cov = params_.linear_acceleration_bias_cov.asDiagonal();

    // Prepare kinematic measurement
    KinematicMeasurement kin;
    kin.timestamp = joint_timestamp;
    kin.contacts_status = state.contact_state_.contacts_status;
    kin.contacts_probability = state.contact_state_.contacts_probability;
    kin.base_linear_velocity = leg_odometry_->getBaseLinearVelocity();
    kin.base_linear_velocity_cov = params_.base_linear_velocity_cov.asDiagonal();
    kin.contacts_position = leg_odometry_->getContactPositions();
    kin.position_cov = params_.contact_position_cov.asDiagonal();
    kin.position_slip_cov = params_.contact_position_slip_cov.asDiagonal();
    kin.base_orientation = attitude_q;
    kin.base_orientation_cov = params_.base_orientation_cov.asDiagonal();
    kin.base_to_foot_positions = base_to_foot_positions;

    // Initialize terrain estimator if needed
    if (params_.enable_terrain_estimation && !terrain_estimator_ && params_.is_contact_ekf) {
        float terrain_height = 0.0;
        int i = 0;

        for (const auto& [cf, cp] : kin.contacts_status) {
            if (cp) {
                i++;
                terrain_height += kin.contacts_position.at(cf).z();
            }
        }

        if (i > 0) {
            terrain_height /= i;
        }

        // Initialize terrain elevation mapper
        terrain_estimator_ = std::make_shared<TerrainElevation>();
        terrain_estimator_->initializeLocalMap(terrain_height, 1e4);
        terrain_estimator_->min_terrain_height_variance_ = params_.minimum_terrain_height_variance;
    }

    // Handle orientation for non-point feet
    if (!state.isPointFeet()) {
        kin.contacts_orientation = leg_odometry_->getContactOrientations();
        kin.orientation_cov = params_.contact_orientation_cov.asDiagonal();
        kin.orientation_slip_cov = params_.contact_orientation_slip_cov.asDiagonal();
    }

    // Compute orientation noise for contacts
    std::map<std::string, Eigen::Matrix3d> kin_contacts_orientation_noise;
    for (const auto& frame : state.getContactsFrame()) {
        const Eigen::Vector3d& lin_vel = kinematic_estimator_->linearVelocity(frame);
        kin.contacts_position_noise[frame].noalias() = lin_vel * lin_vel.transpose();

        if (!state.isPointFeet()) {
            const Eigen::Vector3d& ang_vel = kinematic_estimator_->angularVelocityNoise(frame);
            kin_contacts_orientation_noise[frame].noalias() = ang_vel * ang_vel.transpose();
        }
    }

    if (!state.isPointFeet()) {
        kin.contacts_orientation_noise = std::move(kin_contacts_orientation_noise);
    }

    // Call the base estimator predict step
    state.base_state_.timestamp = imu.timestamp;
    if (params_.is_contact_ekf) {
        state.base_state_ = base_estimator_con_.predict(state.base_state_, imu, kin);
    } else {
        state.base_state_ = base_estimator_.predict(state.base_state_, imu);
    }

    // Transform odometry measurements if available
    if (odom.has_value()) {
        odom->base_position = params_.T_base_to_odom * odom->base_position;

        // Cache matrix multiplication
        const Eigen::Matrix3d R_base_to_odom = params_.T_base_to_odom.linear();
        odom->base_orientation =
            Eigen::Quaterniond(R_base_to_odom * odom->base_orientation.toRotationMatrix());

        // Calculate covariance
        odom->base_position_cov =
            R_base_to_odom * odom->base_position_cov * R_base_to_odom.transpose();
        odom->base_orientation_cov =
            R_base_to_odom * odom->base_orientation_cov * R_base_to_odom.transpose();
    }

    state.base_state_.timestamp = kin.timestamp;

    // Update base state with relative to base contacts
    if (params_.is_contact_ekf) {
        state.base_state_ =
            base_estimator_con_.update(state.base_state_, kin, odom, terrain_estimator_);
    } else {
        state.base_state_ = base_estimator_.update(state.base_state_, kin, odom);

        // Compute the contact pose in the world frame
        std::map<std::string, Eigen::Quaterniond> con_orient;
        const Eigen::Isometry3d& base_pose = state.getBasePose();

        for (const auto& frame : state.getContactsFrame()) {
            state.base_state_.contacts_position[frame].noalias() =
                base_pose * kin.contacts_position.at(frame);

            if (!state.isPointFeet()) {
                con_orient[frame] = Eigen::Quaterniond(
                    base_pose.linear() * kin.contacts_orientation->at(frame).toRotationMatrix());
            }
        }

        if (!state.isPointFeet()) {
            state.base_state_.contacts_orientation = std::move(con_orient);
        }
    }

    // Estimate the imu angular acceleration using the gyro
    if (!gyro_derivative_estimator) {
        gyro_derivative_estimator = std::make_unique<DerivativeEstimator>(
            "Gyro Derivative", params_.imu_rate, params_.gyro_cutoff_frequency, 3);
    }

    // Estimate angular acceleration
    imu.angular_acceleration =
        gyro_derivative_estimator->filter(imu.angular_velocity - state.getImuAngularVelocityBias());

    // Prepare CoM estimation measurements
    kin.com_position.noalias() = state.getBasePose() * base_to_com_position;
    kin.com_position_cov = params_.com_position_cov.asDiagonal();
    kin.com_angular_momentum_derivative.noalias() =
        state.getBasePose().linear() * com_angular_momentum_derivative;

    // Approximate the CoM linear acceleration in the world frame
    const Eigen::Isometry3d& base_pose = state.getBasePose();
    const Eigen::Vector3d base_linear_acceleration =
        base_pose.linear() * (imu.linear_acceleration - state.getImuLinearAccelerationBias()) -
        Eigen::Vector3d(0.0, 0.0, params_.g);
    const Eigen::Vector3d base_angular_velocity =
        base_pose.linear() * (imu.angular_velocity - state.getImuAngularVelocityBias());

    const Eigen::Vector3d base_angular_acceleration = base_pose.linear() * imu.angular_acceleration;
    kin.com_linear_acceleration.noalias() = base_linear_acceleration +
        base_angular_velocity.cross(base_angular_velocity.cross(base_to_com_position)) +
        base_angular_acceleration.cross(base_to_com_position);

    // Update the state
    state.base_state_.base_linear_acceleration = base_linear_acceleration;
    state.base_state_.base_angular_velocity = base_angular_velocity;
    state.base_state_.base_angular_acceleration = base_angular_acceleration;
    state.centroidal_state_.angular_momentum.noalias() = base_pose.linear() * com_angular_momentum;
    state.centroidal_state_.angular_momentum_derivative = kin.com_angular_momentum_derivative;
    state.centroidal_state_.com_linear_acceleration = kin.com_linear_acceleration;

    // Update feet pose/velocity in world frame
    for (const auto& frame : state.getContactsFrame()) {
        // Cache calculations
        const Eigen::Vector3d& base_foot_pos = base_to_foot_positions.at(frame);
        const Eigen::Vector3d transformed_pos = base_pose.linear() * base_foot_pos;

        state.base_state_.feet_position[frame].noalias() = base_pose * base_foot_pos;

        state.base_state_.feet_orientation[frame] = Eigen::Quaterniond(
            base_pose.linear() * base_to_foot_orientations.at(frame).toRotationMatrix());

        state.base_state_.feet_linear_velocity[frame].noalias() =
            state.base_state_.base_linear_velocity +
            state.base_state_.base_angular_velocity.cross(transformed_pos) +
            base_pose.linear() * base_to_foot_linear_velocities.at(frame);

        state.base_state_.feet_angular_velocity[frame].noalias() =
            state.base_state_.base_angular_velocity +
            base_pose.linear() * base_to_foot_angular_velocities.at(frame);
    }

    // Process force-torque measurements if available
    if (ft.has_value()) {
        kin.com_position_process_cov = params_.com_position_process_cov.asDiagonal();
        kin.com_linear_velocity_process_cov = params_.com_linear_velocity_process_cov.asDiagonal();
        kin.external_forces_process_cov = params_.external_forces_process_cov.asDiagonal();
        kin.com_linear_acceleration_cov = params_.com_linear_acceleration_cov.asDiagonal();

        // Compute GRF in world frame
        GroundReactionForceMeasurement grf;
        double den = 0.0;

        for (const auto& frame : state.getContactsFrame()) {
            grf.timestamp = ft->at(frame).timestamp;
            const Eigen::Isometry3d& foot_pose = state.getFootPose(frame);
            const double probability = state.contact_state_.contacts_probability.at(frame);

            if (probability > 0.0) {
                grf.force.noalias() += foot_pose.linear() * ft->at(frame).force;
                grf.cop.noalias() += probability * (foot_pose * ft->at(frame).cop);
                den += probability;
            }
        }

        if (den > 0.0) {
            grf.cop /= den;
        }

        state.centroidal_state_.cop_position = grf.cop;
        state.centroidal_state_.timestamp = grf.timestamp;

        // Update CoM state if in contact
        if (den > 0.0) {
            state.centroidal_state_ = com_estimator_.predict(state.centroidal_state_, kin, grf);
        }

        // Update CoM state with IMU measurements
        state.centroidal_state_ = com_estimator_.updateWithImu(state.centroidal_state_, kin, grf);
    }

    // Update CoM state with kinematic measurements
    state.centroidal_state_.timestamp = kin.timestamp;
    state.centroidal_state_ = com_estimator_.updateWithKinematics(state.centroidal_state_, kin);

    // Check if state has converged
    if (!state.is_valid_ && cycle_++ > params_.convergence_cycles) {
        state.is_valid_ = true;
    }

    // Create a threadpool job to log data 
    // check if the threadpool is already running
    if (!threadpool_->isRunning()) {
        threadpool_->addJob([this, base_state = state.base_state_, 
                               centroidal_state = state.centroidal_state_,
                               contact_state = state.contact_state_,
                               imu=imu, joints=joints, ft=ft]() {
        try {
            // Log all state data to MCAP file
            debug_logger_.log(base_state);
            debug_logger_.log(centroidal_state);
            debug_logger_.log(contact_state);
            debug_logger_.log(imu);
            debug_logger_.log(joints);
            if (ft.has_value()) {
                debug_logger_.log(ft.value());
            }
            if (terrain_estimator_) {
                terrain_estimator_->updateLocalMap();
                // debug_logger_.log(terrain_estimator_->getLocalMap());
            }    
        } catch (const std::exception& e) {
                std::cerr << "Error in logging thread: " << e.what() << std::endl;
        }
        });
    }

    // Update the state
    state_ = std::move(state);
}


std::optional<State> Serow::getState(bool allow_invalid) {
    if (state_.is_valid_ || allow_invalid) {
        return state_;
    } else {
        return std::nullopt;
    }
}

const std::shared_ptr<TerrainElevation>& Serow::getTerrainEstimator() const {
    return terrain_estimator_;
}

}  // namespace serow
