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

Serow::Serow() {}

bool Serow::initialize(const std::string& config_file) {
    // Load configuration JSON
    std::string config_path;
    try {
        config_path = findFilepath(config_file);
        std::ifstream config_file(config_path);
        if (!config_file.is_open()) {
            std::cerr << RED_COLOR << "Failed to open config file: " << config_path << "\n"
                      << WHITE_COLOR;
            return false;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
    }

    auto config_file_stream = std::ifstream(config_path);
    if (!config_file_stream.is_open()) {
        std::cerr << RED_COLOR << "Failed to open config file: " << config_path << "\n"
                  << WHITE_COLOR;
        return false;
    }

    json config;
    try {
        config = json::parse(config_file_stream);
    } catch (const json::parse_error& e) {
        std::cerr << RED_COLOR << "Failed to parse config file: " << e.what() << "\n"
                  << WHITE_COLOR;
        return false;
    }

    // Check if config has any values
    if (config.empty()) {
        std::cerr << RED_COLOR << "Config file is empty\n" << WHITE_COLOR;
        return false;
    }

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

    // Initialize robot name
    if (!checkConfigParam("robot_name", params_.robot_name)) {
        return false;
    }

    // Whether or not to log proprioception, exteroception, and measurement data
    if (!checkConfigParam("log_data", params_.log_data)) {
        return false;
    }

    if (!checkConfigParam("log_measurements", params_.log_measurements)) {
        return false;
    }

    // Initialize base frame
    if (!checkConfigParam("base_frame", params_.base_frame)) {
        return false;
    }

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
    params_.contacts_frame = std::move(contacts_frame);

    if (!checkConfigParam("point_feet", params_.point_feet))
        return false;

    if (!checkConfigParam("imu_rate", params_.imu_rate))
        return false;

    if (!checkConfigParam("attitude_estimator_proportional_gain", params_.Kp))
        return false;

    if (!checkConfigParam("attitude_estimator_integral_gain", params_.Ki))
        return false;

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
    if (!checkConfigParam("terrain_estimator", params_.terrain_estimator_type))
        return false;
    if (!checkConfigParam("minimum_terrain_height_variance",
                          params_.minimum_terrain_height_variance))
        return false;
    if (!checkConfigParam("maximum_contact_points", params_.maximum_contact_points))
        return false;
    if (!checkConfigParam("maximum_recenter_distance", params_.maximum_recenter_distance))
        return false;
    if (!checkConfigParam("minimum_contact_probability", params_.minimum_contact_probability))
        return false;

    // Read log directory parameter
    if (!checkConfigParam("log_dir", params_.log_dir))
        return false;

    // Create log directory if it doesn't exist
    try {
        if (!std::filesystem::exists(params_.log_dir)) {
            std::filesystem::create_directories(params_.log_dir);
            // If serow-timings.txt exists, delete it
            if (std::filesystem::exists(params_.log_dir + "/serow-timings.txt")) {
                std::filesystem::remove(params_.log_dir + "/serow-timings.txt");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Failed to create log directory: " << e.what() << "\n"
                  << WHITE_COLOR;
        return false;
    }

    // Check TFs
    if (!checkConfigArray("R_base_to_gyro", 9))
        return false;
    if (!checkConfigArray("R_base_to_acc", 9))
        return false;

    bool has_ground_truth = !config["T_base_to_ground_truth"].is_null();
    if (has_ground_truth && !checkConfigArray("T_base_to_ground_truth", 16))
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

    // Initialize kinematic estimator
    std::string model_path;
    if (!checkConfigParam("model_path", model_path)) {
        std::cerr << RED_COLOR << "Configuration: model_path not found in config" << WHITE_COLOR
                  << std::endl;
        return false;
    }

    const std::string model_filepath = findFilepath(model_path);
    if (model_filepath.empty()) {
        std::cerr << RED_COLOR << "Cofiguration: Model file '" << model_path << "' not found!"
                  << WHITE_COLOR << std::endl;
        return false;
    }

    kinematic_estimator_ =
        std::make_unique<RobotKinematics>(model_filepath, params_.joint_position_variance);

    // Load matrices
    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            params_.R_base_to_gyro(i, j) = config["R_base_to_gyro"][3 * i + j];
            params_.R_base_to_acc(i, j) = config["R_base_to_acc"][3 * i + j];

            size_t k = 0;
            for (const auto& frame : params_.contacts_frame) {
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

    if (has_ground_truth) {
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < 4; j++) {
                params_.T_base_to_ground_truth(i, j) = config["T_base_to_ground_truth"][4 * i + j];
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

    reset();

    // Create timers
    timers_.clear();
    timers_.try_emplace("imu-outlier-detection");
    timers_.try_emplace("imu-estimation");
    timers_.try_emplace("joint-estimation");
    timers_.try_emplace("forward-kinematics");
    timers_.try_emplace("leg-odometry");
    timers_.try_emplace("base-estimator-predict");
    timers_.try_emplace("base-estimator-update");
    timers_.try_emplace("com-estimator-predict");
    timers_.try_emplace("com-estimator-update");
    timers_.try_emplace("contact-estimation");
    timers_.try_emplace("frame-tree-update");
    timers_.try_emplace("total-time");
    return true;
}

void Serow::initializeLogging() {
    if (!proprioception_logger_job_) {
        proprioception_logger_job_ = std::make_unique<ThreadPool>();
    }
    if (!exteroception_logger_job_) {
        exteroception_logger_job_ = std::make_unique<ThreadPool>();
    }
    if (!measurement_logger_job_) {
        measurement_logger_job_ = std::make_unique<ThreadPool>();
    }
    if (!proprioception_logger_) {
        proprioception_logger_ =
            std::make_unique<ProprioceptionLogger>(params_.log_dir + "/serow_proprioception.mcap");
    }
    if (!exteroception_logger_) {
        exteroception_logger_ =
            std::make_unique<ExteroceptionLogger>(params_.log_dir + "/serow_exteroception.mcap");
    }
    if (!measurement_logger_) {
        measurement_logger_ =
            std::make_unique<MeasurementLogger>(params_.log_dir + "/serow_measurements.mcap");
    }
    if (!timings_logger_job_) {
        timings_logger_job_ = std::make_unique<ThreadPool>();
    }
}

void Serow::logMeasurements(ImuMeasurement imu,
                            const std::map<std::string, JointMeasurement>& joints,
                            std::map<std::string, ForceTorqueMeasurement> ft,
                            std::optional<BasePoseGroundTruth> base_pose_ground_truth) {
    if (!params_.log_measurements) {
        return;
    }
    measurement_logger_job_->addJob([this, imu = imu, joints = joints, ft = ft,
                                     base_pose_ground_truth = std::move(base_pose_ground_truth)]() {
        try {
            if (!measurement_logger_->isInitialized()) {
                measurement_logger_->setStartTime(imu.timestamp);
            }
            // Log all measurement data to MCAP file
            measurement_logger_->log(imu);
            measurement_logger_->log(joints);
            if (!ft.empty()) {
                measurement_logger_->log(ft);
            }
            if (base_pose_ground_truth.has_value()) {
                // Make a copy of the ground truth value
                auto gt = base_pose_ground_truth.value();
                // Transform the base pose to the ground truth frame
                gt.position = params_.T_base_to_ground_truth * gt.position;
                gt.orientation = Eigen::Quaterniond(params_.T_base_to_ground_truth.linear() *
                                                    gt.orientation.toRotationMatrix());
                measurement_logger_->log(gt);
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in measurement logging thread: " << e.what() << std::endl;
        }
    });
}

void Serow::runJointsEstimator(State& state,
                               const std::map<std::string, JointMeasurement>& joints) {
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
                if (state.isInitialized()) {
                    joint_estimators_.at(key).setState(
                        Eigen::Matrix<double, 1, 1>(state.joint_state_.joints_position.at(key)),
                        Eigen::Matrix<double, 1, 1>(state.joint_state_.joints_velocity.at(key)));
                } else {
                    joint_estimators_.at(key).setState(Eigen::Matrix<double, 1, 1>(value.position),
                                                       Eigen::Matrix<double, 1, 1>(0.0));
                }
            }
            joints_velocity[key] = joint_estimators_.at(key).filter(
                Eigen::Matrix<double, 1, 1>(value.position), value.timestamp)(0);
        }
    }
    state.joint_state_.timestamp = joint_timestamp;
    state.joint_state_.joints_position = std::move(joints_position);
    state.joint_state_.joints_velocity = std::move(joints_velocity);
}

bool Serow::runImuEstimator(State& state, ImuMeasurement& imu) {
    // Transform IMU measurements to base frame
    imu.angular_velocity = params_.R_base_to_gyro * imu.angular_velocity;
    imu.linear_acceleration = params_.R_base_to_acc * imu.linear_acceleration;

    // Estimate the base frame attitude
    if (!attitude_estimator_) {
        attitude_estimator_ = std::make_unique<Mahony>(params_.imu_rate, params_.Kp, params_.Ki);
        attitude_estimator_->setState(state.base_state_.base_orientation);
    }

    attitude_estimator_->filter(imu.angular_velocity, imu.linear_acceleration, imu.timestamp);
    imu.orientation = attitude_estimator_->getQ();
    imu.orientation_cov = params_.base_orientation_cov.asDiagonal();

    // IMU bias calibration - Assuming the IMU is stationary
    if (!state.isInitialized()) {
        if (params_.calibrate_initial_imu_bias) {
            if (imu_calibration_cycles_ < params_.max_imu_calibration_cycles) {
                const Eigen::Matrix3d& R_world_to_base = attitude_estimator_->getR();
                params_.bias_gyro += imu.angular_velocity;
                params_.bias_acc.noalias() += imu.linear_acceleration +
                    R_world_to_base.transpose() * Eigen::Vector3d(0.0, 0.0, -params_.g);
                imu_calibration_cycles_++;
                return false;
            } else {
                // Only divide if we've actually accumulated samples
                if (imu_calibration_cycles_ > 0) {
                    params_.bias_acc /= imu_calibration_cycles_;
                    params_.bias_gyro /= imu_calibration_cycles_;
                    params_.calibrate_initial_imu_bias = false;

                    // Update _ with calibrated biases
                    state.base_state_.imu_angular_velocity_bias = params_.bias_gyro;
                    state.base_state_.imu_linear_acceleration_bias = params_.bias_acc;

                    std::cout << "Calibration for stationary IMU finished at "
                              << imu_calibration_cycles_ << std::endl;
                    std::cout << "Gyrometer biases " << params_.bias_gyro.transpose() << std::endl;
                    std::cout << "Accelerometer biases " << params_.bias_acc.transpose()
                              << std::endl;
                }
            }
        }
    }

    // Create the base estimation measurements
    imu.angular_velocity_cov = params_.angular_velocity_cov.asDiagonal();
    imu.angular_velocity_bias_cov = params_.angular_velocity_bias_cov.asDiagonal();
    imu.linear_acceleration_cov = params_.linear_acceleration_cov.asDiagonal();
    imu.linear_acceleration_bias_cov = params_.linear_acceleration_bias_cov.asDiagonal();
    return true;
}

KinematicMeasurement Serow::runForwardKinematics(State& state) {
    kinematic_estimator_->updateJointConfig(state.joint_state_.joints_position,
                                            state.joint_state_.joints_velocity);

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

    // Prepare kinematic measurement
    KinematicMeasurement kin;
    kin.timestamp = state.joint_state_.timestamp;
    kin.base_to_foot_positions = std::move(base_to_foot_positions);
    kin.base_to_foot_orientations = std::move(base_to_foot_orientations);
    kin.base_to_foot_linear_velocities = std::move(base_to_foot_linear_velocities);
    kin.base_to_foot_angular_velocities = std::move(base_to_foot_angular_velocities);
    if (!state.isInitialized()) {
        // Initialize the state
        state.centroidal_state_.com_position = kinematic_estimator_->comPosition();
        for (const auto& frame : state.getContactsFrame()) {
            state.base_state_.contacts_position[frame] = kin.base_to_foot_positions.at(frame);
            state.base_state_.feet_position[frame] = state.base_state_.contacts_position.at(frame);
            state.base_state_.feet_orientation[frame] = kin.base_to_foot_orientations.at(frame);
        }
        if (!state.isPointFeet()) {
            state.base_state_.contacts_orientation = state.base_state_.feet_orientation;
        }
        state.setInitialized(true);
    }
    return kin;
}

void Serow::computeLegOdometry(const State& state, const ImuMeasurement& imu,
                               KinematicMeasurement& kin) {
    // Augment the kinematic measurement with the contact state if a new contact state is not
    // available use the last contact state from the previous cycle since contact state is not
    // changing rapidly
    kin.contacts_status = state.contact_state_.contacts_status;
    kin.contacts_probability = state.contact_state_.contacts_probability;

    // Initialize leg odometry if needed
    if (!leg_odometry_) {
        // Create leg odometry
        leg_odometry_ = std::make_unique<LegOdometry>(
            state.base_state_.base_position, state.base_state_.feet_position,
            state.base_state_.feet_orientation, state.getMass(), params_.tau_0, params_.tau_1,
            params_.joint_rate, params_.g, params_.eps);
    }

    // Perform leg odometry estimation
    leg_odometry_->estimate(
        imu.orientation, imu.angular_velocity, kin.base_to_foot_orientations,
        kin.base_to_foot_positions, kin.base_to_foot_linear_velocities,
        kin.base_to_foot_angular_velocities, state.contact_state_.contacts_force,
        state.contact_state_.contacts_probability, state.contact_state_.contacts_torque);

    kin.base_linear_velocity = leg_odometry_->getBaseLinearVelocity();
    kin.base_linear_velocity_cov = params_.base_linear_velocity_cov.asDiagonal();
    kin.contacts_position = leg_odometry_->getContactPositions();
    kin.position_cov = params_.contact_position_cov.asDiagonal();
    kin.position_slip_cov = params_.contact_position_slip_cov.asDiagonal();

    // Handle orientation for non-point feet
    if (!state.isPointFeet()) {
        kin.contacts_orientation = leg_odometry_->getContactOrientations();
        kin.orientation_cov = params_.contact_orientation_cov.asDiagonal();
        kin.orientation_slip_cov = params_.contact_orientation_slip_cov.asDiagonal();
    }

    // Compute orientation noise for contacts
    std::map<std::string, Eigen::Matrix3d> kin_contacts_orientation_noise;
    for (const auto& frame : state.getContactsFrame()) {
        const Eigen::Vector3d& lin_vel_noise = kinematic_estimator_->linearVelocityNoise(frame);
        kin.contacts_position_noise[frame].noalias() = lin_vel_noise * lin_vel_noise.transpose();

        if (!state.isPointFeet()) {
            const Eigen::Vector3d& ang_vel = kinematic_estimator_->angularVelocityNoise(frame);
            kin_contacts_orientation_noise[frame].noalias() = ang_vel * ang_vel.transpose();
        }
    }

    if (!state.isPointFeet()) {
        kin.contacts_orientation_noise = std::move(kin_contacts_orientation_noise);
    }
}

void Serow::runAngularMomentumEstimator(State& state) {
    // Get the angular momentum around the CoM in base frame coordinates as compute with rigid-body
    // kinematics
    const Eigen::Vector3d& com_angular_momentum = kinematic_estimator_->comAngularMomentum();
    const Eigen::Matrix3d& R_world_to_base = state.base_state_.base_orientation.toRotationMatrix();

    // Initialize angular momentum derivative estimator if needed
    if (!angular_momentum_derivative_estimator) {
        angular_momentum_derivative_estimator = std::make_unique<DerivativeEstimator>(
            "CoM Angular Momentum Derivative", params_.joint_rate,
            params_.angular_momentum_cutoff_frequency, 3);
        if (state.isInitialized()) {
            const Eigen::Matrix3d R_base_to_world = R_world_to_base.transpose();
            angular_momentum_derivative_estimator->setState(
                R_base_to_world * state.centroidal_state_.angular_momentum,
                R_base_to_world * state.centroidal_state_.angular_momentum_derivative);
        } else {
            angular_momentum_derivative_estimator->setState(com_angular_momentum,
                                                            Eigen::Vector3d::Zero());
        }
    }

    // Estimate the angular momentum derivative around the CoM in base frame
    const Eigen::Vector3d& com_angular_momentum_derivative =
        angular_momentum_derivative_estimator->filter(com_angular_momentum,
                                                      state.joint_state_.timestamp);

    // Update the state
    state.centroidal_state_.angular_momentum = R_world_to_base * com_angular_momentum;
    state.centroidal_state_.angular_momentum_derivative =
        R_world_to_base * com_angular_momentum_derivative;
}

void Serow::runContactEstimator(
    State& state, std::map<std::string, ForceTorqueMeasurement>& ft, KinematicMeasurement& kin,
    std::optional<std::map<std::string, ContactMeasurement>> contacts_probability) {
    // Current contact status
    std::map<std::string, bool> current_contact_status = state.contact_state_.contacts_status;

    // Estimate the leg end-effector contact state
    if (!ft.empty()) {
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
                                        state.getMass(), params_.g, params_.median_window));
                    contact_estimators_.at(frame).setState(
                        state.contact_state_.contacts_status.at(frame),
                        state.contact_state_.contacts_force.at(frame).z());
                }
            }

            state.contact_state_.timestamp = ft.at(frame).timestamp;

            // Transform F/T to base frame
            const Eigen::Matrix3d R_foot_to_base =
                kin.base_to_foot_orientations.at(frame).toRotationMatrix();
            const Eigen::Vector3d& frame_force = ft.at(frame).force;
            const Eigen::Matrix3d& R_foot_to_force = params_.R_foot_to_force.at(frame);
            const Eigen::Matrix3d& R_world_to_base =
                state.base_state_.base_orientation.toRotationMatrix();
            contacts_force[frame].noalias() =
                R_world_to_base * R_foot_to_base * R_foot_to_force * frame_force;

            // Process torque if not point feet
            if (!state.isPointFeet() && ft.at(frame).torque.has_value()) {
                contacts_torque[frame].noalias() = R_world_to_base * R_foot_to_base *
                    params_.R_foot_to_torque.at(frame) * ft.at(frame).torque.value();
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
            kin.is_new_contact[frame] = false;
            if (!current_contact_status.at(frame) &&
                state.contact_state_.contacts_status.at(frame)) {
                kin.is_new_contact[frame] = true;
            }

            ft.at(frame).cop = Eigen::Vector3d::Zero();

            // Calculate COP
            if (!state.isPointFeet() && contacts_torque.count(frame) &&
                state.contact_state_.contacts_probability.at(frame) > 0.0) {
                const double z_force = contacts_force.at(frame).z();
                if (std::abs(z_force) > 1e-6) {  // Avoid division by near-zero
                    ft.at(frame).cop =
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
}

void Serow::runBaseEstimator(State& state, const ImuMeasurement& imu,
                             const KinematicMeasurement& kin,
                             std::optional<OdometryMeasurement> odom) {
    timers_["base-estimator-predict"].start();
    // Initialize terrain estimator if needed
    if (params_.enable_terrain_estimation && !terrain_estimator_ && params_.is_contact_ekf) {
        float terrain_height = 0.0;
        int i = 0;

        for (const auto& [cf, cp] : state.contact_state_.contacts_status) {
            if (cp) {
                i++;
                terrain_height += state.base_state_.contacts_position.at(cf).z();
            }
        }

        if (i > 0) {
            terrain_height /= i;
        }

        // Initialize terrain elevation mapper
        if (params_.terrain_estimator_type == "naive") {
            terrain_estimator_ = std::make_shared<NaiveLocalTerrainMapper>();
        } else if (params_.terrain_estimator_type == "fast") {
            terrain_estimator_ = std::make_shared<LocalTerrainMapper>();
        } else {
            throw std::runtime_error("Invalid terrain estimator type: " +
                                     params_.terrain_estimator_type);
        }
        terrain_estimator_->initializeLocalMap(
            terrain_height, 1e4, params_.minimum_terrain_height_variance,
            params_.maximum_recenter_distance, params_.maximum_contact_points,
            params_.minimum_contact_probability);

        terrain_estimator_->recenter({static_cast<float>(state.base_state_.base_position.x()),
                                      static_cast<float>(state.base_state_.base_position.y())});
    }

    // Call the base estimator predict step
    state.base_state_.timestamp = imu.timestamp;
    if (params_.is_contact_ekf) {
        base_estimator_con_.predict(state.base_state_, imu, kin);
    } else {
        base_estimator_.predict(state.base_state_, imu);
    }
    timers_["base-estimator-predict"].stop();

    // Transform odometry measurements if available
    timers_["base-estimator-update"].start();
    if (odom.has_value()) {
        odom->base_position = params_.T_base_to_odom * odom->base_position;

        // Cache matrix multiplication
        const Eigen::Matrix3d& R_base_to_odom = params_.T_base_to_odom.linear();
        odom->base_orientation =
            Eigen::Quaterniond(R_base_to_odom * odom->base_orientation.toRotationMatrix());

        // Calculate covariance
        odom->base_position_cov =
            R_base_to_odom * odom->base_position_cov * R_base_to_odom.transpose();
        odom->base_orientation_cov =
            R_base_to_odom * odom->base_orientation_cov * R_base_to_odom.transpose();
    }

    // Update base state with relative to base contacts
    state.base_state_.timestamp = kin.timestamp;
    const Eigen::Isometry3d base_pose = state.getBasePose();
    if (params_.is_contact_ekf) {
        base_estimator_con_.update(state.base_state_, imu, kin, odom, terrain_estimator_);
    } else {
        base_estimator_.update(state.base_state_, imu, kin, odom);

        // Compute the contact pose in the world frame
        std::map<std::string, Eigen::Quaterniond> con_orient;

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

    // Estimate base angular velocity and linear acceleration
    const Eigen::Vector3d base_angular_velocity =
        imu.angular_velocity - state.getImuAngularVelocityBias();
    const Eigen::Vector3d base_linear_acceleration =
        base_pose.linear() * (imu.linear_acceleration - state.getImuLinearAccelerationBias()) -
        Eigen::Vector3d(0.0, 0.0, params_.g);
    if (!gyro_derivative_estimator) {
        gyro_derivative_estimator = std::make_unique<DerivativeEstimator>(
            "Gyro Derivative", params_.imu_rate, params_.gyro_cutoff_frequency, 3);
        if (state.isInitialized()) {
            const Eigen::Matrix3d R_base_to_world = base_pose.linear().transpose();
            gyro_derivative_estimator->setState(
                R_base_to_world * state.base_state_.base_angular_velocity,
                R_base_to_world * state.base_state_.base_angular_acceleration);
        } else {
            gyro_derivative_estimator->setState(base_angular_velocity, Eigen::Vector3d::Zero());
        }
    }
    const Eigen::Vector3d base_angular_acceleration =
        gyro_derivative_estimator->filter(base_angular_velocity, imu.timestamp);
    state.base_state_.base_angular_velocity = base_pose.linear() * base_angular_velocity;
    state.base_state_.base_angular_acceleration = base_pose.linear() * base_angular_acceleration;
    state.base_state_.base_linear_acceleration = base_linear_acceleration;

    // Update feet pose/velocity in world frame
    for (const auto& frame : state.getContactsFrame()) {
        // Cache calculations
        const Eigen::Vector3d& base_foot_pos = kin.base_to_foot_positions.at(frame);
        const Eigen::Vector3d transformed_pos = base_pose.linear() * base_foot_pos;

        state.base_state_.feet_position[frame].noalias() = base_pose * base_foot_pos;
        state.base_state_.feet_orientation[frame] = Eigen::Quaterniond(
            base_pose.linear() * kin.base_to_foot_orientations.at(frame).toRotationMatrix());

        state.base_state_.feet_linear_velocity[frame].noalias() =
            state.base_state_.base_linear_velocity +
            state.base_state_.base_angular_velocity.cross(transformed_pos) +
            base_pose.linear() * kin.base_to_foot_linear_velocities.at(frame);

        state.base_state_.feet_angular_velocity[frame].noalias() =
            state.base_state_.base_angular_velocity +
            base_pose.linear() * kin.base_to_foot_angular_velocities.at(frame);
    }
    timers_["base-estimator-update"].stop();
}

void Serow::runCoMEstimator(State& state, KinematicMeasurement& kin,
                            std::map<std::string, ForceTorqueMeasurement> ft) {
    // Prepare CoM estimation measurements
    const Eigen::Vector3d& base_to_com_position = kinematic_estimator_->comPosition();
    // Estimate the CoM angular momentum derivative
    runAngularMomentumEstimator(state);
    const Eigen::Isometry3d& base_pose = state.getBasePose();
    kin.com_position.noalias() = base_pose * base_to_com_position;
    kin.com_position_cov = params_.com_position_cov.asDiagonal();
    kin.com_angular_momentum_derivative.noalias() =
        state.centroidal_state_.angular_momentum_derivative;
    kin.com_position_process_cov = params_.com_position_process_cov.asDiagonal();
    kin.com_linear_velocity_process_cov = params_.com_linear_velocity_process_cov.asDiagonal();
    kin.external_forces_process_cov = params_.external_forces_process_cov.asDiagonal();
    kin.com_linear_acceleration_cov = params_.com_linear_acceleration_cov.asDiagonal();

    // Approximate the CoM linear acceleration in the world frame
    const Eigen::Vector3d& base_angular_acceleration = state.base_state_.base_angular_acceleration;
    const Eigen::Vector3d& base_linear_acceleration = state.base_state_.base_linear_acceleration;
    const Eigen::Vector3d& base_angular_velocity = state.base_state_.base_angular_velocity;
    kin.com_linear_acceleration.noalias() = base_linear_acceleration +
        base_angular_velocity.cross(base_angular_velocity.cross(base_to_com_position)) +
        base_angular_acceleration.cross(base_to_com_position);

    state.centroidal_state_.com_linear_acceleration.noalias() = kin.com_linear_acceleration;

    // Process force-torque measurements if available
    if (!ft.empty()) {
        // Compute GRF in world frame
        GroundReactionForceMeasurement grf;
        double den = 0.0;
        for (const auto& frame : state.getContactsFrame()) {
            grf.timestamp = ft.at(frame).timestamp;
            const Eigen::Isometry3d& foot_pose = state.getFootPose(frame);
            const double probability = state.contact_state_.contacts_probability.at(frame);

            if (probability > 0.0) {
                grf.force.noalias() += foot_pose.linear() * ft.at(frame).force;
                grf.cop.noalias() += probability * (foot_pose * ft.at(frame).cop);
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
            timers_["com-estimator-predict"].start();
            com_estimator_.predict(state.centroidal_state_, kin, grf);
            timers_["com-estimator-predict"].stop();
        }

        // Update CoM state with IMU measurements
        timers_["com-estimator-update"].start();
        com_estimator_.updateWithImu(state.centroidal_state_, kin, grf);
    } else {
        timers_["com-estimator-update"].start();
    }

    // Update CoM state with kinematic measurements
    state.centroidal_state_.timestamp = kin.timestamp;
    com_estimator_.updateWithKinematics(state.centroidal_state_, kin);
    timers_["com-estimator-update"].stop();
}

void Serow::logProprioception(const State& state, const ImuMeasurement& imu) {
    if (!params_.log_data) {
        return;
    }
    proprioception_logger_job_->addJob([this, joints_state = state.joint_state_,
                                        base_state = state.base_state_,
                                        centroidal_state = state.centroidal_state_,
                                        contact_state = state.contact_state_, imu = imu,
                                        frame_tfs = frame_tfs_]() {
        try {
            if (!proprioception_logger_->isInitialized()) {
                proprioception_logger_->setStartTime(std::min(base_state.timestamp, imu.timestamp));
            }
            // Log all state data to MCAP file
            proprioception_logger_->log(imu);
            proprioception_logger_->log(joints_state);
            proprioception_logger_->log(contact_state);
            proprioception_logger_->log(centroidal_state);
            proprioception_logger_->log(base_state);
            proprioception_logger_->log(frame_tfs, base_state.timestamp);
        } catch (const std::exception& e) {
            std::cerr << "Error in proprioception logging thread: " << e.what() << std::endl;
        }
    });
}

void Serow::logExteroception(const State& state) {
    if (!params_.log_data) {
        return;
    }
    if (terrain_estimator_ && !exteroception_logger_job_->isRunning() && exteroception_logger_ &&
        ((state.base_state_.timestamp - exteroception_logger_->getLastTimestamp()) > 0.1)) {
        exteroception_logger_job_->addJob([this, ts = state.base_state_.timestamp]() {
            try {
                if (!exteroception_logger_->isInitialized()) {
                    exteroception_logger_->setStartTime(ts);
                }
                const size_t downsample_factor = 4;
                const auto [origin, bound_max, bound_min] = terrain_estimator_->getLocalMapInfo();
                const double res = resolution * downsample_factor;

                // Calculate actual downsampled dimensions
                const uint32_t width = std::ceil((bound_max[0] - bound_min[0]) / res);
                const uint32_t height = std::ceil((bound_max[1] - bound_min[1]) / res);

                exteroception_logger_->setGridParameters(res, width, height, origin[0], origin[1]);

                // Pre-allocate grid with exact size
                std::vector<float> elevation(width * height,
                                             std::numeric_limits<float>::quiet_NaN());
                std::vector<float> variance(width * height,
                                            std::numeric_limits<float>::quiet_NaN());

                // Use integer-based iteration for consistency
                for (uint32_t row = 0; row < height; ++row) {
                    for (uint32_t col = 0; col < width; ++col) {
                        // Calculate world coordinates from grid indices
                        float x = bound_min[0] + col * res;
                        float y = bound_min[1] + row * res;
                        const auto& cell = terrain_estimator_->getElevation({x, y});
                        if (cell.has_value()) {
                            elevation[row * width + col] = cell.value().height;
                            variance[row * width + col] = cell.value().variance;
                        }
                    }
                }

                // Verify size
                if (elevation.size() != width * height) {
                    std::cerr << "Grid size mismatch: expected " << (width * height) << ", got "
                              << elevation.size() << std::endl;
                    return;  // Don't log invalid data
                }

                exteroception_logger_->log(elevation, variance, ts);
            } catch (const std::exception& e) {
                std::cerr << "Error in exteroception logging thread: " << e.what() << std::endl;
            }
        });
    }
}

bool Serow::filter(ImuMeasurement imu, std::map<std::string, JointMeasurement> joints,
                   std::optional<std::map<std::string, ForceTorqueMeasurement>> force_torque,
                   std::optional<OdometryMeasurement> odom,
                   std::optional<std::map<std::string, ContactMeasurement>> contacts_probability,
                   std::optional<BasePoseGroundTruth> base_pose_ground_truth) {
    timers_["total-time"].start();

    // Early return if not initialized and no FT measurement
    if (!is_initialized_) {
        if (!force_torque.has_value()) {
            timers_["total-time"].stop();
            return false;
        }
        is_initialized_ = true;
        initializeLogging();
    }

    // Check if foot frames exist on the F/T measurement
    std::map<std::string, ForceTorqueMeasurement> ft;
    if (force_torque.has_value()) {
        for (const auto& frame : state_.contacts_frame_) {
            if (force_torque.value().count(frame) == 0) {
                throw std::runtime_error("Foot frame <" + frame +
                                         "> does not exist in the force measurements");
            }
        }
        // Force-torque measurements are valid and ready to be consumed
        ft = std::move(force_torque.value());
    }

    // log the incoming measurements
    logMeasurements(imu, joints, ft, base_pose_ground_truth);

    // Update the joint state estimate
    timers_["joint-estimation"].start();
    runJointsEstimator(state_, joints);
    timers_["joint-estimation"].stop();

    // Check if the IMU measurements are valid with the Median Absolute Deviation (MAD)
    timers_["imu-outlier-detection"].start();
    bool is_imu_outlier = isImuMeasurementOutlier(imu);
    timers_["imu-outlier-detection"].stop();
    if (is_imu_outlier) {
        timers_["total-time"].stop();
        return false;
    }

    // Estimate the base frame attitude and initial IMU biases
    timers_["imu-estimation"].start();
    bool calibrated = runImuEstimator(state_, imu);
    timers_["imu-estimation"].stop();
    if (!calibrated) {
        timers_["total-time"].stop();
        return false;
    }

    // Update the kinematic structure
    timers_["forward-kinematics"].start();
    KinematicMeasurement kin = runForwardKinematics(state_);
    timers_["forward-kinematics"].stop();

    // Estimate the contact state
    if (!ft.empty()) {
        timers_["contact-estimation"].start();
        runContactEstimator(state_, ft, kin, contacts_probability);
        timers_["contact-estimation"].stop();
    }

    // Compute the leg odometry and update the kinematic measurement accordingly
    timers_["leg-odometry"].start();
    computeLegOdometry(state_, imu, kin);
    timers_["leg-odometry"].stop();

    // Run the base estimator
    runBaseEstimator(state_, imu, kin, odom);

    // Run the CoM estimator
    runCoMEstimator(state_, kin, ft);

    // Update all frame transformations
    timers_["frame-tree-update"].start();
    updateFrameTree(state_);
    timers_["frame-tree-update"].stop();

    // Check if state has converged
    if (!state_.is_valid_ && cycle_++ > params_.convergence_cycles) {
        state_.is_valid_ = true;
    }

    // Log the estimated state
    logProprioception(state_, imu);
    logExteroception(state_);
    timers_["total-time"].stop();
    logTimings();
    return true;
}

void Serow::updateFrameTree(const State& state) {
    std::map<std::string, Eigen::Isometry3d> frame_tfs;
    const Eigen::Isometry3d& base_pose = state.getBasePose();
    frame_tfs[params_.base_frame] = base_pose;
    for (const auto& frame : kinematic_estimator_->frameNames()) {
        if (frame != params_.base_frame) {
            try {
                frame_tfs[frame] = base_pose * kinematic_estimator_->linkTF(frame);
            } catch (const std::exception& e) {
                std::cerr << "Error in frame " << frame << " TF computation: " << e.what()
                          << std::endl;
            }
        }
    }
    frame_tfs_ = std::move(frame_tfs);
}

std::optional<State> Serow::getState(bool allow_invalid) {
    if (state_.is_valid_ || allow_invalid) {
        return state_;
    } else {
        return std::nullopt;
    }
}

std::optional<BaseState> Serow::getBaseState(bool allow_invalid) {
    if (state_.is_valid_ || allow_invalid) {
        return state_.base_state_;
    } else {
        return std::nullopt;
    }
}

std::optional<ContactState> Serow::getContactState(bool allow_invalid) {
    if (state_.is_valid_ || allow_invalid) {
        return state_.contact_state_;
    } else {
        return std::nullopt;
    }
}

const std::shared_ptr<TerrainElevation>& Serow::getTerrainEstimator() const {
    return terrain_estimator_;
}

Serow::~Serow() {
    stopLogging();
}

bool Serow::isInitialized() const {
    return is_initialized_;
}

void Serow::setState(const State& state) {
    reset();
    if (params_.is_contact_ekf) {
        base_estimator_con_.setState(state.base_state_);
    } else {
        base_estimator_.setState(state.base_state_);
    }
    com_estimator_.setState(state.centroidal_state_);
    state_ = state;
    state_.setInitialized(true);
}

void Serow::stopLogging() {
    if (proprioception_logger_job_) {
        // Wait for all jobs to finish
        while (proprioception_logger_job_->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    if (exteroception_logger_job_) {
        // Wait for all jobs to finish
        while (exteroception_logger_job_->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    if (measurement_logger_job_) {
        // Wait for all jobs to finish
        while (measurement_logger_job_->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    if (timings_logger_job_) {
        // Wait for all jobs to finish
        while (timings_logger_job_->isRunning()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    proprioception_logger_.reset();
    exteroception_logger_.reset();
    measurement_logger_.reset();
    proprioception_logger_job_.reset();
    exteroception_logger_job_.reset();
    measurement_logger_job_.reset();
    last_log_time_.reset();
    for (auto& [name, timer] : timers_) {
        timer.reset();
    }
}

void Serow::reset() {
    joint_estimators_.clear();
    angular_momentum_derivative_estimator.reset();
    gyro_derivative_estimator.reset();
    contact_estimators_.clear();
    attitude_estimator_.reset();
    leg_odometry_.reset();
    terrain_estimator_.reset();
    frame_tfs_.clear();
    imu_outlier_detector_.clear();

    is_initialized_ = false;
    cycle_ = 0;
    imu_calibration_cycles_ = 0;

    // Initialize state
    State state(params_.contacts_frame, params_.point_feet, params_.base_frame);
    state_ = std::move(state);

    // Load bias values from configuration
    state_.base_state_.imu_angular_velocity_bias = params_.bias_gyro;
    state_.base_state_.imu_linear_acceleration_bias = params_.bias_acc;

    // Initialize state uncertainty
    state_.mass_ = kinematic_estimator_->getTotalMass();
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

    // Initialize the base and CoM estimators
    if (params_.is_contact_ekf) {
        base_estimator_con_.init(state_.base_state_, state_.getContactsFrame(),
                                 state_.isPointFeet(), params_.g, params_.imu_rate,
                                 params_.outlier_detection);
    } else {
        base_estimator_.init(state_.base_state_, params_.g, params_.imu_rate,
                             params_.outlier_detection);
    }
    com_estimator_.init(state_.centroidal_state_, state_.getMass(), params_.g,
                        params_.force_torque_rate);

    // Compute a valid history size for the IMU outlier detection
    size_t imu_history_size = params_.imu_rate < 120 ? 40 : params_.imu_rate / 3;
    if (imu_history_size > 200) {
        imu_history_size = 200;
    }

    // Initialize IMU outlier detection storage
    for (size_t i = 0; i < 6; i++) {
        imu_outlier_detector_.push_back(MovingMedianFilter(imu_history_size));
    }

    // Terminate logging threads
    stopLogging();
}

bool Serow::isImuMeasurementOutlier(const ImuMeasurement& imu) {
    const double MAD_THRESHOLD_MULTIPLIER = 6.0;  // 6-sigma rule

    // Add current measurements to filters
    for (size_t i = 0; i < 3; i++) {
        imu_outlier_detector_[i].filter(imu.angular_velocity[i]);
        imu_outlier_detector_[i + 3].filter(imu.linear_acceleration[i]);
    }

    // Need sufficient history for MAD calculation
    if (imu_outlier_detector_[0].size() < imu_outlier_detector_[0].maxSize() / 2) {
        return false;
    }

    // Calculate current medians
    const Eigen::Vector3d av_median(imu_outlier_detector_[0].getMedian(),
                                    imu_outlier_detector_[1].getMedian(),
                                    imu_outlier_detector_[2].getMedian());
    const Eigen::Vector3d la_median(imu_outlier_detector_[3].getMedian(),
                                    imu_outlier_detector_[4].getMedian(),
                                    imu_outlier_detector_[5].getMedian());

    // Helper function to calculate rolling MAD for a single axis
    auto calculateRollingMAD = [](const std::deque<double>& window, double median) -> double {
        std::vector<double> deviations;
        deviations.reserve(window.size());

        for (double value : window) {
            deviations.push_back(std::abs(value - median));
        }

        // Sort for median calculation
        std::sort(deviations.begin(), deviations.end());
        return deviations[deviations.size() / 2];
    };

    // Calculate MAD for each axis
    Eigen::Vector3d av_mad = Eigen::Vector3d::Zero();
    Eigen::Vector3d la_mad = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < 3; i++) {
        av_mad[i] = calculateRollingMAD(imu_outlier_detector_[i].getWindow(), av_median[i]);
        la_mad[i] = calculateRollingMAD(imu_outlier_detector_[i + 3].getWindow(), la_median[i]);
    }

    // Convert MAD to standard deviation approximation (MAD ≈ 0.6745 * σ)
    const double MAD_TO_STD_FACTOR = 1.4826;  // 1 / 0.6745
    const Eigen::Vector3d av_std_dev(av_mad * MAD_TO_STD_FACTOR);
    const Eigen::Vector3d la_std_dev(la_mad * MAD_TO_STD_FACTOR);

    // Calculate z-scores for current measurement
    const Eigen::Vector3d av_z_scores =
        (imu.angular_velocity - av_median).cwiseQuotient(av_std_dev);
    const Eigen::Vector3d la_z_scores =
        (imu.linear_acceleration - la_median).cwiseQuotient(la_std_dev);

    // Check if any component exceeds the threshold
    const bool angular_velocity_outlier =
        (av_z_scores.array().abs() > MAD_THRESHOLD_MULTIPLIER).any();
    const bool linear_acceleration_outlier =
        (la_z_scores.array().abs() > MAD_THRESHOLD_MULTIPLIER).any();

    return angular_velocity_outlier || linear_acceleration_outlier;
}

void Serow::logTimings() {
    // Log to txt file on a seperate thread pool job every 0.5s
    if (!last_log_time_.has_value()) {
        last_log_time_ = std::chrono::high_resolution_clock::now();
    }
    if (std::chrono::high_resolution_clock::now() - last_log_time_.value() <
        std::chrono::milliseconds(500)) {
        return;
    }

    last_log_time_ = std::chrono::high_resolution_clock::now();
    timings_logger_job_->addJob([this]() {
        try {
            std::ofstream file(params_.log_dir + "/serow-timings.txt");
            for (const auto& [name, timer] : timers_) {
                file << "[" << std::chrono::high_resolution_clock::now().time_since_epoch().count()
                     << "] " << name << ": mean=" << timer.getMean() << "ms, min=" << timer.getMin()
                     << "ms, max=" << timer.getMax() << "ms"
                     << " count=" << timer.getCount() << std::endl;
            }
            file.close();
        } catch (const std::exception& e) {
            std::cerr << "Error logging timings: " << e.what() << std::endl;
        }
    });
}

}  // namespace serow
