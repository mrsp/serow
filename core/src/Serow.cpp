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

    // Initialize point feet flag
    if (!checkConfigParam("point_feet", params_.point_feet)) {
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

    if (!checkConfigParam("use_imu_orientation", params_.use_imu_orientation))
        return false;

    if (!checkConfigParam("imu_outlier_detection", params_.imu_outlier_detection))
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
    if (!checkConfigParam("force_torque_rate", params_.force_torque_rate))
        return false;
    if (!checkConfigParam("joint_rate", params_.joint_rate))
        return false;
    if (!checkConfigParam("estimate_joint_velocity", params_.estimate_joint_velocity))
        return false;
    if (!checkConfigParam("joint_position_variance", params_.joint_position_variance))
        return false;
    if (!checkConfigParam("g", params_.g))
        return false;
    if (!checkConfigParam("tau_0", params_.tau_0))
        return false;
    if (!checkConfigParam("tau_1", params_.tau_1))
        return false;
    if (!checkConfigParam("estimate_contact_status", params_.estimate_contact_status))
        return false;
    if (!checkConfigParam("median_window", params_.median_window))
        return false;
    if (!checkConfigParam("convergence_cycles", params_.convergence_cycles))
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

    try {
        kinematic_estimator_ =
            std::make_unique<RobotKinematics>(model_filepath, params_.joint_position_variance);
    } catch (const std::exception& e) {
        std::cerr << RED_COLOR << "Failed to create kinematic estimator: " << e.what()
                  << WHITE_COLOR << std::endl;
        return false;
    } catch (...) {
        std::cerr << RED_COLOR << "Unknown error occurred while creating kinematic estimator"
                  << WHITE_COLOR << std::endl;
        return false;
    }

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
                                                 "com_position_process_covariance",
                                                 "com_linear_velocity_process_covariance",
                                                 "external_forces_process_covariance",
                                                 "com_position_covariance",
                                                 "com_linear_acceleration_covariance",
                                                 "initial_base_position_covariance",
                                                 "initial_base_orientation_covariance",
                                                 "initial_base_linear_velocity_covariance",
                                                 "initial_imu_linear_acceleration_bias_covariance",
                                                 "initial_imu_angular_velocity_bias_covariance",
                                                 "initial_com_position_covariance",
                                                 "initial_com_linear_velocity_covariance",
                                                 "initial_external_forces_covariance"};

    for (const auto& cov_name : cov_arrays) {
        if (!checkConfigArray(cov_name, 3))
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
    loadCovarianceVector("initial_imu_linear_acceleration_bias_covariance",
                         params_.initial_imu_linear_acceleration_bias_cov);
    loadCovarianceVector("initial_imu_angular_velocity_bias_covariance",
                         params_.initial_imu_angular_velocity_bias_cov);
    loadCovarianceVector("initial_com_position_covariance", params_.initial_com_position_cov);
    loadCovarianceVector("initial_com_linear_velocity_covariance",
                         params_.initial_com_linear_velocity_cov);
    loadCovarianceVector("initial_external_forces_covariance", params_.initial_external_forces_cov);


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

    // Compute SG-filter parameters
    // Calculate M based on time horizon (minimum 3 points for 2nd order poly)
    const double time_horizon = 0.02;
    const int M_joint = std::max(3, static_cast<int>(std::round(time_horizon * params_.joint_rate)));
    const int M_imu = std::max(3, static_cast<int>(std::round(time_horizon * params_.imu_rate)));
    // Compute coefficients
    coeffs_joint_ = computeSGCoefficients(M_joint);
    coeffs_imu_ = computeSGCoefficients(M_imu);
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
            std::make_shared<ExteroceptionLogger>(params_.log_dir + "/serow_exteroception.mcap");
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
                double start_time = std::min(imu.timestamp, joints.begin()->second.timestamp);
                if (!ft.empty()) {
                    start_time = std::min(start_time, ft.begin()->second.timestamp);
                }
                if (base_pose_ground_truth.has_value()) {
                    start_time = std::min(start_time, base_pose_ground_truth.value().timestamp);
                }
                measurement_logger_->setStartTime(start_time);
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
                                          DerivativeEstimator(key, coeffs_joint_, params_.joint_rate));
                if (state.isInitialized()) {
                    joint_estimators_.at(key).setState(
                        Eigen::Matrix<double, 1, 1>(state.joint_state_.joints_velocity.at(key)));
                } else {
                    joint_estimators_.at(key).setState(Eigen::Matrix<double, 1, 1>(0.0));
                }
            }
            joints_velocity[key] = joint_estimators_.at(key).filter(
                Eigen::Matrix<double, 1, 1>(value.position), 
                Eigen::Matrix<double, 1, 1>(params_.joint_position_variance), 
                value.timestamp)(0);
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

    const Eigen::Matrix3d R_base_to_gyro_transpose = params_.R_base_to_gyro.transpose();
    const Eigen::Matrix3d R_base_to_acc_transpose = params_.R_base_to_acc.transpose();
    imu.angular_velocity_cov = params_.R_base_to_gyro * params_.angular_velocity_cov.asDiagonal() * R_base_to_gyro_transpose;
    imu.angular_velocity_bias_cov = params_.R_base_to_gyro * params_.angular_velocity_bias_cov.asDiagonal() * R_base_to_gyro_transpose;
    imu.linear_acceleration_cov = params_.R_base_to_acc * params_.linear_acceleration_cov.asDiagonal() * R_base_to_acc_transpose;
    imu.linear_acceleration_bias_cov = params_.R_base_to_acc * params_.linear_acceleration_bias_cov.asDiagonal() * R_base_to_acc_transpose;

    // Estimate the base frame attitude
    if (!attitude_estimator_) {
        attitude_estimator_ = std::make_unique<Mahony>(params_.imu_rate, imu.angular_velocity_cov, 
                                                       imu.linear_acceleration_cov, params_.Kp, params_.Ki);
        attitude_estimator_->setState(state.base_state_.base_orientation);
    }

    attitude_estimator_->filter(imu.angular_velocity, imu.linear_acceleration, imu.timestamp);
    imu.orientation = attitude_estimator_->getQ();
    imu.orientation_cov = attitude_estimator_->getOrientationCov();

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
    kin.timestamp = timestamp_;
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

    // Perform leg odometry estimation
    leg_odometry_->estimate(
        kin.timestamp,
        imu.orientation, imu.angular_velocity, kin.base_to_foot_orientations,
        kin.base_to_foot_positions, kin.base_to_foot_linear_velocities,
        kin.base_to_foot_angular_velocities, state.contact_state_.contacts_force,
        state.contact_state_.contacts_probability, kin.contacts_position_noise, 
        imu.angular_velocity_cov, state.contact_state_.contacts_torque);
   
    kin.base_position = leg_odometry_->getBasePosition();
    kin.base_linear_velocity = leg_odometry_->getBaseLinearVelocity();
    kin.base_linear_velocity_cov = leg_odometry_->getBaseLinearVelocityCov();
    kin.contacts_position = leg_odometry_->getContactPositions();

    // Handle orientation for non-point feet
    if (!state.isPointFeet()) {
        kin.contacts_orientation = leg_odometry_->getContactOrientations();
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
            "CoM Angular Momentum Derivative", coeffs_joint_, params_.joint_rate, 3);
        if (state.isInitialized()) {
            const Eigen::Matrix3d R_base_to_world = R_world_to_base.transpose();
            angular_momentum_derivative_estimator->setState(
                R_base_to_world * state.centroidal_state_.angular_momentum_derivative);
        } else {
            angular_momentum_derivative_estimator->setState(Eigen::Vector3d::Zero());
        }
    }

    // Estimate the angular momentum derivative around the CoM in base frame
    const Eigen::Vector3d& com_angular_momentum_derivative =
        angular_momentum_derivative_estimator->filter(com_angular_momentum,
                                                      Eigen::Vector3d::Ones(),
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
            if (!state.isPointFeet()) {
                if (ft.count(frame) > 0 && ft.at(frame).torque.has_value()) {
                    contacts_torque[frame].noalias() = R_world_to_base * R_foot_to_base *
                        params_.R_foot_to_torque.at(frame) * ft.at(frame).torque.value();
                } else {
                    throw std::runtime_error("No torque measurement provided for frame: " + frame);
                }
            }

            // Estimate the contact status
            if (params_.estimate_contact_status) {
                // Create contact estimators if needed
                if (contact_estimators_.count(frame) == 0) {
                    contact_estimators_.emplace(
                        frame,
                        ContactDetector(frame, 100.0, 10.0,
                                        state.getMass(), params_.g, params_.median_window));
                    contact_estimators_.at(frame).setState(
                        state.contact_state_.contacts_status.at(frame),
                        state.contact_state_.contacts_force.at(frame).z());
                }
                contact_estimators_.at(frame).SchmittTrigger(contacts_force.at(frame).z());
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
        } else {
            throw std::runtime_error("No contact probability provided and contact status estimation is disabled");
        }

        // Compute binary contact status
        for (const auto& frame : state.getContactsFrame()) {
            state.contact_state_.contacts_status[frame] = 
                state.contact_state_.contacts_probability.at(frame) > 0.5;
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
    if (params_.enable_terrain_estimation && !terrain_estimator_) {
        double terrain_height = 0.0;
        double den = 0.0;
        Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
        T_world_to_base.translation() = state.base_state_.base_position;
        T_world_to_base.linear() = state.base_state_.base_orientation.toRotationMatrix();
        for (const auto& [cf, cp] : state.contact_state_.contacts_status) {
            den += cp;
            terrain_height += cp * (T_world_to_base * state.base_state_.contacts_position.at(cf)).z();
        }

        if (den > 0.0) {
            terrain_height /= den;

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
    }

    // Call the base estimator predict step
    base_estimator_.predict(state.base_state_, imu);
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
    const Eigen::Isometry3d& base_pose = state.getBasePose();
    base_estimator_.update(state.base_state_, imu, kin, odom, terrain_estimator_);

    // Estimate base angular velocity and linear acceleration
    const Eigen::Vector3d base_angular_velocity =
        imu.angular_velocity - state.getImuAngularVelocityBias();
    const Eigen::Vector3d base_linear_acceleration =
        base_pose.linear() * (imu.linear_acceleration - state.getImuLinearAccelerationBias()) -
        Eigen::Vector3d(0.0, 0.0, params_.g);
    if (!gyro_derivative_estimator) {
        gyro_derivative_estimator = std::make_unique<DerivativeEstimator>(
            "Gyro Derivative", coeffs_imu_, params_.imu_rate, 3);
        if (state.isInitialized()) {
            const Eigen::Matrix3d R_base_to_world = base_pose.linear().transpose();
            gyro_derivative_estimator->setState(
                R_base_to_world * state.base_state_.base_angular_acceleration);
        } else {
            gyro_derivative_estimator->setState(Eigen::Vector3d::Zero());
        }
    }

    const Eigen::Vector3d base_angular_acceleration =
        gyro_derivative_estimator->filter(base_angular_velocity, imu.angular_velocity_cov.diagonal(), imu.timestamp);
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
    state.base_state_.timestamp = timestamp_;
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
    com_estimator_.updateWithKinematics(state.centroidal_state_, kin);
    state.centroidal_state_.timestamp = timestamp_;
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
        // Capture shared_ptrs directly to ensure they remain valid even if Serow is destroyed
        auto terrain_estimator = terrain_estimator_;
        auto exteroception_logger = exteroception_logger_;
        // Capture resolution value to avoid any potential issues with global variable access
        constexpr double res_base = static_cast<double>(resolution);
        exteroception_logger_job_->addJob([terrain_estimator, exteroception_logger, ts = state.base_state_.timestamp, res_base]() {
            try {
                if (!terrain_estimator || !exteroception_logger) {
                    std::cerr << "Error in exteroception logging: null pointer detected" << std::endl;
                    return;
                }
                if (!exteroception_logger->isInitialized()) {
                    exteroception_logger->setStartTime(ts);
                }
                const size_t downsample_factor = 4;
                const auto [origin, bound_max, bound_min] = terrain_estimator->getLocalMapInfo();
                const double res = res_base * downsample_factor;
                if (!(res > 0.0) || !std::isfinite(res)) {
                    std::cerr << "Error in exteroception logging: invalid resolution " << res
                              << std::endl;
                    return;
                }

                // Calculate actual downsampled dimensions
                const double dx = static_cast<double>(bound_max[0]) - static_cast<double>(bound_min[0]);
                const double dy = static_cast<double>(bound_max[1]) - static_cast<double>(bound_min[1]);
                if (!std::isfinite(dx) || !std::isfinite(dy) || dx <= 0.0 || dy <= 0.0) {
                    // Bounds can be temporarily invalid during init/recenter; just skip logging.
                    return;
                }
                const uint32_t width = static_cast<uint32_t>(std::ceil(dx / res));
                const uint32_t height = static_cast<uint32_t>(std::ceil(dy / res));
                if (width == 0 || height == 0) {
                    return;
                }

                // Prevent overflow in size computations
                const size_t grid_size = static_cast<size_t>(width) * static_cast<size_t>(height);
                constexpr size_t max_grid_size = static_cast<size_t>(map_dim) * static_cast<size_t>(map_dim);
                if (grid_size == 0 || grid_size > max_grid_size) {
                    std::cerr << "Skipping exteroception log due to unexpected grid size: "
                              << grid_size << " (w=" << width << ", h=" << height << ")"
                              << std::endl;
                    return;
                }

                exteroception_logger->setGridParameters(res, width, height, origin[0], origin[1]);

                // Pre-allocate grid with exact size
                std::vector<float> elevation(grid_size,
                                             std::numeric_limits<float>::quiet_NaN());
                std::vector<float> variance(grid_size,
                                            std::numeric_limits<float>::quiet_NaN());

                // Use integer-based iteration for consistency
                for (uint32_t row = 0; row < height; ++row) {
                    for (uint32_t col = 0; col < width; ++col) {
                        // Calculate world coordinates from grid indices
                        float x = bound_min[0] + col * res;
                        float y = bound_min[1] + row * res;
                        const auto& cell = terrain_estimator->getElevation({x, y});
                        if (cell.has_value()) {
                            const uint32_t idx = row * width + col;
                            if (idx < grid_size) {
                                elevation[idx] = cell.value().height;
                                variance[idx] = cell.value().variance;
                            }
                        }
                    }
                }

                // Verify size
                if (elevation.size() != grid_size) {
                    std::cerr << "Grid size mismatch: expected " << grid_size << ", got "
                              << elevation.size() << std::endl;
                    return;  // Don't log invalid data
                }

                exteroception_logger->log(elevation, variance, ts);
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
    const double imu_timestamp = imu.timestamp;
    const double joint_timestamp = joints.begin()->second.timestamp;

    if (imu_timestamp < last_imu_timestamp_ || abs(imu_timestamp - last_imu_timestamp_) < 1e-6) {
        std::cerr << "IMU measurements are out of order, skipping filtering" << std::endl;
        timers_["total-time"].stop();
        return false;
    }

    if (joint_timestamp < last_joint_timestamp_ ||
        abs(joint_timestamp - last_joint_timestamp_) < 1e-6) {
        std::cerr << "Joint measurements are out of order, skipping filtering" << std::endl;
        timers_["total-time"].stop();
        return false;
    }

    if (abs(imu_timestamp - joint_timestamp) > 1e-3) {
        std::cerr << "IMU and joint timestamps are not synchronized, skipping filtering"
                  << std::endl;
        timers_["total-time"].stop();
        return false;
    }

    timestamp_ = std::min(imu_timestamp, joint_timestamp);
    last_imu_timestamp_ = imu_timestamp;
    last_joint_timestamp_ = joint_timestamp;

    auto ft_timestamp = force_torque.has_value()
        ? std::optional<double>(force_torque.value().begin()->second.timestamp)
        : std::nullopt;
     
    if (ft_timestamp.has_value() &&
        (ft_timestamp.value() < last_ft_timestamp_ ||
         abs(force_torque.value().begin()->second.timestamp - last_ft_timestamp_) < 1e-6)) {
        std::cerr << "Force-torque measurement given is the same, clearing it" << std::endl;
        force_torque.reset();
    } else if (ft_timestamp.has_value()) {
        last_ft_timestamp_ = ft_timestamp.value();
    }

    auto odom_timestamp =
        odom.has_value() ? std::optional<double>(odom.value().timestamp) : std::nullopt;

    if (odom_timestamp.has_value() &&
        (odom_timestamp.value() < last_odom_timestamp_ ||
         abs(odom_timestamp.value() - last_odom_timestamp_) < 1e-6)) {
        std::cerr << "Odometry measurement given is the same, clearing it" << std::endl;
        odom.reset();
    } else if (odom_timestamp.has_value()) {
        last_odom_timestamp_ = odom_timestamp.value();
    }

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
    if (params_.imu_outlier_detection) {
        timers_["imu-outlier-detection"].start();
        bool is_imu_outlier = isImuMeasurementOutlier(imu);
        timers_["imu-outlier-detection"].stop();
        if (is_imu_outlier) {
            timers_["total-time"].stop();
            return false;
        }
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
    frame_tfs_.clear();
    const Eigen::Isometry3d& base_pose = state.getBasePose();
    frame_tfs_[params_.base_frame] = base_pose;

    // Cache the kinematic estimator frame names to avoid repeated calls
    const auto& frame_names = kinematic_estimator_->frameNames();
    for (const auto& frame : frame_names) {
        if (frame != params_.base_frame) {
            try {
                frame_tfs_[frame] = base_pose * kinematic_estimator_->linkTF(frame);
            } catch (const std::exception& e) {
                std::cerr << "Error in frame " << frame << " TF computation: " << e.what()
                          << std::endl;
            }
        }
    }
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
    try {
        // Reset the kinematic estimator explicitly to ensure proper Pinocchio cleanup
        if (kinematic_estimator_) {
            kinematic_estimator_.reset();
        }

        // Reset other estimators
        if (attitude_estimator_) {
            attitude_estimator_.reset();
        }

        if (leg_odometry_) {
            leg_odometry_.reset();
        }

        if (terrain_estimator_) {
            terrain_estimator_.reset();
        }

        // Clear containers
        joint_estimators_.clear();
        if (angular_momentum_derivative_estimator) {
            angular_momentum_derivative_estimator.reset();
        }
        if (gyro_derivative_estimator) {
            gyro_derivative_estimator.reset();
        }
        contact_estimators_.clear();

        // Clear other containers
        frame_tfs_.clear();
        imu_outlier_detector_.clear();
        timers_.clear();
    } catch (...) {
        // If anything goes wrong during destruction, just continue
        // This prevents crashes during cleanup
        std::cerr << "Warning: Exception during Serow destruction" << std::endl;
    }
}

bool Serow::isInitialized() const {
    return is_initialized_;
}

void Serow::setState(const State& state) {
    reset();
    base_estimator_.setState(state.base_state_);
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
            params_.initial_base_position_cov.asDiagonal();
        if (!state_.isPointFeet()) {
            contacts_orientation_cov[cf] = params_.initial_base_orientation_cov.asDiagonal();
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
    base_estimator_.init(state_.base_state_, state_.getContactsFrame(), params_.g, params_.imu_rate, 
                         params_.use_imu_orientation, params_.verbose);

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
    const double MAD_THRESHOLD_MULTIPLIER = 6.0;
    const double MAD_TO_STD_FACTOR = 1.4826;

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

    // Pre-allocate vector to avoid repeated allocations
    static thread_local std::vector<double> deviations;
    deviations.clear();
    deviations.reserve(200);  // Reserve maximum expected size

    // Calculate MAD for each axis more efficiently
    Eigen::Vector3d av_mad = Eigen::Vector3d::Zero();
    Eigen::Vector3d la_mad = Eigen::Vector3d::Zero();

    for (size_t i = 0; i < 3; i++) {
        const auto& av_window = imu_outlier_detector_[i].getWindow();
        const auto& la_window = imu_outlier_detector_[i + 3].getWindow();

        // Calculate MAD for angular velocity
        deviations.clear();
        for (double value : av_window) {
            deviations.push_back(std::abs(value - av_median[i]));
        }
        std::nth_element(deviations.begin(), deviations.begin() + deviations.size() / 2,
                         deviations.end());
        av_mad[i] = deviations[deviations.size() / 2];

        // Calculate MAD for linear acceleration
        deviations.clear();
        for (double value : la_window) {
            deviations.push_back(std::abs(value - la_median[i]));
        }
        std::nth_element(deviations.begin(), deviations.begin() + deviations.size() / 2,
                         deviations.end());
        la_mad[i] = deviations[deviations.size() / 2];
    }

    // Convert MAD to standard deviation approximation
    const Eigen::Vector3d av_std_dev(av_mad * MAD_TO_STD_FACTOR);
    const Eigen::Vector3d la_std_dev(la_mad * MAD_TO_STD_FACTOR);

    // Calculate z-scores for current measurement
    const Eigen::Vector3d av_z_scores =
        (imu.angular_velocity - av_median).cwiseQuotient(av_std_dev);
    const Eigen::Vector3d la_z_scores =
        (imu.linear_acceleration - la_median).cwiseQuotient(la_std_dev);

    // Check if any component exceeds the threshold using SIMD-friendly operations
    return (av_z_scores.array().abs() > MAD_THRESHOLD_MULTIPLIER).any() ||
        (la_z_scores.array().abs() > MAD_THRESHOLD_MULTIPLIER).any();
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

// RL-specific functions
std::tuple<ImuMeasurement, KinematicMeasurement, std::map<std::string, ForceTorqueMeasurement>>
Serow::processMeasurements(
    ImuMeasurement imu, std::map<std::string, JointMeasurement> joints,
    std::optional<std::map<std::string, ForceTorqueMeasurement>> force_torque,
    std::optional<std::map<std::string, ContactMeasurement>> contacts_probability) {
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

    // Update the joint state estimate
    runJointsEstimator(state_, joints);

    // Estimate the base frame attitude and initial IMU biases
    runImuEstimator(state_, imu);

    // Update the kinematic structure
    KinematicMeasurement kin = runForwardKinematics(state_);
    // filter() uses timestamp_ instead which is not set here
    kin.timestamp = joints.begin()->second.timestamp;

    // Estimate the contact state
    if (!ft.empty()) {
        runContactEstimator(state_, ft, kin, contacts_probability);
    }

    // Compute the leg odometry and update the kinematic measurement accordingly
    computeLegOdometry(state_, imu, kin);

    // Return the measurements
    return std::make_tuple(imu, kin, ft);
}

void Serow::baseEstimatorPredictStep(const ImuMeasurement& imu, const KinematicMeasurement& kin) {
    // Initialize terrain estimator if needed
    if (params_.enable_terrain_estimation && !terrain_estimator_) {
        double terrain_height = 0.0;
        double den = 0;
        Eigen::Isometry3d T_world_to_base = Eigen::Isometry3d::Identity();
        T_world_to_base.translation() = state_.base_state_.base_position;
        T_world_to_base.linear() = state_.base_state_.base_orientation.toRotationMatrix();
        for (const auto& [cf, cp] : kin.contacts_probability) {
            terrain_height += cp * (T_world_to_base * kin.contacts_position.at(cf)).z();
            den += cp;
        }

        if (den > 0) {
            terrain_height /= den;
            // Initialize terrain elevation mapper
            if (params_.terrain_estimator_type == "naive") {
                terrain_estimator_ = std::make_shared<NaiveLocalTerrainMapper>();
            } else if (params_.terrain_estimator_type == "fast") {
                terrain_estimator_ = std::make_shared<LocalTerrainMapper>();
            } else {
                throw std::runtime_error("Invalid terrain estimator type: " +
                                        params_.terrain_estimator_type);
            }
            terrain_estimator_->initializeLocalMap(static_cast<float>(terrain_height), 1e4,
                                                   params_.minimum_terrain_height_variance);
            terrain_estimator_->recenter({static_cast<float>(state_.base_state_.base_position.x()),
                                          static_cast<float>(state_.base_state_.base_position.y())});
        }
    }

    // Call the base estimator predict step
    state_.base_state_.timestamp = imu.timestamp;
    base_estimator_.predict(state_.base_state_, imu);
}

void Serow::baseEstimatorUpdateWithBaseLinearVelocity(const KinematicMeasurement& kin) {
    state_.base_state_.timestamp = kin.timestamp;
    base_estimator_.updateWithBaseLinearVelocity(state_.base_state_, kin.base_linear_velocity,
                                                 kin.base_linear_velocity_cov);
}

void Serow::baseEstimatorUpdateWithImuOrientation(const ImuMeasurement& imu) {
    state_.base_state_.timestamp = imu.timestamp;
    base_estimator_.updateWithIMUOrientation(state_.base_state_, imu.orientation,
                                             imu.orientation_cov);
}

void Serow::baseEstimatorFinishUpdate(const ImuMeasurement& imu, const KinematicMeasurement& kin) {
    const Eigen::Isometry3d base_pose = state_.getBasePose();
    // Estimate base angular velocity and linear acceleration
    const Eigen::Vector3d base_angular_velocity =
        imu.angular_velocity - state_.getImuAngularVelocityBias();
    const Eigen::Vector3d base_linear_acceleration =
        base_pose.linear() * (imu.linear_acceleration - state_.getImuLinearAccelerationBias()) -
        Eigen::Vector3d(0.0, 0.0, params_.g);
    if (!gyro_derivative_estimator) {
        gyro_derivative_estimator = std::make_unique<DerivativeEstimator>(
            "Gyro Derivative", coeffs_imu_, params_.imu_rate, 3);
        if (state_.isInitialized()) {
            const Eigen::Matrix3d R_base_to_world = base_pose.linear().transpose();
            gyro_derivative_estimator->setState(
                R_base_to_world * state_.base_state_.base_angular_acceleration);
        } else {
            gyro_derivative_estimator->setState(Eigen::Vector3d::Zero());
        }
    }
    const Eigen::Vector3d base_angular_acceleration =
        gyro_derivative_estimator->filter(base_angular_velocity, imu.angular_velocity_cov.diagonal(), imu.timestamp);
    state_.base_state_.base_angular_velocity = base_pose.linear() * base_angular_velocity;
    state_.base_state_.base_angular_acceleration = base_pose.linear() * base_angular_acceleration;
    state_.base_state_.base_linear_acceleration = base_linear_acceleration;

    // Update feet pose/velocity in world frame
    for (const auto& frame : state_.getContactsFrame()) {
        // Cache calculations
        const Eigen::Vector3d& base_foot_pos = kin.base_to_foot_positions.at(frame);
        const Eigen::Vector3d transformed_pos = base_pose.linear() * base_foot_pos;

        state_.base_state_.feet_position[frame].noalias() = base_pose * base_foot_pos;
        state_.base_state_.feet_orientation[frame] = Eigen::Quaterniond(
            base_pose.linear() * kin.base_to_foot_orientations.at(frame).toRotationMatrix());

        state_.base_state_.feet_linear_velocity[frame].noalias() =
            state_.base_state_.base_linear_velocity +
            state_.base_state_.base_angular_velocity.cross(transformed_pos) +
            base_pose.linear() * kin.base_to_foot_linear_velocities.at(frame);

        state_.base_state_.feet_angular_velocity[frame].noalias() =
            state_.base_state_.base_angular_velocity +
            base_pose.linear() * kin.base_to_foot_angular_velocities.at(frame);
    }

    // Update all frame transformations
    updateFrameTree(state_);

    // Check if state has converged
    if (!state_.is_valid_ && cycle_++ > params_.convergence_cycles) {
        state_.is_valid_ = true;
    }
}

}  // namespace serow
