#include <BayesOptimizer.hpp>

BayesOptimizer::BayesOptimizer(DataManager& data_manager, size_t dim,
                               const bayesopt::Parameters& params, const double ate_position_weight,
                               const double ate_orientation_weight,
                               const std::vector<std::string>& params_to_optimize)
    : data_(data_manager),
      foot_frames_(data_.getFootFrames()),
      joint_names_(data_.getJointNames()),
      serow_path_(data_.getSerowPath()),
      ContinuousModel(dim, params),
      params_to_optimize_(params_to_optimize),
      ate_position_weight_(ate_position_weight),
      ate_orientation_weight_(ate_orientation_weight) {
    std::cout << serow_path_ << std::endl;
    original_config_ = serow_path_ + "config/" + data_.getRobotName() + ".json";
    temp_config_ = serow_path_ + "config/" + data_.getRobotName() + "_temp.json";

    printRobotInfo();

    // Load the ground truth data
    gt_position_ = data_.getBasePositionData();
    gt_orientation_ = data_.getBaseOrientationData();

    timestamps_ = data_.getTimestamps();
    ft_measurements_ = data_.getForceTorqueData();
    joint_positions_ = data_.getJointStatesData();
    joint_velocities_ = data_.getJointVelocitiesData();
    linear_acceleration_ = data_.getImuLinearAccelerationData();
    angular_velocity_ = data_.getImuAngularVelocityData();

    std::cout << "Loaded " << gt_position_.size() << " ground truth positions." << std::endl;
    std::cout << "Loaded " << gt_orientation_.size() << " ground truth orientations." << std::endl;
    std::cout << "Loaded " << timestamps_.size() << " timestamps." << std::endl;
    std::cout << "Loaded " << joint_positions_.size() << " joint positions." << std::endl;
    std::cout << "Loaded " << joint_velocities_.size() << " joint velocities." << std::endl;
    std::cout << "Loaded " << linear_acceleration_.size() << " linear acceleration." << std::endl;
    std::cout << "Loaded " << angular_velocity_.size() << " angular velocity." << std::endl;
    std::cout << "Loaded " << joint_names_.size() << " joint names." << std::endl;
    std::cout << "Loaded " << foot_frames_.size() << " foot frames." << std::endl;
}

void BayesOptimizer::writeJSONConfig(const json& j, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to write temporary config file.");
    }
    out << std::setw(2) << j << std::endl;
}

json BayesOptimizer::loadDefaultJson(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open default config file.");
    }
    json j;
    in >> j;
    return j;
}

void BayesOptimizer::saveBestConfig(const vectord& result) {
    std::string best_config_file =
        serow_path_ + "config/" + data_.getRobotName() + "_best_config.json";
    std::cout << "Saving best config to: " << best_config_file << std::endl;

    json best_config = loadDefaultJson(original_config_);

    // Save best config using params_to_optimize_
    for (size_t i = 0; i < params_to_optimize_.size(); ++i) {
        const std::string& param = params_to_optimize_[i];

        size_t open_bracket = param.find('[');
        size_t close_bracket = param.find(']');

        if (open_bracket != std::string::npos && close_bracket != std::string::npos &&
            close_bracket > open_bracket) {
            // It's an array element like "imu_covariance[2]"
            std::string base_name = param.substr(0, open_bracket);
            int index = std::stoi(param.substr(open_bracket + 1, close_bracket - open_bracket - 1));

            if (best_config.contains(base_name) && best_config[base_name].is_array() &&
                index < best_config[base_name].size()) {
                best_config[base_name][index] = result(i);
            } else {
                std::cerr << "Warning: Invalid array parameter '" << param << "' in config.\n";
            }
        } else {
            // Regular scalar parameter
            if (best_config.contains(param)) {
                best_config[param] = result(i);
            } else {
                std::cerr << "Warning: Parameter '" << param << "' not found in config.\n";
            }
        }
    }

    // Save best config to file (optional: different path like "best_config.json")
    writeJSONConfig(best_config, best_config_file);

    std::cout << "Best parameters found:\n";
    for (size_t i = 0; i < params_to_optimize_.size(); ++i) {
        std::cout << params_to_optimize_[i] << ": " << result(i) << std::endl;
    }
}

void BayesOptimizer::printRobotInfo() {
    std::cout << "Robot Name: " << data_.getRobotName() << std::endl;
    std::cout << "Joint names: \n";
    for (const auto& joint_name : joint_names_) {
        std::cout << joint_name << std::endl;
    }
    std::cout << "Foot frames: \n";
    for (const auto& foot_frame : foot_frames_) {
        std::cout << foot_frame << std::endl;
    }
}

double BayesOptimizer::computeATE() {
    // Ensure we have data to compare
    if (gt_position_.empty() || est_positions_.empty() || gt_orientation_.empty() ||
        est_orientations_.empty()) {
        std::cerr << "Error: Empty position data" << std::endl;
        return std::numeric_limits<double>::max();
    }

    // Use the minimum size in case they differ
    size_t count = std::min(gt_position_.size(), est_positions_.size());
    // Sum of squared position errors and orientation errors
    double position_error_sum = 0.0;
    double orientation_error_sum = 0.0;

    for (size_t i = 0; i < count; ++i) {
        // Calculate squared Euclidean distance for position
        double squared_position_distance = 0.0;
        size_t dims = std::min(gt_position_[i].size(), est_positions_[i].size());

        for (size_t j = 0; j < dims; ++j) {
            double diff = gt_position_[i][j] - est_positions_[i][j];
            squared_position_distance += diff * diff;
        }
        position_error_sum += std::sqrt(squared_position_distance);  // L2 norm of position error

        // Calculate orientation error using the logMap function
        const Eigen::Quaterniond gt_quat(gt_orientation_[i][3], gt_orientation_[i][0],
                                         gt_orientation_[i][1], gt_orientation_[i][2]);
        const Eigen::Quaterniond est_quat(est_orientations_[i][3], est_orientations_[i][0],
                                          est_orientations_[i][1], est_orientations_[i][2]);

        const Eigen::Matrix3d gt_R = gt_quat.toRotationMatrix();
        const Eigen::Matrix3d est_R = est_quat.toRotationMatrix();

        // Apply logMap to calculate orientation error (similar to the reward function)
        const Eigen::Matrix3d rel_R = est_R.transpose() * gt_R;
        const Eigen::Vector3d omega = lie::so3::logMap(rel_R);
        orientation_error_sum += omega.norm();
    }

    const double avg_position_error = position_error_sum / count;
    const double avg_orientation_error = orientation_error_sum / count;
    const double combined_ate =
        ate_position_weight_ * avg_position_error + ate_orientation_weight_ * avg_orientation_error;

    std::cout << "Position ATE: " << avg_position_error << std::endl;
    std::cout << "Orientation ATE: " << avg_orientation_error << std::endl;
    std::cout << "Combined ATE: " << combined_ate << std::endl;

    return combined_ate;
}

double BayesOptimizer::evaluateSample(const vectord& x) {
    n_iterations_++;
    std::cout << "\033[32m \nOptimization iteration:\033[0m" << n_iterations_ << std::endl;

    // Reinitialize estimations in each iteration
    est_positions_.clear();
    est_orientations_.clear();

    temp_json_ = loadDefaultJson(original_config_);

    // Updates the tmp json with the new parameters
    for (size_t i = 0; i < params_to_optimize_.size(); ++i) {
        const std::string& name = params_to_optimize_[i];

        // Check if the parameter is an array item like "foo[1]"
        size_t open_bracket = name.find('[');
        size_t close_bracket = name.find(']');

        if (open_bracket != std::string::npos && close_bracket != std::string::npos) {
            // Extract key and index
            std::string key = name.substr(0, open_bracket);
            int index = std::stoi(name.substr(open_bracket + 1, close_bracket - open_bracket - 1));

            // Initialize or resize array if needed
            if (!temp_json_.contains(key) || !temp_json_[key].is_array()) {
                temp_json_[key] = std::vector<double>(index + 1, 0.0);
            }

            temp_json_[key][index] = x(i);
        } else {
            // Simple scalar assignment
            temp_json_[name] = x(i);
        }
    }

    writeJSONConfig(temp_json_, temp_config_);

    // Initialize Serow with the modified config
    if (!estimator_.initialize(data_.getRobotName() + "_temp.json")) {
        throw std::runtime_error("Failed to initialize Serow with config: " + temp_config_);
    }

    for (int i = 0; i < timestamps_.size() * dataset_percentage_ - 1; ++i) {
        const double timestamp = timestamps_[i];
        std::map<std::string, serow::ForceTorqueMeasurement> force_torque;

        for (const auto& foot_frame : foot_frames_) {
            force_torque.insert(
                {foot_frame,
                 serow::ForceTorqueMeasurement{
                     .timestamp = timestamp, .force = ft_measurements_.force.data[foot_frame][i]}});
        }

        serow::ImuMeasurement imu;
        imu.timestamp = timestamp;
        imu.linear_acceleration = Eigen::Vector3d(
            linear_acceleration_[i].x(), linear_acceleration_[i].y(), linear_acceleration_[i].z());
        imu.angular_velocity = Eigen::Vector3d(angular_velocity_[i].x(), angular_velocity_[i].y(),
                                               angular_velocity_[i].z());

        std::map<std::string, serow::JointMeasurement> joints;
        for (int j = 0; j < joint_names_.size(); ++j) {
            joints.insert({joint_names_[j],
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions_[i][j],
                                                   .velocity = joint_velocities_[i][j]}});
        }

        estimator_.filter(imu, joints, force_torque);

        const auto state = estimator_.getState();
        if (!state.has_value()) {
            continue;
        }
        const Eigen::Vector3d base_position = state->getBasePosition();
        const Eigen::Quaterniond base_quaternion = state->getBaseOrientation();
        est_positions_.push_back({base_position.x(), base_position.y(), base_position.z()});
        est_orientations_.push_back(
            {base_quaternion.x(), base_quaternion.y(), base_quaternion.z(), base_quaternion.w()});
    }
    std::cout << "Estimation size: " << est_positions_.size() << std::endl;
    std::cout << "Ground truth size: " << gt_position_.size() << std::endl;

    return computeATE();
}

BayesOptimizer::~BayesOptimizer() {
    // Clean up if necessary
    if (std::remove(temp_config_.c_str()) != 0) {
        std::cerr << "Error deleting temporary config file: " << temp_config_ << std::endl;
    } else {
        std::cout << "Temporary config file deleted successfully." << std::endl;
    }
}
