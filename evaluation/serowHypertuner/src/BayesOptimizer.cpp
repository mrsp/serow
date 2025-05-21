#include <BayesOptimizer.hpp>

BayesOptimizer::BayesOptimizer(DataManager& dataManager, 
  size_t dim, 
  const bayesopt::Parameters& params,
  const std::vector<std::string> &params2optimize_)
: data_(dataManager), ContinuousModel(dim,params), params2optimize_(params2optimize_)
{    
    original_config = data_.getSerowPath() + "config/" + data_.getRobotName() + ".json";
    temp_config = data_.getSerowPath() + "config/" + data_.getRobotName() +"_temp.json";
    robot_frame_config = data_.getSerowPath() + "evaluation/serowHypertuner/config/robots/" + data_.getRobotName() + ".json";
    
    setRobotJointsAndFootFrames(robot_frame_config);
    
    printRobotInfo();

    // Load the ground truth data
    gt_position_ = data_.getPosData();
    gt_orientation_ = data_.getRotData();
    
    timestamps_ = data_.getTimestamps();
    force_measurements_ = data_.getForceData();
    joint_positions_ = data_.getJointStatesData();
    joint_velocities_ = data_.getJointVelocitiesData();
    linear_acceleration_ = data_.getLinAccData();
    angular_velocity_ = data_.getAngVelData();

    std::cout << "Loaded " << gt_position_.size() << " ground truth positions." << std::endl;
    std::cout << "Loaded " << gt_orientation_.size() << " ground truth orientations." << std::endl;
    std::cout << "Loaded " << timestamps_.size() << " timestamps." << std::endl;
    std::cout << "Loaded " << force_measurements_.FR.size() << " force measurements." << std::endl;
    std::cout << "Loaded " << joint_positions_.size() << " joint positions." << std::endl;
    std::cout << "Loaded " << joint_velocities_.size() << " joint velocities." << std::endl;
    std::cout << "Loaded " << linear_acceleration_.size() << " linear acceleration." << std::endl;
    std::cout << "Loaded " << angular_velocity_.size() << " angular velocity." << std::endl;
    std::cout << "Loaded " << joint_names_.size() << " joint names." << std::endl;
    std::cout << "Loaded " << foot_names_.size() << " foot frames." << std::endl;
}

void BayesOptimizer::writeJSONConfig(const json& j, const std::string& path)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to write temporary config file.");
    }
    out << std::setw(2) << j << std::endl;
}

json BayesOptimizer::loadDefaultJson(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open default config file.");
    }
    json j;
    in >> j;
    return j;
}

void BayesOptimizer::saveBestConfig(vectord result){
    std::string best_config_file = data_.getSerowPath() + "config/" + data_.getRobotName() + "_best_config.json";
    std::cout << "Saving best config to: " << best_config_file << std::endl;

      // SAVE BEST CONFIG
    json best_config = loadDefaultJson(original_config);

    // Save best config using params2optimize_
    for (size_t i = 0; i < params2optimize_.size(); ++i) {
        const std::string& param = params2optimize_[i];
        
        size_t open_bracket = param.find('[');
        size_t close_bracket = param.find(']');

        if (open_bracket != std::string::npos && close_bracket != std::string::npos && close_bracket > open_bracket) {
            // It's an array element like "imu_covariance[2]"
            std::string base_name = param.substr(0, open_bracket);
            int index = std::stoi(param.substr(open_bracket + 1, close_bracket - open_bracket - 1));

            if (best_config.contains(base_name) && best_config[base_name].is_array() && index < best_config[base_name].size()) {
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
    for (size_t i = 0; i < params2optimize_.size(); ++i) {
        std::cout << params2optimize_[i] << ": " << result(i) << std::endl;
    }
}

void BayesOptimizer::setRobotJointsAndFootFrames(const std::string & path){
    try
    {
        std::ifstream file(path);
        if (!file.is_open())
        {
            throw std::runtime_error( "Could not open config file: " );
        }

        nlohmann::json config;
        file >> config;

        // Read joint names
        joint_names_.clear();
        for (const auto& name : config["joint_names"]){
            joint_names_.emplace_back(name);
        }

        // Read foot frames
        foot_names_.clear();
        for (const auto& name : config["foot_frames"]){
            foot_names_.emplace_back(name);
        }

        std::cout << "Loaded " << joint_names_.size() << " joint names and "
                << foot_names_.size() << " foot frames." << std::endl;
        return;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return;
    }
}

void BayesOptimizer::printRobotInfo() {
    std::cout << "Robot Name: " << data_.getRobotName() << std::endl;
    std::cout << "Joint names: \n";
    for (const auto& joint_name : joint_names_) {
       std::cout << joint_name << std::endl;
    }
    std::cout << "Foot frames: \n";
    for (const auto& foot_frame : foot_names_) {
       std::cout << foot_frame << std::endl;
    }
}
Eigen::Vector3d BayesOptimizer::logMap(const Eigen::Matrix3d& R) {
    double R11 = R(0, 0);
    double R12 = R(0, 1);
    double R13 = R(0, 2);
    double R21 = R(1, 0);
    double R22 = R(1, 1);
    double R23 = R(1, 2);
    double R31 = R(2, 0);
    double R32 = R(2, 1);
    double R33 = R(2, 2);

    double trace = R.trace();

    Eigen::Vector3d omega = Eigen::Vector3d::Zero();

    // Special case when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
    if (trace + 1.0 < 1e-3) {
        if (R33 > R22 && R33 > R11) {
            // R33 is the largest diagonal, a=3, b=1, c=2
            double W = R21 - R12;
            double Q1 = 2.0 + 2.0 * R33;
            double Q2 = R31 + R13;
            double Q3 = R23 + R32;
            double r = std::sqrt(Q1);
            double one_over_r = 1 / r;
            double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            double sgn_w = W < 0 ? -1.0 : 1.0;
            double mag = M_PI - (2 * sgn_w * W) / norm;
            double scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * Eigen::Vector3d(Q2, Q3, Q1);
        }
        else if (R22 > R11) {
            // R22 is the largest diagonal, a=2, b=3, c=1
            double W = R13 - R31;
            double Q1 = 2.0 + 2.0 * R22;
            double Q2 = R23 + R32;
            double Q3 = R12 + R21;
            double r = std::sqrt(Q1);
            double one_over_r = 1 / r;
            double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            double sgn_w = W < 0 ? -1.0 : 1.0;
            double mag = M_PI - (2 * sgn_w * W) / norm;
            double scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * Eigen::Vector3d(Q3, Q1, Q2);
        }
        else {
            // R11 is the largest diagonal, a=1, b=2, c=3
            double W = R32 - R23;
            double Q1 = 2.0 + 2.0 * R11;
            double Q2 = R12 + R21;
            double Q3 = R31 + R13;
            double r = std::sqrt(Q1);
            double one_over_r = 1 / r;
            double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
            double sgn_w = W < 0 ? -1.0 : 1.0;
            double mag = M_PI - (2 * sgn_w * W) / norm;
            double scale = 0.5 * one_over_r * mag;
            omega = sgn_w * scale * Eigen::Vector3d(Q1, Q2, Q3);
        }
    }
    else {
        double magnitude = 0.0;
        double tr_3 = trace - 3.0;  // could be non-negative if the matrix is off orthogonal
        if (tr_3 < -1e-6) {
            // this is the normal case -1 < trace < 3
            double theta = std::acos((trace - 1.0) / 2.0);
            magnitude = theta / (2.0 * std::sin(theta));
        }
        else {
            // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
            // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
            magnitude = 0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0;
        }

        omega = magnitude * Eigen::Vector3d(R32 - R23, R13 - R31, R21 - R12);
    }
    return omega;
}

double BayesOptimizer::computeATE() {
    // std::cout << "Computing ATE..." << std::endl;
    // std::cout << "Ground truth position size: " << gt_position_.size() << std::endl;
    // std::cout << "Estimation position size: " << est_positions_.size() << std::endl;
    // std::cout << "Ground truth orientation size: " << gt_orientation_.size() << std::endl;
    // std::cout << "Estimation orientation size: " << est_orientations_.size() << std::endl;
    
    // Ensure we have data to compare
    if (gt_position_.empty() || est_positions_.empty() || gt_orientation_.empty() || est_orientations_.empty()) {
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
        position_error_sum += std::sqrt(squared_position_distance); // L2 norm of position error
        
        // Calculate orientation error using the logMap function
        Eigen::Quaterniond gt_quat(gt_orientation_[i][3], gt_orientation_[i][0], 
                                gt_orientation_[i][1], gt_orientation_[i][2]);
        Eigen::Quaterniond est_quat(est_orientations_[i][3], est_orientations_[i][0], 
                                    est_orientations_[i][1], est_orientations_[i][2]);
        
        Eigen::Matrix3d gt_R = gt_quat.toRotationMatrix();
        Eigen::Matrix3d est_R = est_quat.toRotationMatrix();
        
        // Apply logMap to calculate orientation error (similar to the reward function)
        Eigen::Matrix3d rel_R = gt_R.transpose() * est_R;
        Eigen::Vector3d omega = logMap(rel_R);
        double orientation_error = omega.norm();
        orientation_error_sum += orientation_error;
    }

    double avg_position_error = position_error_sum / count;
    double avg_orientation_error = orientation_error_sum / count;
    double position_weight = 1.0;  
    double orientation_weight = 0.5;
    double combined_ate = position_weight * avg_position_error + 
                        orientation_weight * avg_orientation_error;
    
    std::cout << "Position ATE: " << avg_position_error << std::endl;
    std::cout << "Orientation ATE: " << avg_orientation_error << std::endl;
    std::cout << "Combined ATE: " << combined_ate << std::endl;

    return combined_ate;
}


double BayesOptimizer::evaluateSample(const vectord &x) {
    n_iterations_ ++;
    std::cout << "\033[32m \nOptimization iteration:\033[0m" << n_iterations_  << std::endl;
    
    // Reinitialize estimations in each iteration
    est_positions_.clear();
    est_orientations_.clear();

    temp_json = loadDefaultJson(original_config);

    // Updates the tmp json with the new parameters
    for (size_t i = 0; i < params2optimize_.size(); ++i)
    {
        const std::string& name = params2optimize_[i];

        // Check if the parameter is an array item like "foo[1]"
        size_t open_bracket = name.find('[');
        size_t close_bracket = name.find(']');

        if (open_bracket != std::string::npos && close_bracket != std::string::npos)
        {
            // Extract key and index
            std::string key = name.substr(0, open_bracket);
            int index = std::stoi(name.substr(open_bracket + 1, close_bracket - open_bracket - 1));

            // Initialize or resize array if needed
            if (!temp_json.contains(key) || !temp_json[key].is_array())
            {
                temp_json[key] = std::vector<double>(index + 1, 0.0);
            }

            temp_json[key][index] = x(i);
        }
        else
        {
            // Simple scalar assignment
            temp_json[name] = x(i);
        }
    }

    writeJSONConfig(temp_json, temp_config);
    
    // Initialize Serow with the modified config
    if (!SEROW.initialize(data_.getRobotName()+"_temp.json")) {
        throw std::runtime_error("Failed to initialize Serow with config: " + temp_config);
    }

    for (int i = 0; i < timestamps_.size() - 1 ; ++i) {
        double timestamp = timestamps_[i][0];
        std::map<std::string, serow::ForceTorqueMeasurement> force_torque;
        force_torque.insert(
            { foot_names_[0],
                serow::ForceTorqueMeasurement{
                    .timestamp = timestamp,
                    .force = Eigen::Vector3d(force_measurements_.FL[i][0], force_measurements_.FL[i][1],
                                            force_measurements_.FL[i][2])}});
       
        force_torque.insert(
            { foot_names_[1],
                serow::ForceTorqueMeasurement{
                    .timestamp = timestamp,
                    .force = Eigen::Vector3d(force_measurements_.FR[i][0], force_measurements_.FR[i][1],
                                            force_measurements_.FR[i][2])}});

        force_torque.insert(
            { foot_names_[2],
                serow::ForceTorqueMeasurement{
                    .timestamp = timestamp,
                    .force = Eigen::Vector3d(force_measurements_.RL[i][0], force_measurements_.RL[i][1],
                                            force_measurements_.RL[i][2])}});
        force_torque.insert(
            { foot_names_[3],
                serow::ForceTorqueMeasurement{
                    .timestamp = timestamp,
                    .force = Eigen::Vector3d(force_measurements_.RR[i][0], force_measurements_.RR[i][1],
                                            force_measurements_.RR[i][2])}});

        serow::ImuMeasurement imu;
        imu.timestamp = timestamp;
        imu.linear_acceleration = Eigen::Vector3d(linear_acceleration_[i][0], linear_acceleration_[i][1], linear_acceleration_[i][2]);
        imu.angular_velocity = Eigen::Vector3d(angular_velocity_[i][0], angular_velocity_[i][1], angular_velocity_[i][2]);

        std::map<std::string, serow::JointMeasurement> joints;
        joints.insert({joint_names_[0],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][0], .velocity = joint_velocities_[i][0]}});
        joints.insert({joint_names_[1],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][1], .velocity = joint_velocities_[i][1]}});
        joints.insert({joint_names_[2],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][2], .velocity = joint_velocities_[i][2]}});
        joints.insert({joint_names_[3],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][3], .velocity = joint_velocities_[i][3]}});
        joints.insert({joint_names_[4],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][4], .velocity = joint_velocities_[i][4]}});
        joints.insert({joint_names_[5],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][5], .velocity = joint_velocities_[i][5]}});
        joints.insert({joint_names_[6],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][6], .velocity = joint_velocities_[i][6]}});
        joints.insert({joint_names_[7],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][7], .velocity = joint_velocities_[i][7]}});
        joints.insert({joint_names_[8],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][8], .velocity = joint_velocities_[i][8]}});
        joints.insert({joint_names_[9],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][9], .velocity = joint_velocities_[i][9]}});
        joints.insert({joint_names_[10],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][10], .velocity = joint_velocities_[i][10]}});
        joints.insert({joint_names_[11],
                        serow::JointMeasurement{.timestamp = timestamp,
                                                .position = joint_positions_[i][11], .velocity = joint_velocities_[i][11]}});

        SEROW.filter(imu, joints, force_torque);

        auto state = SEROW.getState();
        if (!state.has_value()) {
            continue;
        }
        auto basePos = state->getBasePosition();
        auto baseOrient = state->getBaseOrientation();
        est_positions_.push_back({basePos.x(), basePos.y(), basePos.z()});
        est_orientations_.push_back({baseOrient.x(), baseOrient.y(), baseOrient.z(), baseOrient.w()});
    }
    std::cout << "Estimation size: " << est_positions_.size() << std::endl;
    std::cout << "Ground truth size: " << gt_position_.size() << std::endl;

    return computeATE(); // Replace with your real evaluation logic
}
