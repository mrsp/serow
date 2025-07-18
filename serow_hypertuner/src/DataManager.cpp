#include "DataManager.hpp"

DataManager::DataManager() {
    loadConfig();

    initializeFTMeasurements();

    loadData();

    std::cout << "Data Manager initialized." << std::endl;
}

void DataManager::loadData() {
    std::cout << "Loading data from " << data_file_ << std::endl;

    // Load data from HDF5 files
    base_position_ = readHDF5(data_file_, gt_base_position_dataset_);
    base_orientation_ = readHDF5(data_file_, gt_base_orientation_dataset_);

    linear_acceleration_ =
        convertToEigenVec3(readHDF5(data_file_, imu_linear_acceleration_dataset_));
    angular_velocity_ = convertToEigenVec3(readHDF5(data_file_, imu_angular_velocity_dataset_));
    joint_states_ = readHDF5(data_file_, joint_position_dataset_);
    joint_velocities_ = readHDF5(data_file_, joint_velocity_dataset_);

    // Load force data
    for (int i = 0; i < foot_frames_.size(); ++i) {
        const std::string& foot_frame = foot_frames_[i];
        force_torque_.force.data[foot_frame] =
            convertToEigenVec3(readHDF5(data_file_, feet_force_dataset_[i]));
    }

    timestamps_ = readHDF5_1D(data_file_, timestamps_dataset_);
}

void DataManager::loadConfig() {
    std::cout << "Loading configuration file: experimentConfig.json..." << std::endl;
    std::string file_path = getSerowPath() + "/serow_hypertuner/config/experimentConfig.json";
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + file_path);
    }

    file >> config_;

    data_file_ = resolvePath(config_, config_["Paths"]["data_file"]);

    imu_linear_acceleration_dataset_ = config_["Dataset"]["imu_linear_acceleration"];
    imu_angular_velocity_dataset_ = config_["Dataset"]["imu_angular_velocity"];
    joint_position_dataset_ = config_["Dataset"]["joint_position"];
    joint_velocity_dataset_ = config_["Dataset"]["joint_velocity"];
    gt_base_position_dataset_ = config_["Dataset"]["base_ground_truth_position"];
    gt_base_orientation_dataset_ = config_["Dataset"]["base_ground_truth_orientation"];
    timestamps_dataset_ = config_["Dataset"]["timestamps"];
    feet_force_dataset_ = config_["Dataset"]["feet_force"].get<std::vector<std::string>>();

    robot_frame_config_ =
        getSerowPath() + "serow_hypertuner/config/robots/" + getRobotName() + "_frames.json";

    try {
        std::ifstream file(robot_frame_config_);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open config file: ");
        }

        nlohmann::json config;
        file >> config;

        // Read joint names
        joint_names_.clear();
        for (const auto& name : config["joint_names"]) {
            joint_names_.emplace_back(name);
        }

        // Read foot frames
        foot_frames_.clear();
        for (const auto& name : config["foot_frames"]) {
            foot_frames_.emplace_back(name);
        }

        std::cout << "Loaded " << joint_names_.size() << " joint names and " << foot_frames_.size()
                  << " foot frames." << std::endl;
        return;
    } catch (const std::exception& e) {
        std::cerr << "Error loading config: " << e.what() << std::endl;
        return;
    }
}

void DataManager::initializeFTMeasurements() {
    for (const auto& foot : foot_frames_) {
        force_torque_.force.data[foot] = {};
        force_torque_.torque.data[foot] = {};
    }
}

std::string DataManager::resolvePath(const json& config, const std::string& path) {
    std::string serow_path_env = getSerowPath();
    std::string resolved_path = serow_path_env + path;

    // Extract all placeholders from the config
    base_path_ = config["Paths"]["base_path"];
    experiment_type_ = config["Experiment"]["experiment_type"];
    robot_name_ = config["Experiment"]["robot_name"];
    experiment_name_ = config["Experiment"]["experiment_name"];

    // Replace placeholders
    resolved_path = std::regex_replace(resolved_path, std::regex("\\{base_path\\}"), base_path_);
    resolved_path =
        std::regex_replace(resolved_path, std::regex("\\{experiment_type\\}"), experiment_type_);
    resolved_path =
        std::regex_replace(resolved_path, std::regex("\\{type\\}"), experiment_type_);  // optional
    resolved_path = std::regex_replace(resolved_path, std::regex("\\{robot_name\\}"), robot_name_);
    resolved_path =
        std::regex_replace(resolved_path, std::regex("\\{experiment_name\\}"), experiment_name_);

    return resolved_path;
}

std::vector<Eigen::Vector3d> DataManager::convertToEigenVec3(
    const std::vector<std::vector<double>>& data) {
    std::vector<Eigen::Vector3d> result;
    result.reserve(data.size());

    for (const auto& row : data) {
        if (row.size() != 3) {
            throw std::runtime_error("Each row must have exactly 3 elements to form a Vector3d");
        }
        result.emplace_back(row[0], row[1], row[2]);
    }

    return result;
}

std::string DataManager::getSerowPath() const {
    std::string serow_path_env = std::getenv("SEROW_PATH");

    if (serow_path_env.empty()) {
        throw std::runtime_error("SEROW_PATH environment variable not set");
    }

    if (serow_path_env.back() != '/') {
        serow_path_env.push_back('/');
        return serow_path_env;
    }
    return serow_path_env;
}

std::vector<double> DataManager::readHDF5_1D(const std::string& filename,
                                             const std::string& datasetName) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();

    int ndims = dataspace.getSimpleExtentNdims();
    hsize_t dims[2] = {0, 0};  // Safe for both 1D and 2D
    dataspace.getSimpleExtentDims(dims);

    timestamps_.clear();

    if (ndims == 1) {
        // 1D vector
        timestamps_.resize(dims[0]);
        dataset.read(timestamps_.data(), H5::PredType::NATIVE_DOUBLE);
    } else if (ndims == 2 && dims[1] == 1) {
        // 2D column vector
        std::vector<double> buffer(dims[0]);  // dims[1] == 1, so flat
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
        timestamps_ = std::move(buffer);
    } else {
        throw std::runtime_error("Unsupported timestamp shape for " + datasetName);
    }
    return timestamps_;
}

std::vector<std::vector<double>> DataManager::readHDF5(const std::string& filename,
                                                       const std::string& datasetName) {
    H5::H5File file(filename, H5F_ACC_RDONLY);

    if (datasetName.empty()) {
        throw std::invalid_argument("Dataset name must be non-empty and start with '/'");
    }

    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[2];
    int ndims = dataspace.getSimpleExtentDims(dims);
    if (ndims != 2 || dims[0] == 0 || dims[1] == 0) {
        throw std::runtime_error("Unexpected dataset dimensions for " + datasetName);
    }

    std::vector<double> buffer(dims[0] * dims[1]);
    dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

    std::vector<std::vector<double>> data(dims[0], std::vector<double>(dims[1]));
    for (hsize_t i = 0; i < dims[0]; ++i) {
        for (hsize_t j = 0; j < dims[1]; ++j) {
            data[i][j] = buffer[i * dims[1] + j];
        }
    }
    return data;
}

std::vector<std::vector<double>> DataManager::getBasePositionData() const {
    return base_position_;
}

std::vector<std::vector<double>> DataManager::getBaseOrientationData() const {
    return base_orientation_;
}

std::vector<Eigen::Vector3d> DataManager::getImuLinearAccelerationData() const {
    return linear_acceleration_;
}

std::vector<Eigen::Vector3d> DataManager::getImuAngularVelocityData() const {
    return angular_velocity_;
}

std::vector<std::vector<double>> DataManager::getJointStatesData() const {
    return joint_states_;
}

std::vector<std::vector<double>> DataManager::getJointVelocitiesData() const {
    return joint_velocities_;
}

DataManager::ForceTorqueData DataManager::getForceTorqueData() const {
    return force_torque_;
}

std::string DataManager::getRobotName() const {
    return robot_name_;
}

std::vector<double> DataManager::getTimestamps() const {
    return timestamps_;
}

std::vector<std::string> DataManager::getFootFrames() const {
    return foot_frames_;
}

std::vector<std::string> DataManager::getJointNames() const {
    return joint_names_;
}
