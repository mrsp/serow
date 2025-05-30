#include "DataManager.hpp"

DataManager::DataManager() 
{
    loadConfig();

    loadData();

    std::cout << "Data Manager initialized." << std::endl;
}

void DataManager::loadData() {
  std::cout << "Loading data from "<< data_file_ << std::endl;
  
  // Load data from HDF5 files
  base_position_ = readHDF5(data_file_, "base_ground_truth/position");
  base_orientation_ = readHDF5(data_file_, "base_ground_truth/orientation");

  linear_acceleration_   = convertToEigenVec3(readHDF5(data_file_, "imu/linear_acceleration"));
  angular_velocity_   = convertToEigenVec3(readHDF5(data_file_, "imu/angular_velocity"));
  joint_states_ = readHDF5(data_file_, "joint_states/positions");
  joint_velocities_ = readHDF5(data_file_, "joint_states/velocities");
  
  // Load force data
  force_torque_.force.FR = convertToEigenVec3(readHDF5(data_file_, "feet_force/FR"));
  force_torque_.force.FL = convertToEigenVec3(readHDF5(data_file_, "feet_force/FL"));
  force_torque_.force.RL = convertToEigenVec3(readHDF5(data_file_, "feet_force/RL"));
  force_torque_.force.RR = convertToEigenVec3(readHDF5(data_file_, "feet_force/RR"));
  

  timestamps_ = readHDF5_1D(data_file_, "timestamps");
}

void DataManager::loadConfig() {
    std::cout << "Loading configuration file: experimentConfig.json..." << std::endl;
    std::string filePath = getSerowPath() + "/serow_hypertuner/config/experimentConfig.json";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + filePath);
    }

    file >> config_;

    data_file_ = resolvePath(config_, config_["Paths"]["data_file"]);
}

std::string DataManager::resolvePath(const json& config, const std::string& path) {
    std::string serowPathEnv = getSerowPath();
    std::string resolvedPath = serowPathEnv + path;

    // Extract all placeholders from the config
    base_path_ = config["Paths"]["base_path"];
    experiment_type_ = config["Experiment"]["experiment_type"];
    robot_name_ = config["Experiment"]["robot_name"];
    experiment_name_ = config["Experiment"]["experiment_name"];

    // Replace placeholders
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{base_path\\}"), base_path_);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{experiment_type\\}"), experiment_type_);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{type\\}"), experiment_type_);  // optional
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{robot_name\\}"), robot_name_);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{experiment_name\\}"), experiment_name_);

    return resolvedPath;
}

std::vector<Eigen::Vector3d> DataManager::convertToEigenVec3(const std::vector<std::vector<double>>& data) {
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
  std::string serowPath = std::getenv("SEROW_PATH");

  if (serowPath.empty()) {
      throw std::runtime_error("SEROW_PATH environment variable not set");
  }

  if (serowPath.back() != '/') {
      serowPath.push_back('/');
      return serowPath;
  }
  return serowPath;
}

std::vector<double> DataManager::readHDF5_1D(const std::string& filename, const std::string& datasetName) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet(datasetName);
    H5::DataSpace dataspace = dataset.getSpace();

    int ndims = dataspace.getSimpleExtentNdims();
    hsize_t dims[2] = {0, 0}; // Safe for both 1D and 2D
    dataspace.getSimpleExtentDims(dims);

    timestamps_.clear();

    if (ndims == 1) {
        // 1D vector
        timestamps_.resize(dims[0]);
        dataset.read(timestamps_.data(), H5::PredType::NATIVE_DOUBLE);
    } else if (ndims == 2 && dims[1] == 1) {
        // 2D column vector
        std::vector<double> buffer(dims[0]); // dims[1] == 1, so flat
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

std::string DataManager::getrobot_name_() const {
    return robot_name_;
}

std::vector<double> DataManager::getTimestamps() const{
  return timestamps_;
}
