#include "DataManager.hpp"

DataManager::DataManager() 
{
    // Constructor implementation
    std::cout << "Data Manager initialized." << std::endl;

    loadConfig();

    loadData();
}

void DataManager::loadData() {
  std::cout << "Loading data from "<< DATA_FILE_ << std::endl;
  // Load data from HDF5 files
  robotPos_ = readHDF5(DATA_FILE_, "base_ground_truth/position");
  robotRot_ = readHDF5(DATA_FILE_, "base_ground_truth/orientation");

  linAcc_   = readHDF5(DATA_FILE_, "imu/linear_acceleration");
  angVel_   = readHDF5(DATA_FILE_, "imu/angular_velocity");
  jointStates_ = readHDF5(DATA_FILE_, "joint_states/positions");
  jointVelocities_ = readHDF5(DATA_FILE_, "joint_states/velocities");
  // Load force data
  forceData_.FR = readHDF5(DATA_FILE_, "feet_force/FR");
  forceData_.FL = readHDF5(DATA_FILE_, "feet_force/FL");
  forceData_.RL = readHDF5(DATA_FILE_, "feet_force/RL");
  forceData_.RR = readHDF5(DATA_FILE_, "feet_force/RR");

  timestamps_ = readHDF5(DATA_FILE_, "timestamps");
}

void DataManager::loadConfig() {
    std::cout << "Loading configuration file: experimentConfig.json..." << std::endl;
    std::string filePath = getSerowPath() + "/evaluation/serowHypertuner/config/experimentConfig.json";
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open: " + filePath);
    }

    file >> config_;

    DATA_FILE_ = resolvePath(config_, config_["Paths"]["data_file"]);
}

std::string DataManager::resolvePath(const json& config, const std::string& path) {
    std::string serowPathEnv = std::getenv("SEROW_PATH");
    std::string resolvedPath = serowPathEnv + path;
    // Extract all placeholders from the config
    basePath = config["Paths"]["base_path"];
    experimentType = config["Experiment"]["experiment_type"];
    robotName = config["Experiment"]["robot_name"];
    experimentName = config["Experiment"]["experiment_name"];

    // Replace placeholders
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{base_path\\}"), basePath);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{experiment_type\\}"), experimentType);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{type\\}"), experimentType);  // optional
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{robot_name\\}"), robotName);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{experiment_name\\}"), experimentName);

    return resolvedPath;
}


std::string DataManager::getSerowPath() const {
  const char* serowPath = std::getenv("SEROW_PATH");
  if (serowPath == nullptr) {
      throw std::runtime_error("SEROW_PATH environment variable not set");
  }
  return std::string(serowPath);
}

std::vector<double> DataManager::readHDF5_1D(const std::string& filename, const std::string& datasetName) {
  H5::H5File file(filename, H5F_ACC_RDONLY);

  if (datasetName.empty()) {
      throw std::invalid_argument("Dataset name must be non-empty.");
  }

  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  // Get number of dimensions and check
  int ndims = dataspace.getSimpleExtentNdims();
  if (ndims != 1) {
      throw std::runtime_error("Dataset " + datasetName + " is not 1D.");
  }

  hsize_t dim;
  dataspace.getSimpleExtentDims(&dim);

  std::vector<double> data(dim);
  dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

  return data;
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

std::vector<std::vector<double>> DataManager::getPosData() const {
  return robotPos_;
}

std::vector<std::vector<double>> DataManager::getRotData() const {
  return robotRot_;
}

std::vector<std::vector<double>> DataManager::getLinAccData() const {
  return linAcc_;
}

std::vector<std::vector<double>> DataManager::getAngVelData() const {
  return angVel_;
}

std::vector<std::vector<double>> DataManager::getJointStatesData() const {
  return jointStates_;
}

std::vector<std::vector<double>> DataManager::getJointVelocitiesData() const {
  return jointVelocities_;
}

DataManager::ForceData DataManager::getForceData() const {
  return forceData_;
}

std::string DataManager::getRobotName() const {
    return robotName;
}

std::vector<std::vector<double>> DataManager::getTimestamps() const{
  return timestamps_;
}