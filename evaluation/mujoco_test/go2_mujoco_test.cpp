#include <H5Cpp.h>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <regex>
#include <vector>
#include "serow/Serow.hpp"

using namespace serow;
using json = nlohmann::json;

std::string resolvePath(const json& config, const std::string& path) {
    std::string serowPathEnv = std::getenv("SEROW_PATH");
    std::string resolvedPath = serowPathEnv + path;
    std::string experimentType = config["Experiment"]["type"];
    std::string basePath = config["Paths"]["base_path"];

    // Replace placeholders
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{base_path\\}"), basePath);
    resolvedPath = std::regex_replace(resolvedPath, std::regex("\\{type\\}"), experimentType);

    return resolvedPath;
}

// Saves predictions to .h5 file
void saveDataToHDF5(const std::string& fileName, const std::string& datasetPath,
                    const std::vector<double>& data) {
    if (datasetPath.empty() || datasetPath[0] != '/') {
        throw std::invalid_argument("Dataset path must be non-empty and start with '/'");
    }

    // Open or create the file
    H5::H5File file;
    try {
        file.openFile(fileName, H5F_ACC_RDWR);
    } catch (const H5::FileIException&) {
        file = H5::H5File(fileName, H5F_ACC_TRUNC);
    }

    const size_t lastSlash = datasetPath.find_last_of('/');
    std::string groupPath = datasetPath.substr(0, lastSlash);
    std::string datasetName = datasetPath.substr(lastSlash + 1);

    if (datasetName.empty()) {
        throw std::invalid_argument("Dataset name cannot be empty.");
    }
    H5::Exception::dontPrint();
    // Ensure groups along the path exist
    H5::Group group = file.openGroup("/");
    if (!groupPath.empty()) {
        std::stringstream ss(groupPath);
        std::string token;
        while (std::getline(ss, token, '/')) {
            if (!token.empty()) {
                try {
                    group = group.openGroup(token);
                } catch (const H5::Exception&) {
                    group = group.createGroup(token);
                }
            }
        }
    }

    // Create or open the dataset
    try {
        H5::DataSet dataset = group.openDataSet(datasetName);
    } catch (const H5::Exception&) {
        // Dataset does not exist; create it
        hsize_t dims[1] = {data.size()};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset =
            group.createDataSet(datasetName, H5::PredType::NATIVE_DOUBLE, dataspace);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    }
}

// Reads dataset from HDF5 file (.h5)
std::vector<std::vector<double>> readHDF5(const std::string& filename,
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

/// @brief Writes an elevation map measurement to a binary file
void saveElevationMap(std::array<ElevationCell, map_size> data, double timestamp,
                      std::ofstream& file) {
    if (!file.is_open()) {
        std::cerr << "[saveElevationMap] File stream is not open!\n";
        return;
    }

    // Optional: Write a timestamp or measurement ID
    file.write(reinterpret_cast<const char*>(&timestamp), sizeof(timestamp));

    // Write the height data
    for (size_t i = 0; i < 1024; ++i) {
        for (size_t j = 0; j < 1024; ++j) {
            // std::cout << data[i][j].height << '\n';
            file.write(reinterpret_cast<const char*>(&data[i + 1024 * j].height), sizeof(float));
        }
    }
    // Flush to ensure data is written
    file.flush();
}

int main() {
    try {
        // Initialize Serow
        serow::Serow SEROW;
        if (!SEROW.initialize("go2.json")) {
            throw std::runtime_error("Failed to initialize Serow");
        }

        std::ifstream config_file("../test_config.json");
        if (!config_file.is_open()) {
            std::cerr << "Failed to open test_config.json" << std::endl;
            return 1;
        }
        json config;
        config_file >> config;
        const std::string INPUT_FILE = resolvePath(config, config["Paths"]["data_file"]);
        const std::string OUTPUT_FILE = resolvePath(config, config["Paths"]["prediction_file"]);
        const std::string ELEVATION_MAP_FILE =
            resolvePath(config, config["Paths"]["elevation_map_file"]);

        std::ofstream file(ELEVATION_MAP_FILE);

        // Read IMU data
        auto angular_velocity = readHDF5(INPUT_FILE, "imu/angular_velocity");
        auto linear_acceleration = readHDF5(INPUT_FILE, "imu/linear_acceleration");

        // Read Joint States
        auto joint_positions = readHDF5(INPUT_FILE, "joint_states/positions");

        // Read Feet Forces
        auto feet_force_FR = readHDF5(INPUT_FILE, "feet_force/FR");
        auto feet_force_FL = readHDF5(INPUT_FILE, "feet_force/FL");
        auto feet_force_RL = readHDF5(INPUT_FILE, "feet_force/RL");
        auto feet_force_RR = readHDF5(INPUT_FILE, "feet_force/RR");

        // Read Timestamps
        auto timestamps = readHDF5(INPUT_FILE, "timestamps");

        // Store predictions
        std::vector<double> EstTimestamp;  // timestamp
        std::vector<double> base_pos_x, base_pos_y, base_pos_z, base_rot_x, base_rot_y, base_rot_z,
            base_rot_w;  // Base pose(pos + quat)
        std::vector<double> com_x, com_y, com_z;  // CoM position
        std::vector<double> com_vel_x, com_vel_y, com_vel_z;  // CoM velocity
        std::vector<double> extFx, extFy, extFz;  // External Force
        std::vector<double> b_ax, b_ay, b_az, b_wx, b_wy, b_wz;  // IMU Biases
        std::vector<Eigen::Vector3d> FR_contact_position, FL_contact_position, RL_contact_position,
            RR_contact_position;
        double log_timestamp = timestamps[0][0];

        for (size_t i = 0; i < timestamps.size(); ++i) {
            double timestamp = timestamps[i][0];

            std::map<std::string, serow::ForceTorqueMeasurement> force_torque;
            force_torque.insert(
                {"FR_foot",
                 serow::ForceTorqueMeasurement{
                     .timestamp = timestamp,
                     .force = Eigen::Vector3d(feet_force_FR[i][0], feet_force_FR[i][1],
                                              feet_force_FR[i][2])}});
            force_torque.insert(
                {"FL_foot",
                 serow::ForceTorqueMeasurement{
                     .timestamp = timestamp,
                     .force = Eigen::Vector3d(feet_force_FL[i][0], feet_force_FL[i][1],
                                              feet_force_FL[i][2])}});
            force_torque.insert(
                {"RL_foot",
                 serow::ForceTorqueMeasurement{
                     .timestamp = timestamp,
                     .force = Eigen::Vector3d(feet_force_RL[i][0], feet_force_RL[i][1],
                                              feet_force_RL[i][2])}});
            force_torque.insert(
                {"RR_foot",
                 serow::ForceTorqueMeasurement{
                     .timestamp = timestamp,
                     .force = Eigen::Vector3d(feet_force_RR[i][0], feet_force_RR[i][1],
                                              feet_force_RR[i][2])}});

            serow::ImuMeasurement imu;
            imu.timestamp = timestamp;
            imu.linear_acceleration = Eigen::Vector3d(
                linear_acceleration[i][0], linear_acceleration[i][1], linear_acceleration[i][2]);
            imu.angular_velocity = Eigen::Vector3d(angular_velocity[i][0], angular_velocity[i][1],
                                                   angular_velocity[i][2]);

            std::map<std::string, serow::JointMeasurement> joints;
            joints.insert({"FL_hip_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][0]}});
            joints.insert({"FL_thigh_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][1]}});
            joints.insert({"FL_calf_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][2]}});
            joints.insert({"FR_hip_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][3]}});
            joints.insert({"FR_thigh_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][4]}});
            joints.insert({"FR_calf_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][5]}});
            joints.insert({"RL_hip_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][6]}});
            joints.insert({"RL_thigh_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][7]}});
            joints.insert({"RL_calf_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][8]}});
            joints.insert({"RR_hip_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][9]}});
            joints.insert({"RR_thigh_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][10]}});
            joints.insert({"RR_calf_joint",
                           serow::JointMeasurement{.timestamp = timestamp,
                                                   .position = joint_positions[i][11]}});

            SEROW.filter(imu, joints, force_torque);

            auto state = SEROW.getState();
            if (!state.has_value()) {
                continue;
            }

            // Store the Estimates
            auto basePos = state->getBasePosition();
            auto baseOrient = state->getBaseOrientation();
            auto ExternalForces = state->getCoMExternalForces();
            auto CoMPosition = state->getCoMPosition();
            auto CoMVelocity = state->getCoMLinearVelocity();
            auto linAccelBias = state->getImuLinearAccelerationBias();
            auto angVelBias = state->getImuAngularVelocityBias();
            // Get Contact positions
            auto FR_contact_pos = state->getContactPosition("FR_foot");
            auto FL_contact_pos = state->getContactPosition("FL_foot");
            auto RL_contact_pos = state->getContactPosition("RL_foot");
            auto RR_contact_pos = state->getContactPosition("RR_foot");

            if (FL_contact_pos.has_value()) {
                FL_contact_position.push_back(FL_contact_pos.value());
            } else {
                FL_contact_position.push_back(Eigen::Vector3d::Zero());
            }

            if (FR_contact_pos.has_value()) {
                FR_contact_position.push_back(FR_contact_pos.value());
            } else {
                FR_contact_position.push_back(Eigen::Vector3d::Zero());
            }

            if (RL_contact_pos.has_value()) {
                RL_contact_position.push_back(RL_contact_pos.value());
            } else {
                RL_contact_position.push_back(Eigen::Vector3d::Zero());
            }

            if (RR_contact_pos.has_value()) {
                RR_contact_position.push_back(RR_contact_pos.value());
            } else {
                RR_contact_position.push_back(Eigen::Vector3d::Zero());
            }

            auto terrainEstimator = SEROW.getTerrainEstimator();
            if (!terrainEstimator) {
                continue;
            }

            // if (timestamp - log_timestamp > 0.5) {
            //     saveElevationMap(terrainEstimator->elevation_, timestamp, file);
            //     log_timestamp = timestamp;
            // }

            EstTimestamp.push_back(timestamp);
            base_pos_x.push_back(basePos.x());
            base_pos_y.push_back(basePos.y());
            base_pos_z.push_back(basePos.z());
            base_rot_x.push_back(baseOrient.x());
            base_rot_y.push_back(baseOrient.y());
            base_rot_z.push_back(baseOrient.z());
            base_rot_w.push_back(baseOrient.w());
            com_x.push_back(CoMPosition.x());
            com_y.push_back(CoMPosition.y());
            com_z.push_back(CoMPosition.z());
            com_vel_x.push_back(CoMVelocity.x());
            com_vel_y.push_back(CoMVelocity.y());
            com_vel_z.push_back(CoMVelocity.z());
            extFx.push_back(ExternalForces.x());
            extFy.push_back(ExternalForces.y());
            extFz.push_back(ExternalForces.z());
            b_ax.push_back(linAccelBias.x());
            b_ay.push_back(linAccelBias.y());
            b_az.push_back(linAccelBias.z());
            b_wx.push_back(angVelBias.x());
            b_wy.push_back(angVelBias.y());
            b_wz.push_back(angVelBias.z());
        }
        // Write structured data to HDF5
        H5::H5File outputFile(OUTPUT_FILE, H5F_ACC_TRUNC);  // Create the output file

        saveDataToHDF5(OUTPUT_FILE, "/timestamp/t", EstTimestamp);
        // Base State
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/position/x", base_pos_x);
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/position/y", base_pos_y);
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/position/z", base_pos_z);
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/rotation/x", base_rot_x);
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/rotation/y", base_rot_y);
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/rotation/z", base_rot_z);
        saveDataToHDF5(OUTPUT_FILE, "/base_pose/rotation/w", base_rot_w);

        // CoM state
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/position/x", com_x);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/position/y", com_y);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/position/z", com_z);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/velocity/x", com_vel_x);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/velocity/y", com_vel_y);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/velocity/z", com_vel_z);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/externalForces/x", extFx);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/externalForces/y", extFy);
        saveDataToHDF5(OUTPUT_FILE, "/CoM_state/externalForces/z", extFz);

        // IMU biases
        saveDataToHDF5(OUTPUT_FILE, "/imu_bias/accel/x", b_ax);
        saveDataToHDF5(OUTPUT_FILE, "/imu_bias/accel/y", b_ay);
        saveDataToHDF5(OUTPUT_FILE, "/imu_bias/accel/z", b_az);
        saveDataToHDF5(OUTPUT_FILE, "/imu_bias/angVel/x", b_wx);
        saveDataToHDF5(OUTPUT_FILE, "/imu_bias/angVel/y", b_wy);
        saveDataToHDF5(OUTPUT_FILE, "/imu_bias/angVel/z", b_wz);

        // Contact Positions
        std::vector<double> FL_contact_position_x, FL_contact_position_y, FL_contact_position_z;
        std::vector<double> FR_contact_position_x, FR_contact_position_y, FR_contact_position_z;
        std::vector<double> RL_contact_position_x, RL_contact_position_y, RL_contact_position_z;
        std::vector<double> RR_contact_position_x, RR_contact_position_y, RR_contact_position_z;

        for (const auto& vec : FL_contact_position) {
            FL_contact_position_x.push_back(vec.x());
            FL_contact_position_y.push_back(vec.y());
            FL_contact_position_z.push_back(vec.z());
        }

        for (const auto& vec : FR_contact_position) {
            FR_contact_position_x.push_back(vec.x());
            FR_contact_position_y.push_back(vec.y());
            FR_contact_position_z.push_back(vec.z());
        }

        for (const auto& vec : RL_contact_position) {
            RL_contact_position_x.push_back(vec.x());
            RL_contact_position_y.push_back(vec.y());
            RL_contact_position_z.push_back(vec.z());
        }

        for (const auto& vec : RR_contact_position) {
            RR_contact_position_x.push_back(vec.x());
            RR_contact_position_y.push_back(vec.y());
            RR_contact_position_z.push_back(vec.z());
        }
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/FL_foot/x", FL_contact_position_x);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/FL_foot/y", FL_contact_position_y);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/FL_foot/z", FL_contact_position_z);

        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/FR_foot/x", FR_contact_position_x);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/FR_foot/y", FR_contact_position_y);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/FR_foot/z", FR_contact_position_z);

        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/RL_foot/x", RL_contact_position_x);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/RL_foot/y", RL_contact_position_y);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/RL_foot/z", RL_contact_position_z);

        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/RR_foot/x", RR_contact_position_x);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/RR_foot/y", RR_contact_position_y);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/RR_foot/z", RR_contact_position_z);

        std::cout << "Processing complete. Predictions saved to " << OUTPUT_FILE << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
