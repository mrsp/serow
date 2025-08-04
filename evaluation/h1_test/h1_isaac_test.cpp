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
    std::string serow_path_env = std::getenv("SEROW_PATH");
    std::string resolved_path = serow_path_env + path;
    std::string experiment_type = config["Experiment"]["type"];
    std::string base_path = config["Paths"]["base_path"];

    // Replace placeholders
    resolved_path = std::regex_replace(resolved_path, std::regex("\\{base_path\\}"), base_path);
    resolved_path = std::regex_replace(resolved_path, std::regex("\\{type\\}"), experiment_type);

    return resolved_path;
}

// Saves predictions to .h5 file
void saveDataToHDF5(const std::string& fileName, const std::string& datasetPath,
                    const std::vector<double>& data) {
    if (datasetPath.empty() || datasetPath[0] != '/') {
        throw std::invalid_argument("Dataset path must be non-empty and start with '/'");
    }
    std::cout << fileName << '\n';

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

int main(int argc, char** argv) {
    try {
        std::string config_path = "h1.json";  // Default path

        // If user provides a custom config path
        if (argc > 1) {
            config_path = argv[1];
            std::cout << "Using config from argument: " << config_path << std::endl;
        } else {
            std::cout << "No config argument provided. Using default: " << config_path << std::endl;
        }

        // Initialize Serow with the specified config
        serow::Serow estimator;
        if (!estimator.initialize(config_path)) {
            throw std::runtime_error("Failed to initialize Serow with config: " + config_path);
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
        std::cout << "Using elevation map file: " << ELEVATION_MAP_FILE << std::endl;
        // Read IMU data
        auto angular_velocity = readHDF5(INPUT_FILE, "h1_imu/_angular_velocity");
        auto linear_acceleration = readHDF5(INPUT_FILE, "h1_imu/_linear_acceleration");
        std::cout << "Opened IMU data from " << INPUT_FILE << std::endl;
        // Read Joint States
        auto joint_positions = readHDF5(INPUT_FILE, "h1_joint_states/_position");
        auto joint_velocities = readHDF5(INPUT_FILE, "h1_joint_states/_velocity");
        std::cout << "Opened Joint States from " << INPUT_FILE << std::endl;
        // Read Feet Forces
        auto feet_force_R = readHDF5(INPUT_FILE, "h1_left_ankle_force_torque_states/_wrench/_force");
        auto feet_force_L = readHDF5(INPUT_FILE, "h1_right_ankle_force_torque_states/_wrench/_force");
        std::cout << "Opened Feet Forces from " << INPUT_FILE << std::endl;

        // Read Base Pose Ground Truth
        auto base_gt_positions = readHDF5(INPUT_FILE, "h1_ground_truth_odometry/_pose/_pose/_position");
        auto base_gt_orientations = readHDF5(INPUT_FILE, "h1_ground_truth_odometry/_pose/_pose/_orientation");
        std::cout << "Opened Base Pose Ground Truth from " << INPUT_FILE << std::endl;
        // Read Timestamps
        auto imu_timestamps = readHDF5(INPUT_FILE, "h1_imu/timestamp");
        auto joint_timestamps = readHDF5(INPUT_FILE, "h1_joint_states/timestamp");
        auto feet_force_timestamps = readHDF5(INPUT_FILE, "h1_left_ankle_force_torque_states/timestamp");
        auto base_gt_timestamp = readHDF5(INPUT_FILE, "h1_ground_truth_odometry/timestamp");
        std::cout << "Opened Timestamps from " << INPUT_FILE << std::endl;

        
        std::cout << "imu_timestamps size: " << imu_timestamps.size() << "  " << imu_timestamps[0][0]<< std::endl;
        // Initial timestamps in seconds
        auto init_timestamp_imu = imu_timestamps[0][0] * 1e-9;
        auto init_timestamp_joint = joint_timestamps[0][0] * 1e-9;
        auto init_timestamp_feet_force = feet_force_timestamps[0][0] * 1e-9;
        auto init_timestamp_base_gt = base_gt_timestamp[0][0] * 1e-9;

        // Store predictions
        std::vector<double> EstTimestamp;  // timestamp
        std::vector<double> base_pos_x, base_pos_y, base_pos_z, base_rot_x, base_rot_y, base_rot_z,
            base_rot_w;                                          // Base pose(pos + quat)
        std::vector<double> com_x, com_y, com_z;                 // CoM position
        std::vector<double> com_vel_x, com_vel_y, com_vel_z;     // CoM velocity
        std::vector<double> extFx, extFy, extFz;                 // External Force
        std::vector<double> b_ax, b_ay, b_az, b_wx, b_wy, b_wz;  // IMU Biases
        std::vector<Eigen::Vector3d> R_contact_position, L_contact_position;
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < joint_timestamps.size(); ++i) {

            imu_timestamps[i][0] = (imu_timestamps[i][0] * 1e-9) - init_timestamp_imu;
            joint_timestamps[i][0] = (joint_timestamps[i][0] * 1e-9) - init_timestamp_joint;
            feet_force_timestamps[i][0] = (feet_force_timestamps[i][0] * 1e-9) - init_timestamp_feet_force;
            base_gt_timestamp[i][0] = (base_gt_timestamp[i][0] * 1e-9) - init_timestamp_base_gt;
            
            std::map<std::string, serow::ForceTorqueMeasurement> force_torque;
            force_torque.insert(
                {"right_ankle_link",
                 serow::ForceTorqueMeasurement{
                     .timestamp = feet_force_timestamps[i][0],
                     .force = Eigen::Vector3d(feet_force_R[i][0], feet_force_R[i][1],
                                              feet_force_R[i][2])}});
            force_torque.insert(
                {"left_ankle_link",
                 serow::ForceTorqueMeasurement{
                     .timestamp = feet_force_timestamps[i][0],
                     .force = Eigen::Vector3d(feet_force_L[i][0], feet_force_L[i][1],
                                              feet_force_L[i][2])}});

            serow::ImuMeasurement imu;

            imu.timestamp = imu_timestamps[i][0];
            imu.linear_acceleration = Eigen::Vector3d(
                linear_acceleration[i][0], linear_acceleration[i][1], linear_acceleration[i][2]);
            imu.angular_velocity = Eigen::Vector3d(angular_velocity[i][0], angular_velocity[i][1],
                                                   angular_velocity[i][2]);

            std::map<std::string, serow::JointMeasurement> joints;
            joints.insert({"left_hip_yaw_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][0],
                                                   .velocity = joint_velocities[i][0]}});
            joints.insert({"right_hip_yaw_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][1],
                                                   .velocity = joint_velocities[i][1]}});
            joints.insert({"torso_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][2],
                                                   .velocity = joint_velocities[i][2]}});
            joints.insert({"left_hip_roll_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][3],
                                                   .velocity = joint_velocities[i][3]}});
            joints.insert({"right_hip_roll_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][4],
                                                   .velocity = joint_velocities[i][4]}});
            joints.insert({"left_shoulder_pitch_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][5],
                                                   .velocity = joint_velocities[i][5]}});
            joints.insert({"right_shoulder_pitch_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][6],
                                                   .velocity = joint_velocities[i][6]}});
            joints.insert({"left_hip_pitch_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][7],
                                                   .velocity = joint_velocities[i][7]}});
            joints.insert({"right_hip_pitch_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][8],
                                                   .velocity = joint_velocities[i][8]}});
            joints.insert({"left_shoulder_roll_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][9],
                                                   .velocity = joint_velocities[i][9]}});
            joints.insert({"right_shoulder_roll_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][10],
                                                   .velocity = joint_velocities[i][10]}});
            joints.insert({"left_knee_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][11],
                                                   .velocity = joint_velocities[i][11]}});
            joints.insert({"right_knee_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][12],
                                                   .velocity = joint_velocities[i][12]}});
            joints.insert({"left_shoulder_yaw_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][13],
                                                   .velocity = joint_velocities[i][13]}});
            joints.insert({"right_shoulder_yaw_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][14],
                                                   .velocity = joint_velocities[i][14]}});
            joints.insert({"left_ankle_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][15],
                                                   .velocity = joint_velocities[i][15]}});
            joints.insert({"right_ankle_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][16],
                                                   .velocity = joint_velocities[i][16]}});
            joints.insert({"left_elbow_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][17],
                                                   .velocity = joint_velocities[i][17]}});
            joints.insert({"right_elbow_joint",
                           serow::JointMeasurement{.timestamp = joint_timestamps[i][0],
                                                   .position = joint_positions[i][18],
                                                   .velocity = joint_velocities[i][18]}});

            // If the ground truth for the base pose is available, pass it to the filter for
            // synchronized logging
            if (base_gt_positions.size() > 0 && base_gt_orientations.size() > 0) {
                estimator.filter(imu, joints, force_torque, std::nullopt, std::nullopt,
                                 BasePoseGroundTruth{
                                     .timestamp = base_gt_timestamp[i][0],
                                     .position = Eigen::Vector3d(base_gt_positions[i][0],
                                                                 base_gt_positions[i][1],
                                                                 base_gt_positions[i][2]),
                                     .orientation = Eigen::Quaterniond(
                                         base_gt_orientations[i][0], base_gt_orientations[i][1],
                                         base_gt_orientations[i][2], base_gt_orientations[i][3])});
            } else {
                estimator.filter(imu, joints, force_torque);
            }

            auto state = estimator.getState();
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
            auto R_contact_pos = state->getContactPosition("right_ankle_link");
            auto L_contact_pos = state->getContactPosition("left_ankle_link");
            
            if (L_contact_pos.has_value()) {
                L_contact_position.push_back(L_contact_pos.value());
            } else {
                L_contact_position.push_back(Eigen::Vector3d::Zero());
            }

            if (R_contact_pos.has_value()) {
                R_contact_position.push_back(R_contact_pos.value());
            } else {
                R_contact_position.push_back(Eigen::Vector3d::Zero());
            }


            EstTimestamp.push_back(joint_timestamps[i][0]);  // Convert to seconds
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
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Time taken by for loop: " << duration.count() << " microseconds" << std::endl;

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
        std::vector<double> L_contact_position_x, L_contact_position_y, L_contact_position_z;
        std::vector<double> R_contact_position_x, R_contact_position_y, R_contact_position_z;

        for (const auto& vec : L_contact_position) {
            L_contact_position_x.push_back(vec.x());
            L_contact_position_y.push_back(vec.y());
            L_contact_position_z.push_back(vec.z());
        }

        for (const auto& vec : R_contact_position) {
            R_contact_position_x.push_back(vec.x());
            R_contact_position_y.push_back(vec.y());
            R_contact_position_z.push_back(vec.z());
        }

        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/L_foot/x", L_contact_position_x);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/L_foot/y", L_contact_position_y);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/L_foot/z", L_contact_position_z);

        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/R_foot/x", R_contact_position_x);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/R_foot/y", R_contact_position_y);
        saveDataToHDF5(OUTPUT_FILE, "/contact_positions/R_foot/z", R_contact_position_z);

        std::cout << "Processing complete. Predictions saved to " << OUTPUT_FILE << std::endl;
    

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
