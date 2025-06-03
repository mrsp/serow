#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <H5Cpp.h>
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

using json = nlohmann::ordered_json;

/// @brief DataManager class to manage data loading and configuration
class DataManager {
public:
    ///@brief Force data object definition 3D (x y z) for each leg
    struct ForceTorqueData {
        struct Force{
            std::map<std::string, std::vector<Eigen::Vector3d>> data;
        };

        struct Torque {
            std::map<std::string, std::vector<Eigen::Vector3d>> data;
        };

        Force force;    
        Torque torque;
    };

    /// @brief Constructor
    DataManager();
    
    /// @brief Gets the ground truth robot position data
    /// @return 2D vector of doubles with the robot position data
    std::vector<std::vector<double>> getBasePositionData() const; 
    
    /// @brief Get the robot orientation data
    /// @return 2D vector of doubles with the robot orientation data
    std::vector<std::vector<double>> getBaseOrientationData() const;
    
    /// @brief Get the linear acceleration data
    /// @return 2D vector of doubles with the linear acceleration data
    std::vector<Eigen::Vector3d> getImuLinearAccelerationData() const;
    
    /// @brief Get the angular velocity data
    /// @return 2D vector of doubles with the angular velocity data
    std::vector<Eigen::Vector3d> getImuAngularVelocityData() const;
    
    /// @brief Get the joint states data
    /// @return 2D vector of doubles with the joint states data
    std::vector<std::vector<double>> getJointStatesData() const;

    /// @brief Get the joint velocities data
    /// @return 2D vector of doubles with the joint velocities data
    std::vector<std::vector<double>> getJointVelocitiesData() const;

    /// @brief Get the force/torque data
    /// @return ForceTorqueData struct containing the force and torque data
    ForceTorqueData getForceTorqueData() const;

    /// @brief Finds the path of SEROW from the environment variable SEROW_PATH
    /// @return the path of SEROW
    std::string getSerowPath() const;
    
    /// @brief Get the robot name from the configuration
    /// @return the robot name
    std::string getRobotName() const;

    /// @brief Get the timestamps of the logged data
    /// @return A vector of doubles containing the timestamps
    std::vector<double> getTimestamps() const;

    /// @brief Get the foot frames from the configuration file
    /// @return A vector of strings containing the foot frame names
    std::vector<std::string> getFootFrames() const;

    /// @brief Get the joint names from the configuration file
    /// @return A vector of strings containing the joint names
    std::vector<std::string> getJointNames() const;

private: 
    /// @brief Loads the configuration file (experimentConfig.json) from the config directory
    void loadConfig();

    /// @brief Loads the data from the HDF5 file as defined in the experimentConfig.json file
    void loadData();
    
    /// @brief Initializes the force/torque measurements for each foot frame
    void initializeFTMeasurements();

    /// @brief test configuration file
    nlohmann::json config_;
    /// @brief path to the data file
    std::string  data_file_;

    /// @brief  Reads a 2D dataset from an HDF5 file and returns it as a 2D vector of doubles.
    /// @param filename The path to the HDF5 file to read from.
    /// @param datasetName The dataset inside the HDF5 file (e.g., "imu/linear_acceleration").
    /// @return A 2D vector containing the dataset values.
    std::vector<std::vector<double>> readHDF5(const std::string& filename,
                                              const std::string& datasetName);

    /// @brief Reads a 1D dataset from an HDF5 file and returns it as a vector of doubles.
    /// @param filename The path to the HDF5 file to read from.
    /// @param datasetName The dataset inside the HDF5 file (e.g., "imu/linear_acceleration").
    /// @return A vector containing the dataset values.
    std::vector<double> readHDF5_1D(const std::string& filename, const std::string& datasetName);

    ///@brief Ground truth robot position                                               
    std::vector<std::vector<double>> base_position_;
    
    ///@brief Ground truth robot orientation
    std::vector<std::vector<double>> base_orientation_; 
    
    ///@brief IMU data (linear acceleration)
    std::vector<Eigen::Vector3d> linear_acceleration_;
    
    ///@brief IMU data (angular velocity)
    std::vector<Eigen::Vector3d> angular_velocity_;
    
    ///@brief Joint states (encoder positions)
    std::vector<std::vector<double>> joint_states_;
    
    ///@brief Joint states (estimated velocities)
    std::vector<std::vector<double>> joint_velocities_;

    ///@brief Logged timestamps 
    std::vector<double> timestamps_;
    
    /// @brief Foot frame names
    std::vector<std::string> foot_frames_;

    /// @brief Robot joint names
    std::vector<std::string> joint_names_;
    
    /// @brief Path to the hyper tuner robot configuration file
    std::string robot_frame_config_;
    
    /// @brief Dataset names from experimentConfig.json
    std::string imu_linear_acceleration_dataset_;
    std::string imu_angular_velocity_dataset_;
    std::string joint_position_dataset_;
    std::string joint_velocity_dataset_;
    std::string gt_base_position_dataset_;
    std::string gt_base_orientation_dataset_;
    std::string timestamps_dataset_;
    std::vector<std::string> feet_force_dataset_;


    /// @brief Force data
    ForceTorqueData force_torque_;

    /// @brief Resolves the path of the input file based on the robot's json configuration file 
    /// @param config The experimentConfig.json file
    /// @param path 
    /// @return the path 
    std::string resolvePath(const json& config, const std::string& path);

    /// @brief Converts a 2D vector of doubles to a vector of Eigen::Vector3d
    /// @param data The 2D vector of doubles to convert
    /// @return A vector of Eigen::Vector3d containing the converted data
    std::vector<Eigen::Vector3d> convertToEigenVec3(const std::vector<std::vector<double>>& data);

    std::string base_path_;
    std::string experiment_type_;
    std::string robot_name_;
    std::string experiment_name_;
};
