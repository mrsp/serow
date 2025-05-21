#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <H5Cpp.h>
#include <nlohmann/json.hpp>



using json = nlohmann::ordered_json;

/// @brief DataManager class to manage data loading and configuration
class DataManager{

public:

    ///@brief Force data object definition 3D (x y z) for each leg
    struct ForceData {
        std::vector<std::vector<double>> FR;
        std::vector<std::vector<double>> FL;
        std::vector<std::vector<double>> RL;
        std::vector<std::vector<double>> RR;
    };

    /// @brief Constructor
    DataManager();
    
    /// @brief Gets the ground truth robot position data
    /// @return 2D vector of doubles with the robot position data
    std::vector<std::vector<double>> getPosData() const; 
    
    /// @brief Get the robot orientation data
    /// @return 2D vector of doubles with the robot orientation data
    std::vector<std::vector<double>> getRotData() const;
    
    /// @brief Get the linear acceleration data
    /// @return 2D vector of doubles with the linear acceleration data
    std::vector<std::vector<double>> getLinAccData() const;
    
    /// @brief Get the angular velocity data
    /// @return 2D vector of doubles with the angular velocity data
    std::vector<std::vector<double>> getAngVelData() const;
    
    /// @brief Get the joint states data
    /// @return 2D vector of doubles with the joint states data
    std::vector<std::vector<double>> getJointStatesData() const;

    /// @brief Get the joint velocities data
    /// @return 2D vector of doubles with the joint velocities data
    std::vector<std::vector<double>> getJointVelocitiesData() const;

    
    /// @brief Get the force data
    /// @return ForceData struct containing the force data
    ForceData getForceData() const;

    /// @brief Finds the path of SEROW from the environment variable SEROW_PATH
    /// @return the path of SEROW
    std::string getSerowPath() const;
    
    /// @brief Get the robot name from the configuration
    /// @return the robot name
    std::string getRobotName() const;

    std::vector<std::vector<double>> getTimestamps() const;

private: 
  
    /// @brief Loads the configuration file (experimentConfig.json) from the config directory
    void loadConfig();

    /// @brief Loads the data from the HDF5 file as defined in the experimentConfig.json file
    void loadData();

    /// @brief test configuration file
    nlohmann::json config_;
    /// @brief path to the data file
    std::string  DATA_FILE_;

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
    std::vector<std::vector<double>> robotPos_;
    
    ///@brief Ground truth robot orientation
    std::vector<std::vector<double>> robotRot_; 
    
    ///@brief IMU data (linear acceleration)
    std::vector<std::vector<double>> linAcc_;
    
    ///@brief IMU data (angular velocity)
    std::vector<std::vector<double>> angVel_;
    
    ///@brief Joint states (encoder positions)
    std::vector<std::vector<double>> jointStates_; // Magnetic Field
    
    ///@brief Joint states (encoder velocities)
    std::vector<std::vector<double>> jointVelocities_;
    ///@brief Logged timestamps 
    std::vector<std::vector<double>> timestamps_;
    
    /// @brief Foot frame names
    std::vector<std::string> foot_frames_;
    
    /// @brief Force data
    ForceData forceData_;

    /// @brief Resolves the path of the input file based on the robot's json configuration file 
    /// @param config The experimentConfig.json file
    /// @param path 
    /// @return the path 
    std::string resolvePath(const json& config, const std::string& path);


    std::string basePath;
    std::string experimentType;
    std::string robotName;
    std::string experimentName;

};
