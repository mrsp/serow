#pragma once
#include <nlohmann/json.hpp>

#include <DataManager.hpp>
#include <iostream>
#include "serow/Serow.hpp"
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using vectord = bayesopt::vectord;
using json = nlohmann::ordered_json;

class BayesOptimizer : public bayesopt::ContinuousModel
{
public:
  /// @brief Constructor
  /// @param dataManager_ The data manager object 
  BayesOptimizer(DataManager& dataManager, size_t dim, const bayesopt::Parameters& params, const std::vector<std::string> &param_names);

  /// @brief Evaluates the sample Overrides virtual optimization function from bayesopt
  /// @param x a vectord object containing the parameters to evaluate
  /// @return Defined metric for evaluation of sample
  double evaluateSample(const vectord &x) override;


  /// @brief Saves the best configuration to a file
  void saveBestConfig(vectord result);
  
private:
  /// @brief Data manager object reference
  DataManager& data_;

  /// @brief Serow object reference
  serow::Serow SEROW;

  std::unique_ptr<pinocchio::Model> pmodel_;

  /// @brief Bayesopt parameters
  bayesopt::Parameters params_;

  /// @brief Hyperparameteres to optimize
  std::vector<std::string> params2optimize_;

  /// @brief Computes the logMap of a rotation matrix
  Eigen::Vector3d logMap(const Eigen::Matrix3d& R);

  /// @brief Writes the JSON "j" configuration to a file at "path"
  /// @param j The json object to write
  /// @param path the path to the file
  void writeJSONConfig(const json& j, const std::string& path);


  /// @brief Loads the default JSON configuration from a file
  /// @param path the path to the file
  /// @return the json object
  json loadDefaultJson(const std::string& path);

  /// @brief Sets the robot's joint names, frame names, and foot names
  void setRobotJointsAndFootFrames(const std::string & robot_path);

  /// @brief Prints the robot's information including joint names and foot frames
  void printRobotInfo();

  /// @brief Computes the Absolute Trajectory Error (ATE) between estimated and ground truth positions
  double computeATE();

  /// @brief Counter for the number of iterations
  size_t n_iterations_ = 0;
  
  /// @brief Joint names vector
  std::vector<std::string> joint_names_;

  /// @brief Foot frames
  std::vector<std::string> foot_names_;

  /// @brief Serow estimated positions
  std::vector<std::vector<double>> est_positions_;

  /// @brief Serow estimated orientations
  std::vector<std::vector<double>> est_orientations_;

  /// @brief Ground truth positions
  std::vector<std::vector<double>> gt_position_;

  /// @brief Ground truth orientations
  std::vector<std::vector<double>> gt_orientation_;

  /// @brief Data timestamps
  std::vector<std::vector<double>> timestamps_;

  /// @brief Feet force data
  DataManager::ForceData force_measurements_;

  /// @brief Joint positions
  std::vector<std::vector<double>> joint_positions_;

  /// @brief Joint velocities
  std::vector<std::vector<double>> joint_velocities_;

  ///@brief IMU data (linear acceleration)
  std::vector<std::vector<double>> linear_acceleration_;

  ///@brief IMU data (angular velocity)
  std::vector<std::vector<double>> angular_velocity_;
      
  /// @brief Original config file path
  std::string original_config;

  /// @brief Temporary config file path
  std::string temp_config;

  /// @brief Robot joint & foot frames definition
  std::string robot_frame_config;

  /// @brief URDF file path
  std::string urdf_file_path;
  
  /// @brief Temporary json object
  json temp_json;

  /// @brief Optimization dataset percentage. e.g dataset_size * datset_percentage = total data to be used for optimization 
  int dataset_percentage = 0.6; // %

};