#pragma once
#include <nlohmann/json.hpp>

#include <DataManager.hpp>
#include <iostream>
#include <cstdio>
#include "serow/Serow.hpp"
#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include "serow/lie.hpp"

using vectord = bayesopt::vectord;
using json = nlohmann::ordered_json;

class BayesOptimizer : public bayesopt::ContinuousModel
{
public:
  /// @brief Constructor
  /// @param dataManager_ The data manager object
  /// @param dim The dimension of the optimization problem
  /// @param params The bayesopt parameters
  /// @param param_names The names of the parameters to optimize
  BayesOptimizer(DataManager& data_manager, size_t dim, const bayesopt::Parameters& params, const double ate_position_weight, const double ate_orientation_weight, const std::vector<std::string> &param_names);

  /// @brief Destructor, removes the temporary config file
  ~BayesOptimizer();

  /// @brief Evaluates the sample. Overrides virtual optimization function from bayesopt
  /// @param x A vectord object containing the parameters to evaluate
  /// @return Defined metric for evaluation of sample
  double evaluateSample(const vectord& x) override;

  /// @brief Saves the best configuration to a file
  void saveBestConfig(const vectord& result);

private:
  /// @brief Data manager object reference
  DataManager& data_;

  /// @brief Serow object reference
  serow::Serow estimator_;

  /// @brief Bayesopt parameters
  bayesopt::Parameters params_;

  /// @brief Hyperparameteres to optimize
  std::vector<std::string> params_to_optimize_;

  /// @brief Counter for the number of iterations
  size_t n_iterations_ = 0;
  
  /// @brief Joint names vector
  std::vector<std::string> joint_names_;

  /// @brief Foot frames
  std::vector<std::string> foot_frames_;

  /// @brief Serow estimated positions
  std::vector<std::vector<double>> est_positions_;

  /// @brief Serow estimated orientations
  std::vector<std::vector<double>> est_orientations_;

  /// @brief Ground truth positions
  std::vector<std::vector<double>> gt_position_;

  /// @brief Ground truth orientations
  std::vector<std::vector<double>> gt_orientation_;

  /// @brief Data timestamps
  std::vector<double> timestamps_;

  /// @brief Feet force data
  DataManager::ForceTorqueData ft_measurements_;

  /// @brief Joint positions
  std::vector<std::vector<double>> joint_positions_;

  /// @brief Joint velocities
  std::vector<std::vector<double>> joint_velocities_;

  ///@brief IMU data (linear acceleration)
  std::vector<Eigen::Vector3d> linear_acceleration_;

  ///@brief IMU data (angular velocity)
  std::vector<Eigen::Vector3d> angular_velocity_;
      
  /// @brief Original config file path
  std::string original_config_;

  /// @brief Temporary config file path
  std::string temp_config_;

  /// @brief Robot joint and foot frames definition
  std::string robot_frame_config_;

  /// @brief Path to the Serow installation
  std::string serow_path_;

  /// @brief Temporary json object
  json temp_json_;

  /// @brief Optimization dataset percentage. e.g dataset_size * datset_percentage = total data to be used for optimization 
  double dataset_percentage_ = 1.0;

  /// @brief Weight for the Absolute Trajectory Error (ATE) position component
  double ate_position_weight_;

  /// @brief Weight for the Absolute Trajectory Error (ATE) orientation component
  double ate_orientation_weight_;

  /// @brief Writes the JSON "j" configuration to a file at "path"
  /// @param j The json object to write
  /// @param path the path to the file
  void writeJSONConfig(const json& j, const std::string& path);

  /// @brief Loads the default JSON configuration from a file
  /// @param path the path to the file
  /// @return the json object
  json loadDefaultJson(const std::string& path);

  /// @brief Prints the robot's information including joint names and foot frames
  void printRobotInfo();

  /// @brief Computes the Absolute Trajectory Error (ATE) between estimated and ground truth poses
  double computeATE();
};
