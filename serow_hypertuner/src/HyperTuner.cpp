#include <iostream>
#include <nlohmann/json.hpp>

#include "serow/Serow.hpp"

#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>

#include <BayesOptimizer.hpp>
#include <DataManager.hpp>

using json = nlohmann::ordered_json;
using vectord = bayesopt::vectord;
using namespace serow;


bayesopt::Parameters setBayesSettings(const json & bayes_config_params) {
  bayesopt::Parameters settings;

  // Extract basic parameters
  settings.n_init_samples = bayes_config_params["optimizer"]["n_init_samples"];
  settings.n_iterations = bayes_config_params["optimizer"]["n_iterations"];
  settings.kernel.name = bayes_config_params["optimizer"]["kernel"];
  settings.surr_name = bayes_config_params["optimizer"]["surrogate"];
  settings.verbose_level = bayes_config_params["optimizer"]["verbose_level"];
  settings.crit_name = bayes_config_params["optimizer"]["criterion"];
  settings.n_iter_relearn = bayes_config_params["optimizer"]["n_iter_relearn"];
  settings.noise = bayes_config_params["optimizer"]["noise"];
  settings.force_jump = bayes_config_params["optimizer"]["force_jump"];
  settings.epsilon = bayes_config_params["optimizer"]["epsilon"];
  settings.random_seed = bayes_config_params["optimizer"]["random_seed"];

  return settings;
}

json loadBayesSettingsConfig(const std::string & serow_path) {
  std::string  bayes_params_file_path = serow_path + "serow_hypertuner/config/bayesSettings.json";
  std::cout << "Loading Bayes optimization parameters from: " <<  bayes_params_file_path << std::endl;
  // Open and parse the JSON file
  std::ifstream file( bayes_params_file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " +  bayes_params_file_path);
  }
    
  json bayes_config_params;
  file >> bayes_config_params;
  return bayes_config_params;
}

std::pair<vectord, vectord> readOptimizationParams(const std::string& jsonFilePath,
                            std::vector<std::string>& param_names,
                            vectord& lower_bounds,
                            vectord& upper_bounds) {
  std::ifstream file(jsonFilePath);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << jsonFilePath << std::endl;
    throw std::runtime_error("Failed to open file: " + jsonFilePath);
  }

  json j;
  file >> j;

  if (!j.contains("serowHyperparams") || !j.contains("bounds") ||
    !j["bounds"].contains("lower") || !j["bounds"].contains("upper")) {
    std::cerr << "Invalid JSON format in: " << jsonFilePath << std::endl;
    throw std::runtime_error("Mismatch between number of parameters and bounds");
  }

  param_names = j["serowHyperparams"].get<std::vector<std::string>>();
  std::cout << "Total hyperparameters for optimization:" << param_names.size() << " \nParameters to optimize: " << std::endl;
  for (const auto& param : param_names) {
    std::cout << "  -- " << param << std::endl;
  }
  std::vector<double> lower = j["bounds"]["lower"].get<std::vector<double>>();
  std::vector<double> upper = j["bounds"]["upper"].get<std::vector<double>>();

  if (lower.size() != param_names.size() || upper.size() != param_names.size()) {
    std::cerr << "Mismatch between number of parameters and bounds" << std::endl;
    throw std::runtime_error("Mismatch between number of parameters and bounds");
  }

  lower_bounds = vectord(param_names.size());
  upper_bounds = vectord(param_names.size());

  for (size_t i = 0; i < param_names.size(); ++i) {
    lower_bounds(i) = lower[i];
    upper_bounds(i) = upper[i];
  }

  return std::make_pair(lower_bounds, upper_bounds);
}

int main(int argc, char** argv)
{
  DataManager data_manager;

  json settings_config = loadBayesSettingsConfig(data_manager.getSerowPath());
  bayesopt::Parameters settings = setBayesSettings(settings_config);
    
  std::vector<std::string> params_to_optimize;
  vectord lower_bounds, upper_bounds;

  std::pair<vectord, vectord> bounds = readOptimizationParams(
    data_manager.getSerowPath() + "serow_hypertuner/config/params_to_optimize.json",
    params_to_optimize,
    lower_bounds,
    upper_bounds
  );

  BayesOptimizer bayes_optimizer(data_manager, params_to_optimize.size(), settings, settings_config["ate_params"]["position_weight"], settings_config["ate_params"]["orientation_weight"],  params_to_optimize);
  
  bayes_optimizer.setBoundingBox(
    lower_bounds,
    upper_bounds
  );
  
  vectord result(params_to_optimize.size());  // Pre-allocate result vector

  bayes_optimizer.optimize(result);

  std::cout << "Finished Optimization" << std::endl;

  bayes_optimizer.saveBestConfig(result);

  return 0;
}
