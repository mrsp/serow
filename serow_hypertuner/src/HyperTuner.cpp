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


bayesopt::Parameters loadBayesConfigJson(const std::string & serow_path)
{
  bayesopt::Parameters settings;

  std::string bayesParamsFilePath = serow_path + "serow_hypertuner/config/bayesSettings.json";
  std::cout << "Loading Bayes optimization parameters from: " << bayesParamsFilePath << std::endl;
  try {
    // Open and parse the JSON file
    std::ifstream file(bayesParamsFilePath);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + bayesParamsFilePath);
    }
    
    json bayesConfigParams;
    file >> bayesConfigParams;
    // Extract basic parameters
    settings.n_init_samples = bayesConfigParams["optimizer"]["n_init_samples"];
    settings.n_iterations = bayesConfigParams["optimizer"]["n_iterations"];
    settings.kernel.name = bayesConfigParams["optimizer"]["kernel"];
    settings.surr_name = bayesConfigParams["optimizer"]["surrogate"];
    settings.verbose_level = bayesConfigParams["optimizer"]["verbose_level"];
    settings.crit_name = bayesConfigParams["optimizer"]["criterion"];
    settings.n_iter_relearn = bayesConfigParams["optimizer"]["n_iter_relearn"];
    settings.noise = bayesConfigParams["optimizer"]["noise"];
    settings.force_jump = bayesConfigParams["optimizer"]["force_jump"];
    settings.epsilon = bayesConfigParams["optimizer"]["epsilon"];
    settings.random_seed = bayesConfigParams["optimizer"]["random_seed"];
  } catch (const std::exception& e) {
    std::cerr << "Error loading hyperparameters: " << e.what() << std::endl;
    throw;
  }
  return settings;
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
  DataManager dataManager;

  auto settings = loadBayesConfigJson(dataManager.getSerowPath());
    
  std::vector<std::string> params_to_optimize;
  vectord lower_bounds, upper_bounds;

  std::pair<vectord, vectord> bounds = readOptimizationParams(
    dataManager.getSerowPath() + "serow_hypertuner/config/params_to_optimize.json",
    params_to_optimize,
    lower_bounds,
    upper_bounds
  );

  BayesOptimizer bayesOptimizer(dataManager, params_to_optimize.size(), settings, params_to_optimize);
  
  bayesOptimizer.setBoundingBox(
    lower_bounds,
    upper_bounds
  );
  
  vectord result(params_to_optimize.size());  // Pre-allocate result vector

  bayesOptimizer.optimize(result);

  std::cout << "Finished Optimization" << std::endl;

  bayesOptimizer.saveBestConfig(result);

  return 0;
}
