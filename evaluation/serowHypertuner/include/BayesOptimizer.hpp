#pragma once
#include <nlohmann/json.hpp>

#include <DataManager.hpp>
#include <iostream>

#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>



using vectord = bayesopt::vectord;
using json = nlohmann::ordered_json;


class BayesOptimizer : public bayesopt::ContinuousModel
{
public:
  /// @brief Constructor
  /// @param dataManager_ The data manager object 
  BayesOptimizer(DataManager& dataManager, size_t dim, const bayesopt::Parameters& params);


  /// @brief Perform the optimization
  void optimize();

    double evaluateSample(const vectord &x) override
    {
      return 0.0; 
    }
private:
  /// @brief Data manager object reference
  DataManager& data_;

  /// @brief Load the default configuration file
  void loadBayesConfigJson();

  /// @brief Path to the settings for bayesOpt configuration file
  const std::string bayesParamsFilePath_ = data_.getSerowPath() + "evaluation/serowHypertuner/config/bayesSettings.json";
  
  /// @brief Bayesopt parameters
  bayesopt::Parameters params_;


};