#include <BayesOptimizer.hpp>

BayesOptimizer::BayesOptimizer(DataManager& dataManager, size_t dim, const bayesopt::Parameters& params)
: data_(dataManager), ContinuousModel(dim,params)
{
    // Constructor implementation
    loadBayesConfigJson();

}


void BayesOptimizer::loadBayesConfigJson()
{
  std::cout << "Loading Bayes optimization parameters from: " << bayesParamsFilePath_ << std::endl;
  try {
    // Open and parse the JSON file
    std::ifstream file(bayesParamsFilePath_);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open file: " + bayesParamsFilePath_);
    }
    
    json bayesConfigParams;
    file >> bayesConfigParams;
    
    
    // Extract basic parameters
    params_.n_init_samples = bayesConfigParams["optimizer"]["n_init_samples"];
    params_.n_iterations = bayesConfigParams["optimizer"]["n_iterations"];
    params_.kernel.name = bayesConfigParams["optimizer"]["kernel"]["name"];
    params_.surr_name = bayesConfigParams["optimizer"]["surrogate"];
    params_.verbose_level = bayesConfigParams["optimizer"]["verbose_level"];
    params_.crit_name = bayesConfigParams["optimizer"]["criterion"];
    params_.n_iter_relearn = bayesConfigParams["optimizer"]["n_iter_relearn"];
    params_.noise = bayesConfigParams["optimizer"]["noise"];
    params_.force_jump = bayesConfigParams["optimizer"]["force_jump"];
    params_.epsilon = bayesConfigParams["optimizer"]["epsilon"];
    params_.random_seed = bayesConfigParams["optimizer"]["random_seed"];
  } catch (const std::exception& e) {
    std::cerr << "Error loading hyperparameters: " << e.what() << std::endl;
    return;
  }
  return;
}

void BayesOptimizer::optimize()
{
  // // Load the data
  // data_.loadData();
  // std::cout << "Data loaded successfully." << std::endl;
  
  // // Initialize the optimizer with the loaded parameters
  // optimizer_.setParameters(params_);
  
  // // Perform optimization
  // optimizer_.optimize();
  
  // std::cout << "Optimization completed." << std::endl;
}