#include <iostream>
#include <nlohmann/json.hpp>

#include "serow/Serow.hpp"

#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>

#include <BayesOptimizer.hpp>
#include <DataManager.hpp>

using json = nlohmann::ordered_json;





int main(int argc, char** argv)
{
  DataManager dataManager;
  bayesopt::Parameters params;
    params.n_init_samples = 30; //100 
    params.n_iterations = 100;  //400
    params.kernel.name = "kMaternARD5";
    params.surr_name = "sGaussianProcess";
    params.verbose_level = 2;
    params.crit_name = "cEI";
    params.n_iter_relearn = 10;  // Re-learn GP hyperparameters every 10 iterations
    params.noise = 0.01;  // Helps with numerical stability if your cost function isn't perfectly deterministic
    params.force_jump = 0.1; // Ensures smoother optimization paths (useful if the cost is smooth)
    params.epsilon = 0.01; // Exploitation threshold for EI
    params.random_seed = 42;

  BayesOptimizer bayesOptimizer(dataManager,10, params);
  
  
  
  
  
  
  
  
  
  
  
  
  // auto forceData = dataManager.getForceData();
  // auto posData = dataManager.getPosData();
  // auto rotData = dataManager.getRotData();
  // auto linAccData = dataManager.getLinAccData();
  // auto angVelData = dataManager.getAngVelData();
  // auto jointStatesData = dataManager.getJointStatesData();

  // for (const auto& pos : posData) {
  //   for (const auto& value : pos) {
  //     std::cout << value << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // for (const auto& rot : rotData) {
  //   for (const auto& value : rot) {
  //     std::cout << value << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // for (const auto& linAcc : linAccData) {
  //   for (const auto& value : linAcc) {
  //     std::cout << value << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // for (const auto& angVel : angVelData) {
  //   for (const auto& value : angVel) {
  //     std::cout << value << " ";
  //   }
  //   std::cout << std::endl;
  // }

  return 0;
}