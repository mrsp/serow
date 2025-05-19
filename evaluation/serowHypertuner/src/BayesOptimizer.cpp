#include <BayesOptimizer.hpp>

BayesOptimizer::BayesOptimizer(DataManager& dataManager, 
  size_t dim, 
  const bayesopt::Parameters& params,
  const std::vector<std::string> &params2optimize_)
: data_(dataManager), ContinuousModel(dim,params), params2optimize_(params2optimize_)
{    
  urdf_file_path_ = data_.getSerowPath() + "/urdf/" + data_.getRobotName() + ".urdf";
  original_config = data_.getSerowPath() + "/config/" + data_.getRobotName() + ".json";
  temp_config = data_.getRobotName() +"_temp.json";


  pmodel_ = std::make_unique<pinocchio::Model>();

  pinocchio::urdf::buildModel(urdf_file_path_, *pmodel_, true);


  // Load the ground truth data
  gt_position_ = data_.getPosData();
  gt_orientation_ = data_.getRotData();

  gt_position_ = data_.getPosData();
  gt_orientation_ = data_.getRotData();

  timestamps_ = data_.getTimestamps();
}

void BayesOptimizer::writeJSONConfig(const json& j, const std::string& path)
{
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to write temporary config file.");
    }
    out << std::setw(2) << j << std::endl;
}

json BayesOptimizer::loadDefaultJson(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("Failed to open base config file.");
    }
    json j;
    in >> j;
    return j;
}

Eigen::Vector3d BayesOptimizer::logMap(const Eigen::Matrix3d& R) {
      double R11 = R(0, 0);
      double R12 = R(0, 1);
      double R13 = R(0, 2);
      double R21 = R(1, 0);
      double R22 = R(1, 1);
      double R23 = R(1, 2);
      double R31 = R(2, 0);
      double R32 = R(2, 1);
      double R33 = R(2, 2);

      double trace = R.trace();

      Eigen::Vector3d omega = Eigen::Vector3d::Zero();

      // Special case when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
      if (trace + 1.0 < 1e-3) {
          if (R33 > R22 && R33 > R11) {
              // R33 is the largest diagonal, a=3, b=1, c=2
              double W = R21 - R12;
              double Q1 = 2.0 + 2.0 * R33;
              double Q2 = R31 + R13;
              double Q3 = R23 + R32;
              double r = std::sqrt(Q1);
              double one_over_r = 1 / r;
              double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
              double sgn_w = W < 0 ? -1.0 : 1.0;
              double mag = M_PI - (2 * sgn_w * W) / norm;
              double scale = 0.5 * one_over_r * mag;
              omega = sgn_w * scale * Eigen::Vector3d(Q2, Q3, Q1);
          }
          else if (R22 > R11) {
              // R22 is the largest diagonal, a=2, b=3, c=1
              double W = R13 - R31;
              double Q1 = 2.0 + 2.0 * R22;
              double Q2 = R23 + R32;
              double Q3 = R12 + R21;
              double r = std::sqrt(Q1);
              double one_over_r = 1 / r;
              double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
              double sgn_w = W < 0 ? -1.0 : 1.0;
              double mag = M_PI - (2 * sgn_w * W) / norm;
              double scale = 0.5 * one_over_r * mag;
              omega = sgn_w * scale * Eigen::Vector3d(Q3, Q1, Q2);
          }
          else {
              // R11 is the largest diagonal, a=1, b=2, c=3
              double W = R32 - R23;
              double Q1 = 2.0 + 2.0 * R11;
              double Q2 = R12 + R21;
              double Q3 = R31 + R13;
              double r = std::sqrt(Q1);
              double one_over_r = 1 / r;
              double norm = std::sqrt(Q1 * Q1 + Q2 * Q2 + Q3 * Q3 + W * W);
              double sgn_w = W < 0 ? -1.0 : 1.0;
              double mag = M_PI - (2 * sgn_w * W) / norm;
              double scale = 0.5 * one_over_r * mag;
              omega = sgn_w * scale * Eigen::Vector3d(Q1, Q2, Q3);
          }
      }
      else {
          double magnitude = 0.0;
          double tr_3 = trace - 3.0;  // could be non-negative if the matrix is off orthogonal
          if (tr_3 < -1e-6) {
              // this is the normal case -1 < trace < 3
              double theta = std::acos((trace - 1.0) / 2.0);
              magnitude = theta / (2.0 * std::sin(theta));
          }
          else {
              // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
              // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
              magnitude = 0.5 - tr_3 / 12.0 + tr_3 * tr_3 / 60.0;
          }

          omega = magnitude * Eigen::Vector3d(R32 - R23, R13 - R31, R21 - R12);
      }
      return omega;
    }


double BayesOptimizer::evaluateSample(const vectord &x) {
  n_iterations_ ++;
  std::cout << "\033[32m \nOptimization iteration:\033[0m" << n_iterations_  << std::endl;
  
  // Reinitialize estimations in each iteration
  est_positions_.clear();
  est_orientations_.clear();

  temp_json = loadDefaultJson(original_config);
  
  for (size_t i = 0; i < params2optimize_.size(); ++i)
  {
      const std::string& name = params2optimize_[i];
  
      // Check if the parameter is an array item like "foo[1]"
      size_t open_bracket = name.find('[');
      size_t close_bracket = name.find(']');
  
      if (open_bracket != std::string::npos && close_bracket != std::string::npos)
      {
          // Extract key and index
          std::string key = name.substr(0, open_bracket);
          int index = std::stoi(name.substr(open_bracket + 1, close_bracket - open_bracket - 1));
  
          // Initialize or resize array if needed
          if (!temp_json.contains(key) || !temp_json[key].is_array())
          {
              temp_json[key] = std::vector<double>(index + 1, 0.0);
          }

          temp_json[key][index] = x(i);
      }
      else
      {
          // Simple scalar assignment
          temp_json[name] = x(i);
      }
  }

  writeJSONConfig(temp_json, temp_config);
  
  // Initialize Serow with the modified config
  if (!SEROW.initialize(temp_config)) {
    throw std::runtime_error("Failed to initialize Serow with config: " + temp_config);
  }







  return 42.0; // Replace with your real evaluation logic
}
