
#include <string>
#include <iostream>
#include <vector>
#include <fstream>
#include <nlohmann/json.hpp>
#include <H5Cpp.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <bayesopt/bayesopt.hpp>
#include <bayesopt/parameters.hpp>


using json = nlohmann::ordered_json;


std::string getSerowPath() {
  const char* serowPath = std::getenv("SEROW_PATH");
  if (serowPath == nullptr) {
      throw std::runtime_error("SEROW_PATH environment variable not set");
  }
  return std::string(serowPath);
}


/// Parameters
std::string original_config = getSerowPath() + "/config/anymal_b.json";
std::string temp_config = getSerowPath() + "/config/anymal_b_temp.json";
std::string best_config_file = getSerowPath() + "/config/anymal_best_config.json";
std::string ground_truth_data = getSerowPath() + "/evaluation/fsc_test/anymal_fsc/fsc.h5";
std::string serow_predictions = getSerowPath() + "/evaluation/fsc_test/anymal_fsc/serow_predictions.h5";

std::vector<std::string> param_names = {
  "high_threshold",
  "low_threshold",
  "joint_cutoff_frequency",
  "angular_momentum_cutoff_frequency",
  "gyro_cutoff_frequency",
  "attitude_estimator_proportional_gain",
  "attitude_estimator_integral_gain",
  "joint_position_variance",

  "contact_position_covariance[0]",
  "contact_position_covariance[1]",
  "contact_position_covariance[2]",

  "contact_position_slip_covariance[0]",
  "contact_position_slip_covariance[1]",
  "contact_position_slip_covariance[2]"
};

json loadDefaultJson(const std::string& path)
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

void writeTempConfig(const json& j, const std::string& temp_path)
{
    std::ofstream out(temp_path);
    if (!out)
    {
        throw std::runtime_error("Failed to write temporary config file.");
    }
    out << std::setw(2) << j << std::endl;
}

void runSerow()
{
  std::string cmd = "/home/michael/github/serow/evaluation/fsc_test/build/fsc_test  anymal_b_temp.json";
  int ret_code = std::system(cmd.c_str());
  if (ret_code != 0)
  {
      std::cerr << "Error running serow estimator!" << std::endl;
      return; // Penalize failed runs
  }
}



// Reads dataset from HDF5 file (.h5)
std::vector<std::vector<double>> readHDF5(const std::string& filename,
  const std::string& datasetName) {
  H5::H5File file(filename, H5F_ACC_RDONLY);

  if (datasetName.empty()) {
    throw std::invalid_argument("Dataset name must be non-empty and start with '/'");
  }

  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[2];
  int ndims = dataspace.getSimpleExtentDims(dims);
  if (ndims != 2 || dims[0] == 0 || dims[1] == 0) {
    throw std::runtime_error("Unexpected dataset dimensions for " + datasetName);
  }

  std::vector<double> buffer(dims[0] * dims[1]);
  dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

  std::vector<std::vector<double>> data(dims[0], std::vector<double>(dims[1]));
  for (hsize_t i = 0; i < dims[0]; ++i) {
    for (hsize_t j = 0; j < dims[1]; ++j) {
     data[i][j] = buffer[i * dims[1] + j];
    }
  }
  return data;
}


std::vector<double> readHDF5_1D(const std::string& filename, const std::string& datasetName) {
  H5::H5File file(filename, H5F_ACC_RDONLY);

  if (datasetName.empty()) {
      throw std::invalid_argument("Dataset name must be non-empty.");
  }

  H5::DataSet dataset = file.openDataSet(datasetName);
  H5::DataSpace dataspace = dataset.getSpace();

  // Get number of dimensions and check
  int ndims = dataspace.getSimpleExtentNdims();
  if (ndims != 1) {
      throw std::runtime_error("Dataset " + datasetName + " is not 1D.");
  }

  hsize_t dim;
  dataspace.getSimpleExtentDims(&dim);

  std::vector<double> data(dim);
  dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

  return data;
}


class SerowOptimizer : public bayesopt::ContinuousModel
{
public:
    json j;
    SerowOptimizer(size_t dim, const bayesopt::Parameters& params)
    : ContinuousModel(dim, params)
    {
      gt_position_ = readHDF5(ground_truth_data, "base_ground_truth/position");
      gt_orientation_ = readHDF5(ground_truth_data, "base_ground_truth/orientation"); // x y z w
      std::cout << "Ground truth data loaded." << std::endl;
      
    }

    Eigen::Vector3d logMap(const Eigen::Matrix3d& R) {
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

    double evaluateSample(const vectord &x) override
    {   n_iterations_ ++;
        std::cout << "Optimization Iteration --> " << n_iterations_ << std::endl;

        // 1) Load & modify JSON for this x
        j = loadDefaultJson(original_config);
        for (size_t i = 0; i < param_names.size(); ++i)
        {
            const std::string& name = param_names[i];
        
            // Check if the parameter is an array item like "foo[1]"
            size_t open_bracket = name.find('[');
            size_t close_bracket = name.find(']');
        
            if (open_bracket != std::string::npos && close_bracket != std::string::npos)
            {
                // Extract key and index
                std::string key = name.substr(0, open_bracket);
                int index = std::stoi(name.substr(open_bracket + 1, close_bracket - open_bracket - 1));
        
                // Initialize or resize array if needed
                if (!j.contains(key) || !j[key].is_array())
                {
                    j[key] = std::vector<double>(index + 1, 0.0);
                }

                j[key][index] = x(i);
            }
            else
            {
                // Simple scalar assignment
                j[name] = x(i);
            }
        }


        writeTempConfig(j, temp_config);


        // 2) Run the estimator to produce fresh predictions
        runSerow();

        est_positions_.clear();
        est_orientations_.clear();
        std::cout << "Evaluating sample..." << std::endl;
        std::vector<double> pos_x = readHDF5_1D(serow_predictions, "base_pose/position/x");
        std::vector<double> pos_y = readHDF5_1D(serow_predictions, "base_pose/position/y");
        std::vector<double> pos_z = readHDF5_1D(serow_predictions, "base_pose/position/z");
        est_positions_.resize(pos_x.size());

        for (size_t i = 0 ; i < pos_x.size(); ++i)
        {
            est_positions_[i] = {pos_x[i], pos_y[i], pos_z[i]};
        }
        std::vector<double> rot_x = readHDF5_1D(serow_predictions, "base_pose/rotation/x");
        std::vector<double> rot_y = readHDF5_1D(serow_predictions, "base_pose/rotation/y");
        std::vector<double> rot_z = readHDF5_1D(serow_predictions, "base_pose/rotation/z");
        std::vector<double> rot_w = readHDF5_1D(serow_predictions, "base_pose/rotation/w");
        est_orientations_.resize(pos_x.size());

        for (size_t i = 0 ; i < pos_x.size(); ++i)
        {
          est_orientations_[i] = {rot_x[i], rot_x[i], rot_z[i], rot_w[i]};
        }
        

        std::cout << "Estimations loaded." << std::endl;

        double ate = computeATE();

        std::cout << "ATE for current parameters: " << ate << std::endl;
        return ate;
    }


    /// @brief Compute the Absolute Trajectory Error (ATE) between ground truth and estimated positions
    /// @return double: ATE value
    double computeATE() {
      std::cout << "Computing ATE..." << std::endl;
      std::cout << "Ground truth position size: " << gt_position_.size() << std::endl;
      std::cout << "Estimation position size: " << est_positions_.size() << std::endl;
      std::cout << "Ground truth orientation size: " << gt_orientation_.size() << std::endl;
      std::cout << "Estimation orientation size: " << est_orientations_.size() << std::endl;
      
      // Ensure we have data to compare
      if (gt_position_.empty() || est_positions_.empty() || gt_orientation_.empty() || est_orientations_.empty()) {
          std::cerr << "Error: Empty position data" << std::endl;
          return std::numeric_limits<double>::max();
      }

      
      // Use the minimum size in case they differ
      size_t count = std::min(gt_position_.size(), est_positions_.size());
      // Sum of squared position errors and orientation errors
      double position_error_sum = 0.0;
      double orientation_error_sum = 0.0;
      
      for (size_t i = 0; i < count; ++i) {
        // Calculate squared Euclidean distance for position
        double squared_position_distance = 0.0;
        size_t dims = std::min(gt_position_[i].size(), est_positions_[i].size());
        
        for (size_t j = 0; j < dims; ++j) {
            double diff = gt_position_[i][j] - est_positions_[i][j];
            squared_position_distance += diff * diff;
        }
        position_error_sum += std::sqrt(squared_position_distance); // L2 norm of position error
        
        // Calculate orientation error using the logMap function
        // First, convert quaternions to rotation matrices
        Eigen::Quaterniond gt_quat(gt_orientation_[i][3], gt_orientation_[i][0], 
                                  gt_orientation_[i][1], gt_orientation_[i][2]);
        Eigen::Quaterniond est_quat(est_orientations_[i][3], est_orientations_[i][0], 
                                   est_orientations_[i][1], est_orientations_[i][2]);
        
        Eigen::Matrix3d gt_R = gt_quat.toRotationMatrix();
        Eigen::Matrix3d est_R = est_quat.toRotationMatrix();
        
        // Apply logMap to calculate orientation error (similar to the reward function)
        Eigen::Matrix3d rel_R = gt_R.transpose() * est_R;
        Eigen::Vector3d omega = logMap(rel_R);
        double orientation_error = omega.norm();
        orientation_error_sum += orientation_error;
      }

      double avg_position_error = position_error_sum / count;
      double avg_orientation_error = orientation_error_sum / count;
      double position_weight = 1.0;  
      double orientation_weight = 0.5;
      double combined_ate = position_weight * avg_position_error + 
                          orientation_weight * avg_orientation_error;
    
      std::cout << "Position ATE: " << avg_position_error << std::endl;
      std::cout << "Orientation ATE: " << avg_orientation_error << std::endl;
      std::cout << "Combined ATE: " << combined_ate << std::endl;

      return combined_ate;
  }

    // double optimize(vectord &result);
private:
  std::vector<std::vector<double>> gt_position_;
  std::vector<std::vector<double>> gt_orientation_;
  std::vector<std::vector<double>> est_positions_;
  std::vector<std::vector<double>> est_orientations_;
  int n_iterations_ = 0;
};



int main()
{
    // Define optimizer params
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
    // Search bounds: define range for each parameter
    SerowOptimizer optimizer(param_names.size(), params);

    vectord lower_bounds(param_names.size());
    vectord upper_bounds(param_names.size());
    

    // "high_threshold",
    lower_bounds(0) = 40.0;
    upper_bounds(0) = 80.0;
    // "low_threshold",
    lower_bounds(1) = 0.0;  
    upper_bounds(1) = 40.0;  
    // "joint_cutoff_frequency",    
    lower_bounds(2) = 1.0;  
    upper_bounds(2) = 100.0;  
    // "angular_momentum_cutoff_frequency",
    lower_bounds(3) = 1.0;  
    upper_bounds(3) = 20.0;  
    // "gyro_cutoff_frequency",
    lower_bounds(4) = 1.0;  
    upper_bounds(4) = 100.0;  
    // "attitude_estimator_proportional_gain",
    lower_bounds(5) = 0.0;  
    upper_bounds(5) = 4.5;  
    // "attitude_estimator_integral_gain"
    lower_bounds(6) = 0.0;  
    upper_bounds(6) = 4.5;  
    // "joint_position_variance"
    lower_bounds(7) = 1e-5;
    upper_bounds(7) = 1e-1;

    for (int i = 8 ; i < param_names.size(); ++i){
      lower_bounds(i) = 1e-7;
      upper_bounds(i) = 1e-1;
    }

    if (lower_bounds.size() == param_names.size() && upper_bounds.size() == param_names.size()){
      std::cout << "Bounds set successfully" << std::endl;
    }

    optimizer.setBoundingBox(
      lower_bounds,  // lower bounds (just examples)
      upper_bounds   // upper bounds
    );
  
  vectord result(param_names.size());  // Pre-allocate result vector
  optimizer.optimize(result);


  std::cout << "Finished Optimization" << std::endl;
  
  // SAVE BEST CONFIG
  json best_config = loadDefaultJson(original_config);

  // Save best config using param_names
  for (size_t i = 0; i < param_names.size(); ++i) {
    const std::string& param = param_names[i];
    
    size_t open_bracket = param.find('[');
    size_t close_bracket = param.find(']');

    if (open_bracket != std::string::npos && close_bracket != std::string::npos && close_bracket > open_bracket) {
        // It's an array element like "imu_covariance[2]"
        std::string base_name = param.substr(0, open_bracket);
        int index = std::stoi(param.substr(open_bracket + 1, close_bracket - open_bracket - 1));

        if (best_config.contains(base_name) && best_config[base_name].is_array() && index < best_config[base_name].size()) {
            best_config[base_name][index] = result(i);
        } else {
            std::cerr << "Warning: Invalid array parameter '" << param << "' in config.\n";
        }
    } else {
        // Regular scalar parameter
        if (best_config.contains(param)) {
            best_config[param] = result(i);
        } else {
            std::cerr << "Warning: Parameter '" << param << "' not found in config.\n";
        }
    }
  }
    
  // Save best config to file (optional: different path like "best_config.json")
  writeTempConfig(best_config, best_config_file);



  // Print the optimized parameters and final error
  std::cout << "\n===== OPTIMIZATION RESULTS =====\n";
  std::cout << "Best parameters found:\n";
  for (size_t i = 0; i < param_names.size(); ++i) {
      std::cout << param_names[i] << ": " << result(i) << std::endl;
  }
  return 0;
}