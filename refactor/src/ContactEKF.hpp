/**
* Copyright (C) 2024 Stylianos Piperakis, Ownage Dynamics L.P.
* Serow is free software: you can redistribute it and/or modify it under the terms of the GNU 
* General Public License as published by the Free Software Foundation, version 3.
* 
* Serow is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
* even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
* General Public License for more details.
* 
* You should have received a copy of the GNU General Public License along with Serow. If not, 
* see <https://www.gnu.org/licenses/>.
**/
/**
 * @brief Base Estimator combining Inertial Measurement Unit (IMU), relative to the base leg contact 
 * measurements and an external odometry e.g Visual Odometry (VO) or Lidar Odometry (LO)  
 * @author Stylianos Piperakis
 * @details State is: 
 * Base position in world frame
 * Base linear velocity in base frame
 * Base orientation w.r.t the world frame
 * Gyro bias in base frame
 * Accelerometer bias in base frame
 * Leg contact position in world frame
 * Leg contact orientation in world frame
 * More info in Nonlinear State Estimation for Humanoid Robot Walking
 * https://www.researchgate.net/publication/326194869_Nonlinear_State_Estimation_for_Humanoid_Robot_Walking
 */
#pragma once

#include "Measurement.hpp"
#include "State.hpp"

#include <iostream>

namespace serow {

struct OutlierDetector {
    // Beta distribution parameters more info in Outlier-Robust State Estimation for Humanoid Robots
    // https://www.researchgate.net/publication/334745931_Outlier-Robust_State_Estimation_for_Humanoid_Robots
    double zeta = 1.0;
    double f_0 = 0.1;
    double e_0 = 0.9;
    double f_t = 0.1;
    double e_t = 0.9;
    double threshold = 1e-5;
    int iters = 4;

    // Digamma function approximation
    double computePsi(double xxx);

    // Initializes the estimation
    void init();

    // Compute the outlier indicator zeta
    void estimate(const Eigen::Matrix3d& BetaT, const Eigen::Matrix3d& R);
};

// State is velocity  - orientation - position - gyro bias - accel bias - 15 + 6 x N contact
// position - contact orientation
class ContactEKF {
   public:
    void init(const BaseState& state, std::unordered_set<std::string> contacts_frame,
              bool point_feet, double g, double imu_rate, bool outlier_detection = false);
    BaseState predict(const BaseState& state, const ImuMeasurement& imu,
                      const KinematicMeasurement& kin);
    BaseState update(const BaseState& state, const KinematicMeasurement& kin,
                     std::optional<OdometryMeasurement> odom = std::nullopt,
                     std::optional<TerrainMeasurement> terrain = std::nullopt);

   private:
    int num_states_{};
    int num_inputs_{};
    int contact_dim_{};
    int num_leg_end_effectors_{};
    std::unordered_set<std::string> contacts_frame_;
    bool point_feet_{};
    bool outlier_detection_{};
    // Predict step sampling time
    double nominal_dt_{};
    // Gravity vector
    Eigen::Vector3d g_;
    // State indices
    Eigen::Array3i v_idx_;
    Eigen::Array3i r_idx_;
    Eigen::Array3i p_idx_;
    Eigen::Array3i bg_idx_;
    Eigen::Array3i ba_idx_;
    std::unordered_map<std::string, Eigen::Array3i> pl_idx_;
    std::unordered_map<std::string, Eigen::Array3i> rl_idx_;
    // Input indices
    Eigen::Array3i ng_idx_;
    Eigen::Array3i na_idx_;
    Eigen::Array3i nbg_idx_;
    Eigen::Array3i nba_idx_;
    std::unordered_map<std::string, Eigen::Array3i> npl_idx_;
    std::unordered_map<std::string, Eigen::Array3i> nrl_idx_;
    // Previous imu timestamp
    std::optional<double> last_imu_timestamp_;

    /// Error Covariance, Linearized state transition model, Identity matrix, state uncertainty
    /// matrix 15 + 6N x 15 + 6N
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> I_, P_;
    /// Linearized state-input model 15 + 6N x 12 + 6N
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Lc_;

    OutlierDetector contact_outlier_detector;

    BaseState computeDiscreteDynamics(
        const BaseState& state, double dt, Eigen::Vector3d angular_velocity,
        Eigen::Vector3d linear_acceleration,
        std::optional<std::unordered_map<std::string, bool>> contacts_status,
        std::optional<std::unordered_map<std::string, Eigen::Vector3d>> contacts_position,
        std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientations =
            std::nullopt);

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> computePredictionJacobians(
        const BaseState& state, Eigen::Vector3d angular_velocity,
        Eigen::Vector3d linear_acceleration, double dt);

    BaseState updateWithContacts(
        const BaseState& state,
        const std::unordered_map<std::string, Eigen::Vector3d>& contacts_position,
        std::unordered_map<std::string, Eigen::Matrix3d> contacts_position_noise,
        const std::unordered_map<std::string, bool>& contacts_status,
        const Eigen::Matrix3d& position_cov,
        std::optional<std::unordered_map<std::string, Eigen::Quaterniond>> contacts_orientation,
        std::optional<std::unordered_map<std::string, Eigen::Matrix3d>> contacts_orientation_noise,
        std::optional<Eigen::Matrix3d> orientation_cov);

    BaseState updateWithOdometry(const BaseState& state, const Eigen::Vector3d& base_position,
                                 const Eigen::Quaterniond& base_orientation,
                                 const Eigen::Matrix3d& base_position_cov,
                                 const Eigen::Matrix3d& base_orientation_cov);

    BaseState updateWithTerrain(const BaseState& state,
                                const std::unordered_map<std::string, bool>& contacts_status,
                                double terrain_height, double terrain_cov);

    void updateState(BaseState& state, const Eigen::VectorXd& dx, const Eigen::MatrixXd& P) const;
    BaseState updateStateCopy(const BaseState& state, const Eigen::VectorXd& dx,
                              const Eigen::MatrixXd& P) const;
};

}  // namespace serow
