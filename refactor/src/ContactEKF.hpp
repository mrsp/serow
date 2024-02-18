/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once
/**
 * @brief Base Estimator combining Inertial Measurement Unit (IMU) and Odometry Measuruements either
 * from leg odometry or external odometry e.g Visual Odometry (VO) or Lidar Odometry (LO)
 * @author Stylianos Piperakis
 * @details State is  position in World frame
 * velocity in  Base frame
 * orientation of Body frame wrt the World frame
 * accelerometer bias in Base frame
 * gyro bias in Base frame
 * Measurements are: Base Position/Orinetation in World frame by Leg Odometry or Visual Odometry
 * (VO) or Lidar Odometry (LO), when VO/LO is considered the kinematically computed base velocity
 * (Twist) is also employed for update.
 */

#include <iostream>

#include "Measurement.hpp"
#include "State.hpp"

namespace serow {

// State is velocity  - orientation - position - gyro bias - accel bias - 15 + 6 x N contact
// position - contact orientation
class ContactEKF {
   public:
    void init(const BaseState& state, std::unordered_set<std::string> contacts_frame,
              bool point_feet, double g, double imu_rate);
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
};

}  // namespace serow
