/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once
#include <gtest/gtest_prod.h>

#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "CoMEKF.hpp"
#include "ContactDetector.hpp"
#include "ContactEKF.hpp"
#include "DerivativeEstimator.hpp"
#include "JointEstimator.hpp"
#include "LegOdometry.hpp"
#include "Mahony.hpp"
#include "Measurement.hpp"
#include "RobotKinematics.hpp"
#include "State.hpp"

namespace serow {

class Serow {
   public:
    Serow(std::string config);

    void filter(
        ImuMeasurement imu, std::unordered_map<std::string, JointMeasurement> joints,
        std::optional<std::unordered_map<std::string, ForceTorqueMeasurement>> ft = std::nullopt,
        std::optional<OdometryMeasurement> odom = std::nullopt,
        std::optional<std::unordered_map<std::string, ContactMeasurement>> contact_probabilities =
            std::nullopt);

    std::optional<State> getState(bool allow_invalid = false);

   private:
    struct Params {
        double mass{};
        double g{};
        bool calibrate_imu{};
        double max_imu_calibration_cycles{};
        double imu_rate{};
        double gyro_cutoff_frequency{};
        Eigen::Vector3d bias_gyro{Eigen::Vector3d::Zero()};
        Eigen::Vector3d bias_acc{Eigen::Vector3d::Zero()};
        Eigen::Matrix3d R_base_to_gyro{Eigen::Matrix3d::Identity()};
        Eigen::Matrix3d R_base_to_acc{Eigen::Matrix3d::Identity()};
        double joint_rate{};
        bool estimate_joint_velocity{};
        double joint_cutoff_frequency{};
        double joint_position_variance{};
        double angular_momentum_cutoff_frequency{};
        double tau_0{};
        double tau_1{};
        double force_torque_rate{};
        std::unordered_map<std::string, Eigen::Matrix3d> R_foot_to_force;
        std::unordered_map<std::string, Eigen::Matrix3d> R_foot_to_torque;
        bool estimate_contact_status{};
        double high_threshold{};
        double low_threshold{};
        int median_window{};
        int convergence_cycles{};
        Eigen::Vector3d angular_velocity_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d angular_velocity_bias_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d linear_acceleration_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d linear_acceleration_bias_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_position_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_orientation_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_position_slip_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_orientation_slip_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d com_position_process_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d com_linear_velocity_process_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d external_forces_process_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d com_position_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d com_linear_acceleration_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_base_position_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_base_orientation_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_base_linear_velocity_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_contact_position_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_contact_orientation_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_imu_linear_acceleration_bias_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_imu_angular_velocity_bias_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_com_position_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_com_linear_velocity_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d initial_external_forces_cov{Eigen::Vector3d::Zero()};
        double eps{0.1};
        Eigen::Isometry3d T_base_to_odom{Eigen::Isometry3d::Identity()};
        bool is_flat_terrain{};
        double terrain_height_covariance{};
    };

    Params params_;
    std::unordered_map<std::string, JointEstimator> joint_estimators_;
    std::unique_ptr<DerivativeEstimator> angular_momentum_derivative_estimator;
    std::unique_ptr<DerivativeEstimator> gyro_derivative_estimator;
    std::unordered_map<std::string, ContactDetector> contact_estimators_;
    State state_;
    ContactEKF base_estimator_;
    CoMEKF com_estimator_;
    std::unique_ptr<Mahony> attitude_estimator_;
    std::unique_ptr<RobotKinematics> kinematic_estimator_;
    std::unique_ptr<LegOdometry> leg_odometry_;
    bool is_initialized_{};
    size_t cycle_{};
    size_t imu_calibration_cycles_{};
    std::optional<TerrainMeasurement> terrain_ = std::nullopt;
};

}  // namespace serow
