/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

#include "CoMEKF.hpp"
#include "ContactEKF.hpp"
#include "ContactDetector.hpp"
#include "JointEstimator.hpp"
#include "LegOdometry.hpp"
#include "Mahony.hpp"
#include "RobotKinematics.hpp"
#include "State.hpp"

namespace serow {

struct JointMeasurement {
    double timestamp{};
    double position{};
    std::optional<double> velocity{};
};

class Serow {
   public:
    Serow(std::string config);
    void filter(ImuMeasurement imu, std::unordered_map<std::string, JointMeasurement> joints,
                std::optional<std::unordered_map<std::string, ForceTorqueMeasurement>> ft,
                std::optional<std::unordered_map<std::string, double>> contact_probabilities);

   private:
    struct Params {
        double mass{};
        double g{};
        bool calibrate_imu{};
        double max_imu_calibration_cycles{};
        double imu_rate{};
        Eigen::Vector3d bias_gyro{Eigen::Vector3d::Zero()};
        Eigen::Vector3d bias_acc{Eigen::Vector3d::Zero()};
        Eigen::Matrix3d R_base_to_gyro{Eigen::Matrix3d::Identity()};
        Eigen::Matrix3d R_base_to_acc{Eigen::Matrix3d::Identity()};
        double joint_rate{};
        bool estimate_joint_velocity{};
        double joint_cutoff_frequency{};
        double joint_position_variance{};
        double tau_0{};
        double tau_1{};
        bool estimate_contact{};
        std::unordered_map<std::string, Eigen::Matrix3d> R_base_to_force;
        std::unordered_map<std::string, Eigen::Matrix3d> R_base_to_torque;
        bool estimate_contact_status{};
        double high_threshold{};
        double low_threshold{};
        int median_window{};
        Eigen::Vector3d angular_velocity_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d angular_velocity_bias_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d linear_acceleration_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d linear_acceleration_bias_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_position_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_orientation_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_position_slip_cov{Eigen::Vector3d::Zero()};
        Eigen::Vector3d contact_orientation_slip_cov{Eigen::Vector3d::Zero()};
        double eps = 0.1;
    };

    Params params_;
    std::unordered_map<std::string, JointEstimator> joint_estimators_;
    std::unordered_map<std::string, ContactDetector> contact_estimators_;
    State state_;
    ContactEKF base_estimator_;
    CoMEKF com_estimator_;
    std::unique_ptr<Mahony> attitude_estimator_;
    std::unique_ptr<RobotKinematics> kinematic_estimator_;
    std::unique_ptr<LegOdometry> leg_odometry_;
};

}  // namespace serow
