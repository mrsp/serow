/*
 * Copyright Stylianos Piperakis, Ownage Dynamics L.P.
 * License: GNU: https://www.gnu.org/licenses/gpl-3.0.html
 */
#include "Serow.hpp"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace serow {
    Serow::Serow(std::string config_file){
        auto config = json::parse(config_file);
        // Initialize the attitude estimator
        double params_.imu_rate = config.imu_rate;
        double Kp = config.attitude_estimator_proportional_gain;
        double Ki = config.attitude_estimator_integral_gain;
        attitude_estimator_ = std::make_unique<Mahony>(params_.imu_rate, Kp, Ki);

        // Initialize the kinematic estimator
        kinematic_estimator_ = std::make_unique<RobotKinematics>(config.model_path);
        params_.calibrate_imu = config.calibrate_imu;
        if(!params_.calibrate_imu) {
            params_.bias_acc << config.bias_acc[0], config.bias_acc[1], config.bias_acc[2];
            params_.bias_gyro << config.bias_gyro[0], config.bias_gyro[1], config.bias_gyro[2];
        }
    }

    void Serow::filter(ImuMeasurement imu, std::unordered_map<std::string, JointMeasurement> joints,
                       std::unordered_map<std::string, ForceTorqueMeasurement> ft,
                       std::optional<std::unordered_map<std::string, double>> contact_probabilities) {
        // Estimate the joint velocities
        std::unordered_map<std::string, double> joint_positions;
        std::unordered_map<std::string, double> joint_velocities;
        for (const auto& [key, value]: joints) {
            if (!joint_estimators_.count(key)) {
                joint_estimators_[key].init(key, joint_rate, joint_cuttoff_frequency);
            }
            joint_positions[key] = joints.at(key)
            joint_velocities[key] = joint_estimators_.at(key).filter(value);
        }

        // Estimate the base frame attitude
        Eigen::Vector3d wbb = params_.T_base_to_gyro.linear() * imu.angular_velocity;
        Eigen::Vector3d abb = params_.T_base_to_acc.linear() * imu.linear_acceleration;

        attitude_estimator_->filter(wbb, abb);
        const Eigen::Matrix3d& R_world_to_base = attitude_estimator_->getR();
        static imu_calibration_cycles = 0;
        if (params_.calibrate_imu && imu_calibration_cycles < params_.max_imu_calibration_cycles) {
            params_.bias_gyro += wbb;
            params_.bias_acc += abb - R_world_to_base.transpose() * Vector3d(0, 0, params_.g);
            imu_calibration_cycles++;
            return;
        } else if (params.calibrate_imu) {
            params_.bias_acc /= imu_calibration_cycles;
            params_.bias_gyro /= imu_calibration_cycles;
            params_.calibrate_imu = false;
            std::cout << "Calibration finished at " << imu_calibration_cycles << std::endl;
            std::cout << "Gyro biases " << params_.bias_gyro.transpose() << std::endl;
            std::cout << "Accelerometer biases " << params_.bias_acc.transpose() << std::endl;
        }
        
        // Update the Kinematic Structure
        kinematic_estimator_->updateJointConfig(joint_positions, joint_velocities, 0.1);

        // Get the CoM w.r.t the base frame
        Eigen::Vector3d base_to_com_position = kinematic_estimator_->comPosition();
        
        // Get the leg end-effector kinematics w.r.t the base frame 
        std::unordered_map<std::string, Eigen::Quaterniond> base_to_foot_orientations;
        std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_positions;
        std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_linear_velocities;
        std::unordered_map<std::string, Eigen::Vector3d> base_to_foot_angular_velocities;
        for (contact_frame : state_.contacts_frame) {
            base_to_foot_orientations[contact_frame] =
                kinematic_estimator_->linkOrientation(contact_frame);
            base_to_foot_positions[contact_frame] =
                kinematic_estimator_->linkPosition(contact_frame);
            base_to_foot_angular_ velocities[contact_frame] =
                kinematic_estimator_->getAngularVelocity(contact_frame);
            base_to_foot_linear_velocities[contact_frame] =
                kinematic_estimator_->getLinearVelocity(contact_frame);
        }   

        // Compute the leg odometry and the contact points
        if (!leg_odometry_) {
            leg_odometry_ = std::make_unique<LegOdometry>(
                base_to_foot_positions, base_to_foot_orientations, params_.mass, params_.tau0,
                params_.tau1, params_.joint_rate, params_.g);
        }
        leg_odometry_->estimate(
            attitude_estimator_->getQ(),
            attitude_estimator_->getGyro() - attitude_estimator_->getR() * params_.bias_g,
            base_to_foot_orientations, base_to_foot_positions, base_to_foot_linear_velocities,
            base_to_foot_orientations, contact_probabilities, contact_forces, contact_torques);
    }
}