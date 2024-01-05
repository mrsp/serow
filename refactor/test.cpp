#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <unordered_map>

#include "Serow.hpp"

using json = nlohmann::json;

TEST(Operation, Development) {
    std::unordered_set<std::string> contacts_frame;

    std::ifstream f("../config/test.json");
    json data = json::parse(f);
    for(int i = 0; i < data["foot_frames"].size(); i++){
        contacts_frame.insert({data["foot_frames"][std::to_string(i)]});
    }
    std::string model_path = data["model_path"];
    bool point_feet = data["point_feet"];

    State state(contacts_frame, point_feet);
    serow::RobotKinematics kinematics(model_path);

    state.contacts_position_.insert({"left_foot", Eigen::Vector3d(0.1, 0.5, -0.5)});
    std::unordered_map<std::string, Eigen::Quaterniond> lo{
        {"left_foot", Eigen::Quaterniond::Identity()}};
    std::unordered_map<std::string, Eigen::Matrix3d> lo_cov{
        {"left_foot", Eigen::Matrix3d::Identity() * 1e-6}};

    state.contacts_orientation_.emplace(lo);
    state.contacts_status_.insert({"left_foot", true});

    ContactEKF base_ekf;
    Eigen::Vector3d g = Eigen::Vector3d(0, 0, -9.81);
    double imu_rate = 1000;
    base_ekf.init(state, imu_rate);

    ImuMeasurement imu;
    imu.timestamp = 0.01;
    imu.linear_acceleration = Eigen::Vector3d(0.1, -0.1, 0.05) - g;
    imu.angular_velocity = Eigen::Vector3d(-0.1, 0.1, 0.0);
    imu.angular_velocity_cov = Eigen::Matrix3d::Identity() * 1e-4;
    imu.linear_acceleration_cov = Eigen::Matrix3d::Identity() * 1e-3;
    imu.angular_velocity_bias_cov = Eigen::Matrix3d::Identity() * 1e-5;
    imu.linear_acceleration_bias_cov = Eigen::Matrix3d::Identity() * 5e-4;

    KinematicMeasurement kin;
    kin.timestamp = 0.1;
    kin.contacts_position.insert({"left_foot", Eigen::Vector3d(0.1, 0.5, -0.5)});
    kin.contacts_orientation.emplace(lo);
    kin.contacts_status.insert({"left_foot", true});
    kin.contacts_probability.insert({"left_foot", 1.0});
    kin.contacts_position_noise.insert({"left_foot", Eigen::Matrix3d::Identity() * 1e-6});
    kin.contacts_orientation_noise.emplace(lo_cov);
    kin.position_cov = Eigen::Matrix3d::Identity() * 1e-6;
    kin.orientation_cov = Eigen::Matrix3d::Identity() * 1e-6;
    kin.position_slip_cov = Eigen::Matrix3d::Identity() * 1e-6;
    kin.orientation_slip_cov = Eigen::Matrix3d::Identity() * 1e-3;

    State predicted_state = base_ekf.predict(state, imu, kin);
    std::cout << "Base position after predict " << predicted_state.base_position_.transpose()
              << std::endl;
    std::cout << "Base velocity after predict " << predicted_state.base_linear_velocity_.transpose()
              << std::endl;
    std::cout << "Base orientation after predict " << predicted_state.base_orientation_
              << std::endl;
    std::cout << "Left contact position after predict "
              << predicted_state.contacts_position_.at("left_foot").transpose() << std::endl;
    std::cout << "Left contact orientation after predict "
              << predicted_state.contacts_orientation_->at("left_foot") << std::endl;
    State updated_state = base_ekf.update(predicted_state, kin);
    std::cout << "Left contact position after update "
              << updated_state.contacts_position_.at("left_foot").transpose() << std::endl;
    std::cout << "Left contact orientation after update "
              << updated_state.contacts_orientation_->at("left_foot") << std::endl;

}
