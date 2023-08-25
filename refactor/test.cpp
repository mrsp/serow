#include <iostream>
#include <unordered_map>
#include <string>
#include "ContactEKF.hpp"
#include "State.hpp"
#include <gtest/gtest.h>

TEST(Operation, Predict) {
    std::unordered_set<std::string> contacts_frame;
    contacts_frame.insert({"left_foot"});
    State state(contacts_frame, true);

    state.contacts_position_.insert({"left_foot", Eigen::Vector3d(0.1, 0.5, -0.5)});
    state.contacts_status_.insert({"left_foot", true});

    ContactEKF base_ekf;
    Eigen::Vector3d g = Eigen::Vector3d(0, 0, -9.81);
    base_ekf.init(state);

    ImuMeasurement imu;
    imu.timestamp = 0.1;
    imu.linear_acceleration = Eigen::Vector3d(0.1, -0.1, 0.05) - g;
    imu.angular_velocity = Eigen::Vector3d(-0.1, 0.1, 0.0);

    KinematicMeasurement kin;
    kin.timestamp = 0.1;
    kin.contacts_position.insert({"left_foot", Eigen::Vector3d(0.1, 0.5, -0.5)});
    kin.contacts_status.insert({"left_foot", true});
    kin.contacts_probability.insert({"left_foot", 1.0});
    kin.contacts_position_noise.insert({"left_foot", Eigen::Matrix3d::Identity() * 1e-6});

    State predicted_state = base_ekf.predict(state, imu, kin);

    std::cout << "Base position after predict " << predicted_state.base_position_.transpose()
              << std::endl;
    std::cout << "Base velocity after predict " << predicted_state.base_linear_velocity_.transpose()
              << std::endl;
    std::cout << "Base orientation after predict " << predicted_state.base_orientation_<<std::endl;
    std::cout << "Left contact position after predict "
              << predicted_state.contacts_position_.at("left_foot").transpose() << std::endl;

    State updated_state = base_ekf.update(predicted_state, kin);
    std::cout << "Left contact position after update "
              << updated_state.contacts_position_.at("left_foot").transpose() << std::endl;
}
