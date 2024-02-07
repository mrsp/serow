#include <gtest/gtest.h>

#include <iostream>
#include <string>
#include <unordered_map>

#include "../src/Serow.hpp"

using json = nlohmann::json;

TEST(Operation, Development) {
    serow::Serow SERoW("../config/nao.json");

    Eigen::Vector3d g = Eigen::Vector3d(0, 0, -9.81);
    serow::ImuMeasurement imu;
    imu.timestamp = 0.01;
    imu.linear_acceleration = Eigen::Vector3d(0.1, -0.1, 0.05) - g;
    imu.angular_velocity = Eigen::Vector3d(-0.1, 0.1, 0.0);

    std::unordered_map<std::string, serow::JointMeasurement> joints;
    serow::JointMeasurement jm{.timestamp = 0.01, .position = 0.0};
    joints.insert({"HeadYaw", jm});
    joints.insert({"HeadPitch", jm});
    joints.insert({"LHipYawPitch", jm});
    joints.insert({"LHipRoll", jm});
    joints.insert({"LHipPitch", jm});
    joints.insert({"LKneePitch", jm});
    joints.insert({"LAnklePitch", jm});
    joints.insert({"LAnkleRoll", jm});
    joints.insert({"LShoulderPitch", jm});
    joints.insert({"LShoulderRoll", jm});
    joints.insert({"LElbowYaw", jm});
    joints.insert({"LElbowRoll", jm});
    joints.insert({"LWristYaw", jm});
    joints.insert({"LHand", jm});
    joints.insert({"RHipYawPitch", jm});
    joints.insert({"RHipRoll", jm});
    joints.insert({"RHipPitch", jm});
    joints.insert({"RKneePitch", jm});
    joints.insert({"RAnklePitch", jm});
    joints.insert({"RAnkleRoll", jm});
    joints.insert({"RShoulderPitch", jm});
    joints.insert({"RShoulderRoll", jm});
    joints.insert({"RElbowYaw", jm});
    joints.insert({"RElbowRoll", jm});
    joints.insert({"RWristYaw", jm});
    joints.insert({"RHand", jm});

    serow::ForceTorqueMeasurement ft{.timestamp = 0.01, .force = Eigen::Vector3d(0.0, 0.0, 40.0)};
    std::unordered_map<std::string, serow::ForceTorqueMeasurement> force_torque;
    force_torque.insert({"l_ankle", ft});
    force_torque.insert({"r_ankle", ft});

    SERoW.filter(imu, joints, force_torque);
    serow::State state = SERoW.getState();
    EXPECT_FALSE(state.base_position_ != state.base_position_);
    EXPECT_FALSE(state.base_linear_velocity_ != state.base_linear_velocity_);
    EXPECT_FALSE(state.base_orientation_ != state.base_orientation_);
    for (const auto& cf : state.getContactsFrame()) {
        EXPECT_FALSE(state.contacts_position_.at(cf) != state.contacts_position_.at(cf));
        if (!state.point_feet_) {
            EXPECT_FALSE(state.contacts_orientation_->at(cf) != state.contacts_orientation_->at(cf));
        }
    }
    EXPECT_FALSE(state.com_position_ != state.com_position_);
    EXPECT_FALSE(state.com_linear_velocity_ != state.com_linear_velocity_);
    EXPECT_FALSE(state.external_forces_ != state.external_forces_);

    std::cout << "Base position " << state.base_position_.transpose() << std::endl;
    std::cout << "Base velocity " << state.base_linear_velocity_.transpose() << std::endl;
    std::cout << "Base orientation " << state.base_orientation_ << std::endl;
    std::cout << "Left contact position " << state.contacts_position_.at("l_ankle").transpose()
              << std::endl;
    std::cout << "Right contact position " << state.contacts_position_.at("r_ankle").transpose()
              << std::endl;
    if (!state.point_feet_) {
        std::cout << "Left contact orientation " << state.contacts_orientation_->at("l_ankle")
                  << std::endl;
        std::cout << "Right contact orientation " << state.contacts_orientation_->at("r_ankle")
                  << std::endl;
    }
    std::cout << "CoM position " << state.com_position_.transpose() << std::endl;
    std::cout << "CoM linear velocity " << state.com_linear_velocity_.transpose() << std::endl;
    std::cout << "CoM external forces " << state.external_forces_.transpose() << std::endl;
}
