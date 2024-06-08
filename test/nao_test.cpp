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
#include <gtest/gtest.h>

#include <chrono>
#include <iostream>
#include <map>
#include <serow/Serow.hpp>
#include <string>

TEST(SerowTests, NaoTest) {
    serow::Serow SEROW;
    EXPECT_TRUE(SEROW.initialize("nao.json"));

    Eigen::Vector3d g = Eigen::Vector3d(0, 0, -9.81);
    serow::ImuMeasurement imu;
    imu.timestamp = 0.01;
    imu.linear_acceleration = Eigen::Vector3d(0.1, -0.1, 0.05) - g;
    imu.angular_velocity = Eigen::Vector3d(-0.1, 0.1, 0.0);

    std::map<std::string, serow::JointMeasurement> joints;
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
    std::map<std::string, serow::ForceTorqueMeasurement> force_torque;
    force_torque.insert({"l_ankle", ft});
    force_torque.insert({"r_ankle", ft});

    auto t0 = std::chrono::high_resolution_clock::now();
    SEROW.filter(imu, joints, force_torque);
    auto t1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "SEROW filtering loop duration " << duration.count() << " us " << std::endl;
    t0 = std::chrono::high_resolution_clock::now();
    auto state = SEROW.getState(true);
    t1 = std::chrono::high_resolution_clock::now();
    EXPECT_TRUE(state.has_value());
    duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
    std::cout << "SEROW get state duration " << duration.count() << " us " << std::endl;

    EXPECT_FALSE(state->getBasePosition() != state->getBasePosition());
    EXPECT_FALSE(state->getBaseLinearVelocity() != state->getBaseLinearVelocity());
    EXPECT_FALSE(state->getBaseOrientation() != state->getBaseOrientation());
    for (const auto& cf : state->getContactsFrame()) {
        EXPECT_FALSE(*state->getContactPosition(cf) != *state->getContactPosition(cf));
        if (!state->isPointFeet()) {
            EXPECT_FALSE(*state->getContactOrientation(cf) != *state->getContactOrientation(cf));
        }
    }
    EXPECT_FALSE(state->getCoMPosition() != state->getCoMPosition());
    EXPECT_FALSE(state->getCoMLinearVelocity() != state->getCoMLinearVelocity());
    EXPECT_FALSE(state->getCoMExternalForces() != state->getCoMExternalForces());

    std::cout << "Base position " << state->getBasePosition().transpose() << std::endl;
    std::cout << "Base velocity " << state->getBaseLinearVelocity().transpose() << std::endl;
    std::cout << "Base orientation " << state->getBaseOrientation() << std::endl;
    std::cout << "Left contact position " << state->getContactPosition("l_ankle")->transpose()
              << std::endl;
    std::cout << "Right contact position " << state->getContactPosition("r_ankle")->transpose()
              << std::endl;
    if (!state->isPointFeet()) {
        std::cout << "Left contact orientation " << *state->getContactOrientation("l_ankle")
                  << std::endl;
        std::cout << "Right contact orientation " << *state->getContactOrientation("r_ankle")
                  << std::endl;
    }
    std::cout << "CoM position " << state->getCoMPosition().transpose() << std::endl;
    std::cout << "CoM linear velocity " << state->getCoMLinearVelocity().transpose() << std::endl;
    std::cout << "CoM external forces " << state->getCoMExternalForces().transpose() << std::endl;
}
