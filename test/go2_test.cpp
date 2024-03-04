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
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

#include <Serow.hpp>

using json = nlohmann::json;

TEST(SerowTests, Go2Test) {
    serow::Serow SEROW("../../config/go2.json");
    const double mass = 8.096;             // kg
    const double g = 9.81;                 // m/s^2
    const double mg = mass * g;            // N
    const double bias = 14;                // potatos
    const double den = 172.91 - 4 * bias;  // potatos

    std::fstream file("../../test/data/go2.csv", std::ios::in);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
        return;
    }
    // Read the CSV file
    std::vector<std::vector<std::string>> data;
    std::vector<std::string> row;
    std::string line, word;
    while (std::getline(file, line)) {
        row.clear();
        std::stringstream str(line);

        while (std::getline(str, word, ',')) {
            row.push_back(word);
        }
        data.push_back(row);
    }
    file.close();

    // Parse the data, create the measurements, and run the filtering loop
    for (size_t i = 1; i < data.size(); i++) {
        double timestamp = std::stod(data[i][0]) * 1e-9;  // s

        std::unordered_map<std::string, serow::ForceTorqueMeasurement> force_torque;
        Eigen::Vector3d force =
            Eigen::Vector3d(0, 0, std::clamp((std::stod(data[i][1]) - bias) * mg / den, 0.0, mg));
        force_torque.insert(
            {"FR_foot", serow::ForceTorqueMeasurement{.timestamp = timestamp, .force = force}});

        force =
            Eigen::Vector3d(0, 0, std::clamp((std::stod(data[i][2]) - bias) * mg / den, 0.0, mg));
        force_torque.insert(
            {"FL_foot", serow::ForceTorqueMeasurement{.timestamp = timestamp, .force = force}});

        force =
            Eigen::Vector3d(0, 0, std::clamp((std::stod(data[i][3]) - bias) * mg / den, 0.0, mg));
        force_torque.insert(
            {"RR_foot", serow::ForceTorqueMeasurement{.timestamp = timestamp, .force = force}});

        force =
            Eigen::Vector3d(0, 0, std::clamp((std::stod(data[i][4]) - bias) * mg / den, 0.0, mg));
        force_torque.insert(
            {"RL_foot", serow::ForceTorqueMeasurement{.timestamp = timestamp, .force = force}});

        serow::ImuMeasurement imu;
        imu.timestamp = timestamp;
        imu.linear_acceleration =
            Eigen::Vector3d(std::stod(data[i][5]), std::stod(data[i][6]), std::stod(data[i][7]));
        imu.angular_velocity =
            Eigen::Vector3d(std::stod(data[i][8]), std::stod(data[i][9]), std::stod(data[i][10]));

        std::unordered_map<std::string, serow::JointMeasurement> joints;
        joints.insert(
            {"FR_hip_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][11])}});
        joints.insert(
            {"FR_thigh_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][12])}});
        joints.insert(
            {"FR_calf_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][13])}});
        joints.insert(
            {"FL_hip_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][14])}});
        joints.insert(
            {"FL_thigh_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][15])}});
        joints.insert(
            {"FL_calf_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][16])}});
        joints.insert(
            {"RR_hip_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][17])}});
        joints.insert(
            {"RR_thigh_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][18])}});
        joints.insert(
            {"RR_calf_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][19])}});
        joints.insert(
            {"RL_hip_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][20])}});
        joints.insert(
            {"RL_thigh_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][21])}});
        joints.insert(
            {"RL_calf_joint",
             serow::JointMeasurement{.timestamp = timestamp, .position = std::stod(data[i][22])}});

        auto t0 = std::chrono::high_resolution_clock::now();
        SEROW.filter(imu, joints, force_torque);
        auto t1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        std::cout << "SEROW filtering loop duration " << duration.count() << " us " << std::endl;
        t0 = std::chrono::high_resolution_clock::now();
        auto state = SEROW.getState();
        t1 = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
        std::cout << "SEROW get state duration " << duration.count() << " us " << std::endl;
        if (!state.has_value()) {
            continue;
        }
        EXPECT_FALSE(state->getBasePosition() != state->getBasePosition());
        EXPECT_FALSE(state->getBaseLinearVelocity() != state->getBaseLinearVelocity());
        EXPECT_FALSE(state->getBaseOrientation() != state->getBaseOrientation());
        for (const auto& cf : state->getContactsFrame()) {
            if (state->getContactPosition(cf)) {
                EXPECT_FALSE(*state->getContactPosition(cf) != *state->getContactPosition(cf));
            }
        }
        EXPECT_FALSE(state->getCoMPosition() != state->getCoMPosition());
        EXPECT_FALSE(state->getCoMLinearVelocity() != state->getCoMLinearVelocity());
        EXPECT_FALSE(state->getCoMExternalForces() != state->getCoMExternalForces());

        std::cout << "Base position " << state->getBasePosition().transpose() << std::endl;
        std::cout << "Base velocity " << state->getBaseLinearVelocity().transpose() << std::endl;
        std::cout << "Base orientation " << state->getBaseOrientation() << std::endl;
        std::cout << "IMU gyro bias " << state->getImuAngularVelocityBias().transpose()
                  << std::endl;
        std::cout << "IMU acc bias " << state->getImuLinearAccelerationBias().transpose()
                  << std::endl;
        for (const auto& cf : state->getContactsFrame()) {
            if (!state->getContactPosition(cf)) {
                continue;
            }
            std::cout << cf << " contact position " << state->getContactPosition(cf)->transpose()
                      << std::endl;
        }
        std::cout << "CoM position " << state->getCoMPosition().transpose() << std::endl;
        std::cout << "CoM linear velocity " << state->getCoMLinearVelocity().transpose()
                  << std::endl;
        std::cout << "CoM external forces " << state->getCoMExternalForces().transpose()
                  << std::endl;
    }
}
