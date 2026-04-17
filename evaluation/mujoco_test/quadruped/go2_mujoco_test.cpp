#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <regex>
#include <chrono>

#include <mcap/reader.hpp>
#include <mcap/writer.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "serow/Serow.hpp"

using namespace serow;
using json = nlohmann::json;

// Helper: Resolve Paths
std::string resolvePath(const json& config, const std::string& path) {
    const char* env_p = std::getenv("SEROW_PATH");
    std::string serow_path_env = (env_p) ? env_p : ""; 
    
    std::string resolved_path = serow_path_env + path;
    std::string experiment_type = config["Experiment"]["type"];
    std::string base_path = config["Paths"]["base_path"];

    resolved_path = std::regex_replace(resolved_path, std::regex("\\{base_path\\}"), base_path);
    resolved_path = std::regex_replace(resolved_path, std::regex("\\{type\\}"), experiment_type);

    return resolved_path;
}

//  Map Joint Names
std::string mapJointName(const std::string& pyName) {
    std::string leg = pyName.substr(0, 2); 
    std::string part = pyName.substr(3);    

    std::string serowPart;
    if (part == "Hip") serowPart = "hip";
    else if (part == "Thigh") serowPart = "thigh";
    else if (part == "Calf") serowPart = "calf";
    else return "";

    return leg + "_" + serowPart + "_joint";
}

int main(int argc, char** argv) {
    try {
        std::string config_path = "go2.json"; 

        if (argc > 1) {
            config_path = argv[1];
            std::cout << "Using config: " << config_path << std::endl;
        }

        // Initialize Serow
        serow::Serow estimator;
        if (!estimator.initialize(config_path)) {
            throw std::runtime_error("Failed to initialize Serow.");
        }

        // Load Config
        std::ifstream config_file("../test_config.json");
        if (!config_file.is_open()) {
            std::cerr << "Failed to open test_config.json" << std::endl;
            return 1;
        }
        json config;
        config_file >> config;
        
        const std::string INPUT_FILE = resolvePath(config, config["Paths"]["data_file"]);
        const std::string OUTPUT_FILE = resolvePath(config, config["Paths"]["prediction_file"]);
        
        std::cout << "Input: " << INPUT_FILE << "\nOutput: " << OUTPUT_FILE << std::endl;

        // ---------------------------------------------------------
        // SETUP MCAP WRITER
        // ---------------------------------------------------------
        mcap::McapWriter writer;
        mcap::McapWriterOptions writerOptions("serow_estimator");
        auto status = writer.open(OUTPUT_FILE, writerOptions);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open output MCAP: " + status.message);
        }

        mcap::Schema outputSchema("SerowState", "jsonschema", "");
        writer.addSchema(outputSchema);

        mcap::Channel outputChannel("serow_predictions", "json", outputSchema.id);
        writer.addChannel(outputChannel);

        // ---------------------------------------------------------
        // SETUP MCAP READER
        // ---------------------------------------------------------
        mcap::McapReader reader;
        status = reader.open(INPUT_FILE);
        if (!status.ok()) {
            throw std::runtime_error("Failed to open input MCAP: " + status.message);
        }

        // ---------------------------------------------------------
        // PROCESSING LOOP
        // ---------------------------------------------------------
        size_t message_count = 0;
        auto start_time = std::chrono::high_resolution_clock::now();

        auto messages = reader.readMessages();
        

        for (const auto& msgView : messages) {
            const mcap::Message& msg = msgView.message;
            
            auto channelPtr = reader.channel(msg.channelId);
            if(channelPtr->topic != "/robot_state") continue;

            // Deserialize Input JSON
            std::string payload(reinterpret_cast<const char*>(msg.data), msg.dataSize);
            json j_in = json::parse(payload);
            double timestamp = j_in["timestamp"];

            // Parse IMU 
            serow::ImuMeasurement imu;
            imu.timestamp = timestamp;
            imu.linear_acceleration = Eigen::Vector3d(
                j_in["imu"]["linear_acceleration"]["x"],
                j_in["imu"]["linear_acceleration"]["y"],
                j_in["imu"]["linear_acceleration"]["z"]
            );
            imu.angular_velocity = Eigen::Vector3d(
                j_in["imu"]["angular_velocity"]["x"],
                j_in["imu"]["angular_velocity"]["y"],
                j_in["imu"]["angular_velocity"]["z"]
            );

            // Parse Forces
            std::map<std::string, serow::ForceTorqueMeasurement> force_torque;
            std::vector<std::string> legs = {"FL", "FR", "RL", "RR"};
            for(const auto& leg : legs) {
                force_torque.insert({ leg + "_foot", serow::ForceTorqueMeasurement{
                    .timestamp = timestamp,
                    .force = Eigen::Vector3d(
                        j_in["feet_forces"][leg]["x"],
                        j_in["feet_forces"][leg]["y"],
                        j_in["feet_forces"][leg]["z"]
                    )
                }});
            }

            //  Parse Joints 
            std::map<std::string, serow::JointMeasurement> joints;
            json j_joints = j_in["joint_states"];
            for (auto& [key, val] : j_joints.items()) {
                std::string serow_name = mapJointName(key);
                if(!serow_name.empty()) {
                    joints.insert({serow_name, serow::JointMeasurement{
                        .timestamp = timestamp,
                        .position = val["position"],
                        .velocity = val["velocity"]
                    }});
                }
            }

            // Run Filter
            if(j_in.contains("base_ground_truth")) {
                BasePoseGroundTruth gt;
                gt.timestamp = timestamp;
                gt.position = Eigen::Vector3d(
                    j_in["base_ground_truth"]["position"]["x"],
                    j_in["base_ground_truth"]["position"]["y"],
                    j_in["base_ground_truth"]["position"]["z"]
                );
                gt.orientation = Eigen::Quaterniond(
                    j_in["base_ground_truth"]["orientation"]["w"],
                    j_in["base_ground_truth"]["orientation"]["x"],
                    j_in["base_ground_truth"]["orientation"]["y"],
                    j_in["base_ground_truth"]["orientation"]["z"]
                );
                estimator.filter(imu, joints, force_torque, std::nullopt, std::nullopt, gt);
            } else {
                estimator.filter(imu, joints, force_torque);
            }

            //  Write Output 
            auto state = estimator.getState(true);
            if (state.has_value()) {
                auto basePos = state->getBasePosition();
                auto baseOrient = state->getBaseOrientation();
                auto baseLinVel = state->getBaseLinearVelocity();
                
                // Get CoM Data
                auto comPos = state->getCoMPosition();
                auto comVel = state->getCoMLinearVelocity();
                auto extForce = state->getCoMExternalForces();

                // Get Bias Data
                auto biasAcc = state->getImuLinearAccelerationBias();
                auto biasGyr = state->getImuAngularVelocityBias();

                // Get Contact Positions
                auto get_contact = [&](const std::string& leg) -> Eigen::Vector3d {
                   auto val = state->getContactPosition(leg + "_foot");
                   return val.has_value() ? val.value() : Eigen::Vector3d::Zero();
                };

                json j_out;
                j_out["timestamp"] = timestamp;

                // Base Pose
                j_out["base_pose"]["position"] = { {"x", basePos.x()}, {"y", basePos.y()}, {"z", basePos.z()} };
                j_out["base_pose"]["rotation"] = { {"w", baseOrient.w()}, {"x", baseOrient.x()}, {"y", baseOrient.y()}, {"z", baseOrient.z()} };
                j_out["base_pose"]["linear_velocity"] = { {"x", baseLinVel.x()}, {"y", baseLinVel.y()}, {"z", baseLinVel.z()} };
                
                // CoM State
                j_out["CoM_state"]["position"] = { {"x", comPos.x()}, {"y", comPos.y()}, {"z", comPos.z()} };
                j_out["CoM_state"]["velocity"] = { {"x", comVel.x()}, {"y", comVel.y()}, {"z", comVel.z()} };
                j_out["CoM_state"]["externalForces"] = { {"x", extForce.x()}, {"y", extForce.y()}, {"z", extForce.z()} };

                // IMU Biases
                j_out["imu_bias"]["accel"] = { {"x", biasAcc.x()}, {"y", biasAcc.y()}, {"z", biasAcc.z()} };
                j_out["imu_bias"]["angVel"] = { {"x", biasGyr.x()}, {"y", biasGyr.y()}, {"z", biasGyr.z()} };

                // Contact Positions
                j_out["contact_positions"]["FL_foot"] = { {"x", get_contact("FL").x()}, {"y", get_contact("FL").y()}, {"z", get_contact("FL").z()} };
                j_out["contact_positions"]["FR_foot"] = { {"x", get_contact("FR").x()}, {"y", get_contact("FR").y()}, {"z", get_contact("FR").z()} };
                j_out["contact_positions"]["RL_foot"] = { {"x", get_contact("RL").x()}, {"y", get_contact("RL").y()}, {"z", get_contact("RL").z()} };
                j_out["contact_positions"]["RR_foot"] = { {"x", get_contact("RR").x()}, {"y", get_contact("RR").y()}, {"z", get_contact("RR").z()} };

                std::string output_payload = j_out.dump();

                mcap::Message outMsg;
                outMsg.channelId = outputChannel.id;
                outMsg.sequence = message_count++;
                outMsg.logTime = msg.logTime;
                outMsg.publishTime = msg.publishTime;
                outMsg.data = reinterpret_cast<const std::byte*>(output_payload.data());
                outMsg.dataSize = output_payload.size();

                auto writeStatus = writer.write(outMsg);
                if (!writeStatus.ok()) {
                    std::cerr << "Warning: Failed to write message: " << writeStatus.message << std::endl;
                }
            }
        }

        writer.close();
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Done. " << message_count << " frames in " << duration.count() << " us." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
