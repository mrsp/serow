#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <string>
#include <vector>

#include <mcap/reader.hpp>
#include <mcap/writer.hpp>

#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "serow/Serow.hpp"

using namespace serow;
using json = nlohmann::json;

static constexpr bool USE_JOINT_EFFORT_CONTACT_WRENCH = false;

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
    if (part == "Hip")
        serowPart = "hip";
    else if (part == "Thigh")
        serowPart = "thigh";
    else if (part == "Calf")
        serowPart = "calf";
    else
        return "";

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
        std::cout << "Contact wrench mode: "
                  << (USE_JOINT_EFFORT_CONTACT_WRENCH
                          ? "joint efforts -> SEROW contact wrench estimator"
                          : "feet_forces from MCAP")
                  << std::endl;

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
            if (channelPtr->topic != "/robot_state")
                continue;

            // Deserialize Input JSON
            std::string payload(reinterpret_cast<const char*>(msg.data), msg.dataSize);
            json j_in = json::parse(payload);
            double timestamp = j_in["timestamp"];

            // Parse IMU
            serow::ImuMeasurement imu;
            imu.timestamp = timestamp;
            imu.linear_acceleration = Eigen::Vector3d(j_in["imu"]["linear_acceleration"]["x"],
                                                      j_in["imu"]["linear_acceleration"]["y"],
                                                      j_in["imu"]["linear_acceleration"]["z"]);
            imu.angular_velocity = Eigen::Vector3d(j_in["imu"]["angular_velocity"]["x"],
                                                   j_in["imu"]["angular_velocity"]["y"],
                                                   j_in["imu"]["angular_velocity"]["z"]);

            // Read the measured contact forces exactly as stored in the MCAP.
            // These vectors are expressed in each local foot/sensor frame and are
            // retained unchanged for comparison with the GMO estimate.
            const std::vector<std::string> legs = {"FL", "FR", "RL", "RR"};
            std::map<std::string, Eigen::Vector3d> measured_forces_local;

            if (!j_in.contains("feet_forces")) {
                throw std::runtime_error("Input message has no feet_forces field.");
            }

            for (const auto& leg : legs) {
                measured_forces_local[leg] =
                    Eigen::Vector3d(j_in["feet_forces"][leg]["x"], j_in["feet_forces"][leg]["y"],
                                    j_in["feet_forces"][leg]["z"]);
            }

            // When false, the measured forces are supplied to SEROW. When true,
            // force_torque remains nullopt and SEROW estimates the contact wrench
            // from joint efforts using the generalized momentum observer (GMO).
            std::optional<std::map<std::string, serow::ForceTorqueMeasurement>> force_torque =
                std::nullopt;

            if (!USE_JOINT_EFFORT_CONTACT_WRENCH) {
                std::map<std::string, serow::ForceTorqueMeasurement> ft_map;
                for (const auto& leg : legs) {
                    serow::ForceTorqueMeasurement ft;
                    ft.timestamp = timestamp;
                    ft.force = measured_forces_local.at(leg);
                    ft.torque = Eigen::Vector3d::Zero();
                    ft_map[leg + "_foot"] = ft;
                }
                force_torque = ft_map;
            }

            //  Parse Joints
            std::map<std::string, serow::JointMeasurement> joints;
            json j_joints = j_in["joint_states"];

            static bool warned_missing_joint_effort = false;

            for (auto& [key, val] : j_joints.items()) {
                std::string serow_name = mapJointName(key);
                if (!serow_name.empty()) {
                    serow::JointMeasurement jm;
                    jm.timestamp = timestamp;
                    jm.position = val["position"];
                    jm.velocity = val["velocity"];

                    if (val.contains("effort")) {
                        jm.effort = val["effort"];
                    } else if (val.contains("torque")) {
                        jm.effort = val["torque"];
                    } else if (val.contains("joint_effort")) {
                        jm.effort = val["joint_effort"];
                    } else {
                        jm.effort = 0.0;

                        if (USE_JOINT_EFFORT_CONTACT_WRENCH && !warned_missing_joint_effort) {
                            std::cerr
                                << "Warning: USE_JOINT_EFFORT_CONTACT_WRENCH=true, but at least "
                                << "one joint_state entry has no effort/torque/joint_effort field. "
                                << "Missing efforts are set to zero, so contact wrench estimation "
                                << "will not be meaningful unless the MCAP contains efforts.\n";
                            warned_missing_joint_effort = true;
                        }
                    }

                    joints[serow_name] = jm;
                }
            }

            // Run Filter
            std::optional<serow::BasePoseGroundTruth> base_pose_gt = std::nullopt;

            if (j_in.contains("base_ground_truth")) {
                serow::BasePoseGroundTruth gt;
                gt.timestamp = timestamp;
                gt.position = Eigen::Vector3d(j_in["base_ground_truth"]["position"]["x"],
                                              j_in["base_ground_truth"]["position"]["y"],
                                              j_in["base_ground_truth"]["position"]["z"]);
                gt.orientation = Eigen::Quaterniond(j_in["base_ground_truth"]["orientation"]["w"],
                                                    j_in["base_ground_truth"]["orientation"]["x"],
                                                    j_in["base_ground_truth"]["orientation"]["y"],
                                                    j_in["base_ground_truth"]["orientation"]["z"]);

                base_pose_gt = gt;
            }

            const bool ok = estimator.filter(
                imu, joints,
                force_torque,   // nullopt when using joint-effort contact wrench estimator
                std::nullopt,   // no external odometry
                std::nullopt,   // no external contact probabilities
                base_pose_gt);  // optional GT if present in MCAP

            if (!ok) {
                continue;
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

                json j_out;
                j_out["timestamp"] = timestamp;

                // Base Pose
                j_out["base_pose"]["position"] = {
                    {"x", basePos.x()}, {"y", basePos.y()}, {"z", basePos.z()}};
                j_out["base_pose"]["rotation"] = {{"w", baseOrient.w()},
                                                  {"x", baseOrient.x()},
                                                  {"y", baseOrient.y()},
                                                  {"z", baseOrient.z()}};
                j_out["base_pose"]["linear_velocity"] = {
                    {"x", baseLinVel.x()}, {"y", baseLinVel.y()}, {"z", baseLinVel.z()}};

                // CoM State
                j_out["CoM_state"]["position"] = {
                    {"x", comPos.x()}, {"y", comPos.y()}, {"z", comPos.z()}};
                j_out["CoM_state"]["velocity"] = {
                    {"x", comVel.x()}, {"y", comVel.y()}, {"z", comVel.z()}};
                j_out["CoM_state"]["externalForces"] = {
                    {"x", extForce.x()}, {"y", extForce.y()}, {"z", extForce.z()}};

                // IMU Biases
                j_out["imu_bias"]["accel"] = {
                    {"x", biasAcc.x()}, {"y", biasAcc.y()}, {"z", biasAcc.z()}};
                j_out["imu_bias"]["angVel"] = {
                    {"x", biasGyr.x()}, {"y", biasGyr.y()}, {"z", biasGyr.z()}};

                // Contact-force comparison in each local foot frame.
                // State::getContactForce() returns f_W and getFootOrientation()
                // returns q_WF. Therefore f_F = R_WF^T f_W = q_WF^{-1} * f_W.
                for (const auto& leg : legs) {
                    const std::string frame_name = leg + "_foot";
                    const Eigen::Vector3d& measured_force_local = measured_forces_local.at(leg);

                    const Eigen::Quaterniond q_world_foot =
                        state->getFootOrientation(frame_name).normalized();
                    const auto estimated_force_world_opt = state->getContactForce(frame_name);
                    const bool force_available = estimated_force_world_opt.has_value();

                    const Eigen::Vector3d estimated_force_world =
                        estimated_force_world_opt.value_or(Eigen::Vector3d::Zero());
                    const Eigen::Vector3d estimated_force_local =
                        q_world_foot.conjugate() * estimated_force_world;

                    j_out["measured_contact_forces_local"][frame_name] = {
                        {"x", measured_force_local.x()},
                        {"y", measured_force_local.y()},
                        {"z", measured_force_local.z()}};

                    j_out["estimated_contact_forces_local"][frame_name] = {
                        {"x", estimated_force_local.x()},
                        {"y", estimated_force_local.y()},
                        {"z", estimated_force_local.z()}};

                    // Keep the original GMO world-frame output and the exact
                    // orientation used for the transformation as diagnostics.
                    j_out["estimated_contact_forces_world"][frame_name] = {
                        {"x", estimated_force_world.x()},
                        {"y", estimated_force_world.y()},
                        {"z", estimated_force_world.z()}};
                    j_out["foot_orientation_world"][frame_name] = {{"w", q_world_foot.w()},
                                                                   {"x", q_world_foot.x()},
                                                                   {"y", q_world_foot.y()},
                                                                   {"z", q_world_foot.z()}};
                    j_out["estimated_contact_force_available"][frame_name] = force_available;
                }

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
                    std::cerr << "Warning: Failed to write message: " << writeStatus.message
                              << std::endl;
                }
            }
        }

        writer.close();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Done. " << message_count << " frames in " << duration.count() << " us."
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
