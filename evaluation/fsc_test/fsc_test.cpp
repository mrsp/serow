#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <regex>
#include <cstdlib>
#include <stdexcept>
#include <optional>
#include <vector>
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>
#include "serow/Serow.hpp"

using json = nlohmann::json;

static Eigen::Vector3d vec3FromJson(const json& j) {
    if (j.contains("x") && j.contains("y") && j.contains("z"))
        return { j["x"].get<double>(), j["y"].get<double>(), j["z"].get<double>() };
    return Eigen::Vector3d::Zero();
}

static Eigen::Quaterniond quatFromJson(const json& j) {
    return Eigen::Quaterniond(j.value("w", 1.0), j.value("x", 0.0), j.value("y", 0.0), j.value("z", 0.0)).normalized();
}

static json vec3ToJson(const Eigen::Vector3d& v) {
    return { {"x", v.x()}, {"y", v.y()}, {"z", v.z()} };
}

static json quatToJson(const Eigen::Quaterniond& q) {
    return { {"w", q.w()}, {"x", q.x()}, {"y", q.y()}, {"z", q.z()} };
}

struct FrameData {
    double timestamp{};
    serow::ImuMeasurement imu;
    std::map<std::string, serow::JointMeasurement> joints;
    std::map<std::string, serow::ForceTorqueMeasurement> forces;
    std::optional<serow::BasePoseGroundTruth> ground_truth;
};

static FrameData parseFrame(const json& j, const std::map<std::string, std::string>& contact_map) {
    FrameData f;
    f.timestamp = j.value("timestamp", 0.0);

    // IMU
    f.imu.timestamp = f.timestamp;
    if (j.contains("imu")) {
        f.imu.linear_acceleration = vec3FromJson(j["imu"]["linear_acceleration"]);
        f.imu.angular_velocity    = vec3FromJson(j["imu"]["angular_velocity"]);
    }

    // Joints
    if (j.contains("joint_states")) {
        for (const auto& [name, data] : j["joint_states"].items()) {
            serow::JointMeasurement jm;
            jm.timestamp = f.timestamp;
            jm.position  = data.value("position", 0.0);
            jm.velocity  = data.value("velocity", 0.0);

            f.joints[name] = jm;   
        }
    }
    // Forces & Torques
    if (j.contains("feet_forces")) {
        for (const auto& [json_key, serow_link_name] : contact_map) {
            if (j["feet_forces"].contains(json_key)) {
                serow::ForceTorqueMeasurement ft;
                ft.timestamp = f.timestamp;
                ft.force     = vec3FromJson(j["feet_forces"][json_key]);
                
                // Check if torques exist for this specific foot, otherwise set to zero
                if (j.contains("feet_torques") && j["feet_torques"].contains(json_key)) {
                    ft.torque = vec3FromJson(j["feet_torques"][json_key]);
                } else {
                    ft.torque = Eigen::Vector3d::Zero();
                }
                
                f.forces[serow_link_name] = ft;
            }
        }
    }

    // Ground Truth
    if (j.contains("base_ground_truth")) {
        const auto& gt_j = j["base_ground_truth"];
        serow::BasePoseGroundTruth gt;
        gt.timestamp   = f.timestamp;
        gt.position    = vec3FromJson(gt_j["position"]);
        gt.orientation = quatFromJson(gt_j["orientation"]);
        f.ground_truth = gt;
    }


    return f;
}

static void requireOk(const mcap::Status& s, const std::string& context) {
    if (!s.ok()) throw std::runtime_error(context + ": " + s.message);
}   

static mcap::Timestamp toNs(double seconds) {
    return (seconds > 0.0) ? static_cast<mcap::Timestamp>(seconds * 1e9) : 0;
}

int main(int argc, char** argv) {
    try {
        std::string config_path = "../test_config.json";
        double percentage = 100.0;

        if (argc > 1) {
            std::string arg1 = argv[1];
            if (arg1.find(".json") != std::string::npos) {
                config_path = arg1;
                if (argc > 2) percentage = std::stod(argv[2]);
            } else {
                percentage = std::stod(arg1); 
            }
        }

        // Clamp percentage between 0 and 100
        percentage = std::max(0.0, std::min(100.0, percentage));

        std::ifstream cfg_file(config_path);
        if (!cfg_file.is_open()) throw std::runtime_error("Cannot open config: " + config_path);
        json cfg; cfg_file >> cfg;

        std::string serow_path_env = "";
        if (const char* env_p = std::getenv("SEROW_PATH")) {
            serow_path_env = env_p;
            if (!serow_path_env.empty() && serow_path_env.back() != '/') serow_path_env += "/";
        }

        std::string robot_name = cfg["Target"].value("robot", "anymal_b");
        std::string exp_name   = cfg["Target"].value("experiment", "fsc");
        std::string base_path  = cfg["Paths"].value("base_path", ".");

        std::string robot_config  = robot_name + ".json"; 
        std::string input_topic   = "/robot_state"; 
        std::string full_exp_name = robot_name + "_" + exp_name; 

        auto resolve = [&](std::string s) {
            s = serow_path_env + s;
            s = std::regex_replace(s, std::regex("\\{base_path\\}"), base_path);
            s = std::regex_replace(s, std::regex("\\{robot\\}"), robot_name);
            s = std::regex_replace(s, std::regex("\\{experiment\\}"), exp_name);
            s = std::regex_replace(s, std::regex("\\{full_exp\\}"), full_exp_name);
            return std::regex_replace(s, std::regex("/{2,}"), "/");
        };

        const std::string input_mcap  = resolve(cfg["Paths"]["data_file"]);
        const std::string output_mcap = resolve(cfg["Paths"]["prediction_file"]);

        if (!cfg["RobotDatabase"].contains(robot_name)) {
            throw std::runtime_error("Robot '" + robot_name + "' not found in RobotDatabase!");
        }
        std::map<std::string, std::string> contact_map = 
            cfg["RobotDatabase"][robot_name]["contact_map"].get<std::map<std::string, std::string>>();

        std::cout << "[CONFIG] Input:  " << input_mcap << "\n"
                  << "[CONFIG] Output: " << output_mcap << "\n";

        serow::Serow estimator;
        if (!estimator.initialize(robot_config)) throw std::runtime_error("Serow init failed");
        
        mcap::McapWriter writer;
        mcap::McapWriterOptions writerOptions("serow_estimator");
        requireOk(writer.open(output_mcap, writerOptions), "Open output");

        mcap::Channel channel("serow_predictions", "json", 0);
        writer.addChannel(channel);

        mcap::McapReader reader;
        requireOk(reader.open(input_mcap), "Open input");

        size_t total_topic_frames = 0;
        auto stats = reader.statistics();
        if (stats.has_value()) {
            for (const auto& [channel_id, count] : stats->channelMessageCounts) {
                auto ch = reader.channel(channel_id);
                if (ch && ch->topic == input_topic) {
                    total_topic_frames += count;
                }
            }
        } else {
            for (const auto& view : reader.readMessages()) {
                auto ch = reader.channel(view.message.channelId);
                if (ch && ch->topic == input_topic) total_topic_frames++;
            }
        }

        size_t max_frames = static_cast<size_t>(total_topic_frames * (percentage / 100.0));
        if (max_frames == 0) max_frames = total_topic_frames; // Safety fallback

        std::cout << "[INFO] Dataset contains " << total_topic_frames << " frames on " << input_topic << ".\n";
        std::cout << "[INFO] Processing " << percentage << "% -> " << max_frames << " frames max.\n";
        // ---------------------------------------------

        uint32_t seq = 0;
        double last_timestamp = -1.0;
        size_t frame_count = 0;

        for (const auto& view : reader.readMessages()) {
            const mcap::Message& msg = view.message;
            auto ch = reader.channel(msg.channelId);
            
            if (!ch || ch->topic != input_topic) continue;

            const std::string payload(reinterpret_cast<const char*>(msg.data), msg.dataSize);
            const FrameData frame = parseFrame(json::parse(payload), contact_map);

            if (last_timestamp < 0) { last_timestamp = frame.timestamp; continue; }
            if (frame.timestamp <= last_timestamp) continue;
            last_timestamp = frame.timestamp;

            if (frame_count % 100 == 0) std::cout << "\r[INFO] Frame " << frame_count << " / " << max_frames << std::flush;

            if (frame.ground_truth.has_value())
                estimator.filter(frame.imu, frame.joints, frame.forces, std::nullopt, std::nullopt, frame.ground_truth);
            else
                estimator.filter(frame.imu, frame.joints, frame.forces);
            
            auto state = estimator.getState(true);
            
            if (state.has_value()) {
                json j_out;
                
                auto basePos = state->getBasePosition();
                auto baseOrient = state->getBaseOrientation();
                auto baseLinVel = state->getBaseLinearVelocity();

                j_out["timestamp"] = frame.timestamp;
                j_out["base_pose"]["position"] = { {"x", basePos.x()}, {"y", basePos.y()}, {"z", basePos.z()} };
                j_out["base_pose"]["rotation"] = { {"w", baseOrient.w()}, {"x", baseOrient.x()}, {"y", baseOrient.y()}, {"z", baseOrient.z()} };
                j_out["base_pose"]["linear_velocity"] = { {"x", baseLinVel.x()}, {"y", baseLinVel.y()}, {"z", baseLinVel.z()} };
                auto comPos = state->getCoMPosition();
                auto comVel = state->getCoMLinearVelocity();
                auto extForce = state->getCoMExternalForces();

                j_out["CoM_state"]["position"] = { {"x", comPos.x()}, {"y", comPos.y()}, {"z", comPos.z()} };
                j_out["CoM_state"]["velocity"] = { {"x", comVel.x()}, {"y", comVel.y()}, {"z", comVel.z()} };
                j_out["CoM_state"]["externalForces"] = { {"x", extForce.x()}, {"y", extForce.y()}, {"z", extForce.z()} };

                auto biasAcc = state->getImuLinearAccelerationBias();
                auto biasGyr = state->getImuAngularVelocityBias();

                j_out["imu_bias"]["accel"] = { {"x", biasAcc.x()}, {"y", biasAcc.y()}, {"z", biasAcc.z()} };
                j_out["imu_bias"]["angVel"] = { {"x", biasGyr.x()}, {"y", biasGyr.y()}, {"z", biasGyr.z()} };

                for (const auto& [json_key, serow_link_name] : contact_map) {
                    auto prob = state->getContactProbability(serow_link_name);
                    j_out["contact_probabilities"][serow_link_name] = prob.has_value() ? prob.value() : 0.0; 
                }

                std::string output_payload = j_out.dump();

                mcap::Message outMsg;
                outMsg.channelId = channel.id;
                outMsg.sequence = seq++;
                outMsg.logTime = (msg.logTime > 0) ? msg.logTime : toNs(frame.timestamp);
                outMsg.publishTime = outMsg.logTime;
                outMsg.data = reinterpret_cast<const std::byte*>(output_payload.data());
                outMsg.dataSize = output_payload.size();

                requireOk(writer.write(outMsg), "Failed to write message");
            }            

            ++frame_count;

            if (frame_count >= max_frames) {
                std::cout << "\n[INFO] Reached target limit of " << percentage << "% (" << max_frames << " frames). Stopping early.\n";
                break;
            }
            // -----------------------
        }
        
        writer.close();
        reader.close();
        std::cout << "\n[INFO] Done. Processed " << frame_count << " frames.\n";

    } catch (const std::exception& e) {
        std::cerr << "\n[FATAL] " << e.what() << "\n";
        return 1;
    }
    return 0;
}
