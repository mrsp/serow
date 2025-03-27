/**
 * Copyright (C) 2025 Stylianos Piperakis, Ownage Dynamics L.P.
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
#include "DebugLogger.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <mcap/mcap.hpp>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace {

struct BinaryBaseState {
    double timestamp;
    double base_position[3];                 // x, y, z
    double base_orientation[4];              // w, x, y, z (quaternion)
    double base_linear_velocity[3];          // vx, vy, vz
    double base_angular_velocity[3];         // wx, wy, wz
    double base_linear_acceleration[3];      // ax, ay, az
    double base_angular_acceleration[3];     // alpha_x, alpha_y, alpha_z
    double imu_linear_acceleration_bias[3];  // bias_x, bias_y, bias_z
    double imu_angular_velocity_bias[3];     // bias_wx, bias_wy, bias_wz
    uint32_t num_contacts;                   // Number of contacts
    uint32_t num_feet;                       // Number of feet
    // Dynamic arrays for contacts and feet data
    struct ContactData {
        char name[32];          // Contact name
        double position[3];     // x, y, z
        double orientation[4];  // w, x, y, z (quaternion)
    };
    struct FootData {
        char name[32];               // Foot name
        double position[3];          // x, y, z
        double orientation[4];       // w, x, y, z (quaternion)
        double linear_velocity[3];   // vx, vy, vz
        double angular_velocity[3];  // wx, wy, wz
    };
    std::vector<ContactData> contacts;
    std::vector<FootData> feet;
};

struct BinaryImuMeasurement {
    double timestamp;
    double linear_acceleration[3];
    double angular_velocity[3];
};

struct BinaryJointMeasurement {
    double timestamp;
    uint32_t num_joints;
    // Dynamic array of joint data
    struct JointData {
        char name[32];  // Fixed size for joint name
        double position;
        double velocity;
    };
    std::vector<JointData> joints;  // Use vector instead of flexible array
};

struct BinaryContactData {
    char name[32];  // Fixed size for contact name
    bool status;
    double probability;
    double force[3];   // fx, fy, fz
    double torque[3];  // tx, ty, tz (optional)
    bool has_torque;   // flag to indicate if torque is present
};

struct BinaryContactState {
    double timestamp;
    uint32_t num_contacts;
    // Dynamic array of contact data
    std::vector<BinaryContactData> contacts;  // Use vector instead of flexible array
};

struct BinaryForceTorqueData {
    char name[32];  // Fixed size for sensor name
    double timestamp;
    double force[3];   // fx, fy, fz
    double cop[3];     // copx, copy, copz
    bool has_torque;   // flag to indicate if torque is present
    double torque[3];  // tx, ty, tz (optional)
};

struct BinaryForceTorqueMeasurement {
    double timestamp;
    uint32_t num_measurements;
    // Dynamic array of force-torque data
    std::vector<BinaryForceTorqueData> measurements;  // Use vector instead of flexible array
};

struct BinaryCentroidalState {
    double timestamp;
    double com_position[3];         // x, y, z
    double com_linear_velocity[3];  // vx, vy, vz
    double external_forces[3];      // fx, fy, fz
};

}  // namespace

namespace serow {

// Implementation class definition
class DebugLogger::Impl {
public:
    explicit Impl(const std::string& filename) {
        // Open the file
        file_writer_ = std::make_unique<mcap::FileWriter>();
        if (!file_writer_->open(filename).ok()) {
            throw std::runtime_error("Failed to open MCAP file for writing");
        }

        // Create the logger with ROS2 profile
        writer_ = std::make_unique<mcap::McapWriter>();
        mcap::McapWriterOptions options("ros2");
        writer_->open(*file_writer_, options);

        // Initialize schemas and channels
        initializeSchemas();
        initializeChannels();
    }

    ~Impl() {
        if (writer_) {
            writer_->close();
        }
    }


    void log(const ImuMeasurement& imu_measurement) {
        // Create JSON message
        nlohmann::json json_data;
        json_data["timestamp"] = imu_measurement.timestamp;
        json_data["linear_acceleration"] = {
            imu_measurement.linear_acceleration.x(),
            imu_measurement.linear_acceleration.y(),
            imu_measurement.linear_acceleration.z()
        };
        json_data["angular_velocity"] = {
            imu_measurement.angular_velocity.x(),
            imu_measurement.angular_velocity.y(),
            imu_measurement.angular_velocity.z()
        };

        // Convert JSON to string
        std::string json_str = json_data.dump();
        
        writeMessage(1, imu_sequence_++, imu_measurement.timestamp,
                     reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
    }


private:
    void writeMessage(uint16_t channel_id, uint64_t sequence, double timestamp,
                      const std::byte* data, size_t data_size) {
        mcap::Message message;
        message.channelId = channel_id;
        message.sequence = sequence;
        message.logTime = timestamp;
        message.publishTime = timestamp;
        message.dataSize = data_size;
        message.data = data;

        auto status = writer_->write(message);
        if (status.code != mcap::StatusCode::Success) {
            std::cerr << "Failed to write message for channel " << channel_id << std::endl;
        }
    }

    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        // schemas.push_back(createBaseStateSchema());
        schemas.push_back(createImuMeasurementSchema());
        // schemas.push_back(createJointMeasurementSchema());
        // schemas.push_back(createContactStateSchema());
        // schemas.push_back(createForceTorqueMeasurementSchema());
        // schemas.push_back(createCentroidalStateSchema());

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.push_back(createChannel(1, "ImuMeasurement"));
        // channels.push_back(createChannel(2, "JointMeasurement"));
        // channels.push_back(createChannel(3, "ContactState"));
        // channels.push_back(createChannel(4, "ForceTorqueMeasurement"));
        // channels.push_back(createChannel(5, "CentroidalState"));

        for (auto& channel : channels) {
            writer_->addChannel(channel);
        }
    }


    mcap::Schema createImuMeasurementSchema() {
        mcap::Schema schema;
        schema.name = "ImuMeasurement";
        schema.encoding = "jsonschema";
        std::string schema_data = 
            "{"
            "    \"type\": \"object\","
            "    \"properties\": {"
            "        \"timestamp\": {"
            "            \"type\": \"number\","
            "            \"description\": \"Timestamp of the measurement (s)\""
            "        },"
            "        \"linear_acceleration\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"Linear acceleration in x, y, z (m/s^2)\""
            "        },"
            "        \"angular_velocity\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"Angular velocity in x, y, z (rad/s)\""
            "        }"
            "    },"
            "    \"required\": [\"timestamp\", \"linear_acceleration\", \"angular_velocity\"]"
            "}";
        
        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }


    mcap::Channel createChannel(uint16_t id, const std::string& topic) {
        mcap::Channel channel;
        channel.id = id;
        channel.topic = topic;
        channel.schemaId = id;
        channel.messageEncoding = "json";  // Changed from "binary" to "json"
        return channel;
    }

    // Sequence counters
    uint64_t base_state_sequence_ = 0;
    uint64_t centroidal_sequence_ = 0;
    uint64_t contact_sequence_ = 0;
    uint64_t imu_sequence_ = 0;
    uint64_t joint_sequence_ = 0;
    uint64_t ft_sequence_ = 0;

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
};

// Public interface implementation
DebugLogger::DebugLogger(const std::string& filename) : pimpl_(std::make_unique<Impl>(filename)) {}

DebugLogger::~DebugLogger() = default;

// void DebugLogger::log(const BaseState& base_state) { pimpl_->log(base_state); }
// void DebugLogger::log(const CentroidalState& centroidal_state) { pimpl_->log(centroidal_state); }
void DebugLogger::log(const ImuMeasurement& imu_measurement) { pimpl_->log(imu_measurement); }
// void DebugLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {
//     pimpl_->log(joints_measurement);
// }
// void DebugLogger::log(const ContactState& contact_state) { pimpl_->log(contact_state); }
// void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {
//     pimpl_->log(ft_measurement);
// }

}  // namespace serow
