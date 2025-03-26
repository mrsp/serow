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

    void log(const BaseState& base_state) {
        // Minimize copies using Eigen::Map
        BinaryBaseState data{};
        data.timestamp = base_state.timestamp;

        // Zero-copy Eigen mapping
        Eigen::Map<Eigen::Vector3d> pos_map(data.base_position);
        pos_map = base_state.base_position;

        Eigen::Map<Eigen::Vector4d> orient_map(data.base_orientation);
        orient_map << base_state.base_orientation.w(), base_state.base_orientation.x(),
            base_state.base_orientation.y(), base_state.base_orientation.z();

        // Similar zero-copy mappings for other vectors
        Eigen::Map<Eigen::Vector3d>(data.base_linear_velocity) = base_state.base_linear_velocity;
        Eigen::Map<Eigen::Vector3d>(data.base_angular_velocity) = base_state.base_angular_velocity;
        Eigen::Map<Eigen::Vector3d>(data.base_linear_acceleration) =
            base_state.base_linear_acceleration;
        Eigen::Map<Eigen::Vector3d>(data.base_angular_acceleration) =
            base_state.base_angular_acceleration;
        Eigen::Map<Eigen::Vector3d>(data.imu_linear_acceleration_bias) =
            base_state.imu_linear_acceleration_bias;
        Eigen::Map<Eigen::Vector3d>(data.imu_angular_velocity_bias) =
            base_state.imu_angular_velocity_bias;

        // Handle contacts and feet
        data.num_contacts = static_cast<uint32_t>(base_state.contacts_position.size());
        data.num_feet = static_cast<uint32_t>(base_state.feet_position.size());

        // Convert contacts with minimal copying
        data.contacts.reserve(data.num_contacts);
        for (const auto& [contact_name, position] : base_state.contacts_position) {
            BinaryBaseState::ContactData contact_data{};

            // Efficient name copying
            std::strncpy(contact_data.name, contact_name.c_str(), sizeof(contact_data.name) - 1);
            contact_data.name[sizeof(contact_data.name) - 1] = '\0';

            // Zero-copy position mapping
            Eigen::Map<Eigen::Vector3d>(contact_data.position) = position;

            // Optional orientation handling
            if (base_state.contacts_orientation) {
                const auto& orientation = base_state.contacts_orientation.value().at(contact_name);
                Eigen::Map<Eigen::Vector4d>(contact_data.orientation) << orientation.w(),
                    orientation.x(), orientation.y(), orientation.z();
            } else {
                contact_data.orientation[0] = 1.0;  // Identity quaternion
                contact_data.orientation[1] = 0.0;
                contact_data.orientation[2] = 0.0;
                contact_data.orientation[3] = 0.0;
            }

            data.contacts.push_back(contact_data);
        }

        // Similar zero-copy approach for feet data
        data.feet.reserve(data.num_feet);
        for (const auto& [foot_name, position] : base_state.feet_position) {
            BinaryBaseState::FootData foot_data{};

            std::strncpy(foot_data.name, foot_name.c_str(), sizeof(foot_data.name) - 1);
            foot_data.name[sizeof(foot_data.name) - 1] = '\0';

            Eigen::Map<Eigen::Vector3d>(foot_data.position) = position;

            const auto& orientation = base_state.feet_orientation.at(foot_name);
            Eigen::Map<Eigen::Vector4d>(foot_data.orientation) << orientation.w(), orientation.x(),
                orientation.y(), orientation.z();

            const auto& linear_velocity = base_state.feet_linear_velocity.at(foot_name);
            Eigen::Map<Eigen::Vector3d>(foot_data.linear_velocity) = linear_velocity;

            const auto& angular_velocity = base_state.feet_angular_velocity.at(foot_name);
            Eigen::Map<Eigen::Vector3d>(foot_data.angular_velocity) = angular_velocity;

            data.feet.push_back(foot_data);
        }

        // Calculate exact memory requirement
        size_t total_size = sizeof(BinaryBaseState) +
                            (data.contacts.size() * sizeof(BinaryBaseState::ContactData)) +
                            (data.feet.size() * sizeof(BinaryBaseState::FootData));

        // Write message with precise sizing
        writeMessage(0, base_state_sequence_++, base_state.timestamp,
                     reinterpret_cast<const std::byte*>(&data), total_size);
    }

    void log(const CentroidalState& centroidal_state) {
        BinaryCentroidalState data{};
        data.timestamp = centroidal_state.timestamp;

        // Zero-copy Eigen mapping
        Eigen::Map<Eigen::Vector3d>(data.com_position) = centroidal_state.com_position;
        Eigen::Map<Eigen::Vector3d>(data.com_linear_velocity) =
            centroidal_state.com_linear_velocity;
        Eigen::Map<Eigen::Vector3d>(data.external_forces) = centroidal_state.external_forces;

        writeMessage(5, centroidal_sequence_++, centroidal_state.timestamp,
                     reinterpret_cast<const std::byte*>(&data), sizeof(BinaryCentroidalState));
    }

    void log(const ImuMeasurement& imu_measurement) {
        BinaryImuMeasurement data{};
        data.timestamp = imu_measurement.timestamp;

        // Zero-copy Eigen mapping for IMU data
        Eigen::Map<Eigen::Vector3d>(data.linear_acceleration) = imu_measurement.linear_acceleration;
        Eigen::Map<Eigen::Vector3d>(data.angular_velocity) = imu_measurement.angular_velocity;

        writeMessage(1, imu_sequence_++, imu_measurement.timestamp,
                     reinterpret_cast<const std::byte*>(&data), sizeof(BinaryImuMeasurement));
    }

    void log(const std::map<std::string, JointMeasurement>& joints_measurement) {
        if (joints_measurement.empty()) {
            return;
        }

        // Use the first measurement's timestamp as the message timestamp
        const auto& first_measurement = joints_measurement.begin()->second;
        BinaryJointMeasurement data{};
        data.timestamp = first_measurement.timestamp;
        data.num_joints = static_cast<uint32_t>(joints_measurement.size());

        // Pre-allocate vector space
        data.joints.reserve(data.num_joints);

        // Fill in joint data
        for (const auto& [joint_name, measurement] : joints_measurement) {
            BinaryJointMeasurement::JointData joint_data{};

            // Copy joint name
            std::strncpy(joint_data.name, joint_name.c_str(), sizeof(joint_data.name) - 1);
            joint_data.name[sizeof(joint_data.name) - 1] = '\0';

            // Copy position and velocity
            joint_data.position = measurement.position;
            joint_data.velocity =
                measurement.velocity.value_or(0.0);  // Use 0.0 if velocity is not available

            data.joints.push_back(joint_data);
        }

        // Calculate total size needed for the message
        const size_t header_size = sizeof(double) + sizeof(uint32_t);  // timestamp + num_joints
        const size_t total_size =
            header_size + (data.joints.size() * sizeof(BinaryJointMeasurement::JointData));

        // Allocate memory for the message
        std::vector<std::byte> buffer(total_size);
        auto* data_ptr = reinterpret_cast<BinaryJointMeasurement*>(buffer.data());
        data_ptr->timestamp = data.timestamp;
        data_ptr->num_joints = data.num_joints;

        // Copy joint data
        std::memcpy(reinterpret_cast<std::byte*>(data_ptr) + header_size, data.joints.data(),
                    data.joints.size() * sizeof(BinaryJointMeasurement::JointData));

        writeMessage(2, joint_sequence_++, data.timestamp, buffer.data(), total_size);
    }

    void log(const ContactState& contact_state) {
        BinaryContactState data{};
        data.timestamp = contact_state.timestamp;
        data.num_contacts = static_cast<uint32_t>(contact_state.contacts_status.size());

        // Pre-allocate vector space
        data.contacts.reserve(data.num_contacts);

        // Fill in contact data
        for (const auto& [contact_name, contact_status] : contact_state.contacts_status) {
            BinaryContactData contact_data{};

            // Copy contact name
            std::strncpy(contact_data.name, contact_name.c_str(), sizeof(contact_data.name) - 1);
            contact_data.name[sizeof(contact_data.name) - 1] = '\0';

            // Copy contact data
            contact_data.status = contact_status;
            contact_data.probability = 1.0;   // Default value since ContactState only has bool
            contact_data.has_torque = false;  // Default value since ContactState only has bool

            // Set default force and torque
            contact_data.force[0] = 0.0;
            contact_data.force[1] = 0.0;
            contact_data.force[2] = 0.0;
            contact_data.torque[0] = 0.0;
            contact_data.torque[1] = 0.0;
            contact_data.torque[2] = 0.0;

            data.contacts.push_back(contact_data);
        }

        // Calculate total size needed for the message
        const size_t header_size = sizeof(double) + sizeof(uint32_t);  // timestamp + num_contacts
        const size_t total_size = header_size + (data.contacts.size() * sizeof(BinaryContactData));

        // Allocate memory for the message
        std::vector<std::byte> buffer(total_size);
        auto* data_ptr = reinterpret_cast<BinaryContactState*>(buffer.data());
        data_ptr->timestamp = data.timestamp;
        data_ptr->num_contacts = data.num_contacts;

        // Copy contact data
        std::memcpy(reinterpret_cast<std::byte*>(data_ptr) + header_size, data.contacts.data(),
                    data.contacts.size() * sizeof(BinaryContactData));

        writeMessage(3, contact_sequence_++, contact_state.timestamp, buffer.data(), total_size);
    }

    void log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {
        if (ft_measurement.empty()) {
            return;
        }

        BinaryForceTorqueMeasurement data{};
        data.timestamp = ft_measurement.begin()->second.timestamp;
        data.num_measurements = static_cast<uint32_t>(ft_measurement.size());

        // Calculate total size needed for the message
        const size_t header_size =
            sizeof(double) + sizeof(uint32_t);  // timestamp + num_measurements
        const size_t measurement_size = sizeof(BinaryForceTorqueData);
        const size_t total_size = header_size + (data.num_measurements * measurement_size);

        // Allocate memory for the message
        std::vector<std::byte> buffer(total_size);
        auto* data_ptr = reinterpret_cast<BinaryForceTorqueMeasurement*>(buffer.data());
        data_ptr->timestamp = data.timestamp;
        data_ptr->num_measurements = data.num_measurements;

        // Fill in measurement data
        size_t measurement_idx = 0;
        for (const auto& [sensor_name, measurement] : ft_measurement) {
            auto& measurement_data = data_ptr->measurements[measurement_idx];

            // Copy sensor name
            std::strncpy(measurement_data.name, sensor_name.c_str(),
                         sizeof(measurement_data.name) - 1);
            measurement_data.name[sizeof(measurement_data.name) - 1] = '\0';

            // Copy measurement data
            measurement_data.timestamp = measurement.timestamp;
            Eigen::Map<Eigen::Vector3d>(measurement_data.force) = measurement.force;
            Eigen::Map<Eigen::Vector3d>(measurement_data.cop) = measurement.cop;

            // Handle torque if available
            measurement_data.has_torque = measurement.torque.has_value();
            if (measurement_data.has_torque) {
                Eigen::Map<Eigen::Vector3d>(measurement_data.torque) = measurement.torque.value();
            } else {
                measurement_data.torque[0] = 0.0;
                measurement_data.torque[1] = 0.0;
                measurement_data.torque[2] = 0.0;
            }

            measurement_idx++;
        }

        writeMessage(4, ft_sequence_++, data.timestamp, buffer.data(), total_size);
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
        schemas.push_back(createBaseStateSchema());
        schemas.push_back(createImuMeasurementSchema());
        schemas.push_back(createJointMeasurementSchema());
        schemas.push_back(createContactStateSchema());
        schemas.push_back(createForceTorqueMeasurementSchema());
        schemas.push_back(createCentroidalStateSchema());

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.push_back(createChannel(0, "BaseState"));
        channels.push_back(createChannel(1, "ImuMeasurement"));
        channels.push_back(createChannel(2, "JointMeasurement"));
        channels.push_back(createChannel(3, "ContactState"));
        channels.push_back(createChannel(4, "ForceTorqueMeasurement"));
        channels.push_back(createChannel(5, "CentroidalState"));

        for (auto& channel : channels) {
            writer_->addChannel(channel);
        }
    }

    // Helper methods for schema and channel creation
    mcap::Schema createBaseStateSchema() {
        mcap::Schema schema;
        schema.name = "BinaryBaseState";
        schema.encoding = "binary";

        // JSON schema description
        std::string schema_data = R"({
            "fields": [
                {"name": "timestamp", "type": "float64"},
                {"name": "base_position", "type": "float64[3]"},
                {"name": "base_orientation", "type": "float64[4]"},
                {"name": "base_linear_velocity", "type": "float64[3]"},
                {"name": "base_angular_velocity", "type": "float64[3]"},
                {"name": "base_linear_acceleration", "type": "float64[3]"},
                {"name": "base_angular_acceleration", "type": "float64[3]"},
                {"name": "imu_linear_acceleration_bias", "type": "float64[3]"},
                {"name": "imu_angular_velocity_bias", "type": "float64[3]"},
                {"name": "num_contacts", "type": "uint32"},
                {"name": "num_feet", "type": "uint32"},
                {"name": "contacts", "type": "array", "items": {
                    "fields": [
                        {"name": "name", "type": "string", "size": 32},
                        {"name": "position", "type": "float64[3]"},
                        {"name": "orientation", "type": "float64[4]"}
                    ]
                }},
                {"name": "feet", "type": "array", "items": {
                    "fields": [
                        {"name": "name", "type": "string", "size": 32},
                        {"name": "position", "type": "float64[3]"},
                        {"name": "orientation", "type": "float64[4]"},
                        {"name": "linear_velocity", "type": "float64[3]"},
                        {"name": "angular_velocity", "type": "float64[3]"}
                    ]
                }}
            ]
        })";

        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));

        return schema;
    }

    mcap::Schema createImuMeasurementSchema() {
        mcap::Schema schema;
        schema.name = "BinaryImuMeasurement";
        schema.encoding = "binary";
        std::string schema_data = R"({
            "fields": [
                {"name": "timestamp", "type": "float64"},
                {"name": "linear_acceleration", "type": "float64[3]"},
                {"name": "angular_velocity", "type": "float64[3]"},
                {"name": "orientation", "type": "float64[4]"}
            ]
        })";
        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createJointMeasurementSchema() {
        mcap::Schema schema;
        schema.name = "BinaryJointMeasurement";
        schema.encoding = "binary";
        std::string schema_data = R"({
            "fields": [
                {"name": "timestamp", "type": "float64"},
                {"name": "num_joints", "type": "uint32"},
                {"name": "joints", "type": "array", "items": {
                    "fields": [
                        {"name": "name", "type": "string"},
                        {"name": "position", "type": "float64"},
                        {"name": "velocity", "type": "float64"}
                    ]
                }}
            ]
        })";
        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createContactStateSchema() {
        mcap::Schema schema;
        schema.name = "BinaryContactState";
        schema.encoding = "binary";
        std::string schema_data = R"({
            "fields": [
                {"name": "timestamp", "type": "float64"},
                {"name": "num_contacts", "type": "uint32"},
                {"name": "contacts", "type": "array", "items": {
                    "fields": [
                        {"name": "name", "type": "string"},
                        {"name": "status", "type": "bool"},
                        {"name": "probability", "type": "float64"},
                        {"name": "force", "type": "float64[3]"},
                        {"name": "torque", "type": "float64[3]"},
                        {"name": "has_torque", "type": "bool"}
                    ]
                }}
            ]
        })";
        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createForceTorqueMeasurementSchema() {
        mcap::Schema schema;
        schema.name = "BinaryForceTorqueMeasurement";
        schema.encoding = "binary";
        std::string schema_data = R"({
            "fields": [
                {"name": "timestamp", "type": "float64"},
                {"name": "num_measurements", "type": "uint32"},
                {"name": "measurements", "type": "array", "items": {
                    "fields": [
                        {"name": "name", "type": "string"},
                        {"name": "timestamp", "type": "float64"},
                        {"name": "force", "type": "float64[3]"},
                        {"name": "cop", "type": "float64[3]"},
                        {"name": "has_torque", "type": "bool"},
                        {"name": "torque", "type": "float64[3]"}
                    ]
                }}
            ]
        })";
        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createCentroidalStateSchema() {
        mcap::Schema schema;
        schema.name = "BinaryCentroidalState";
        schema.encoding = "binary";
        std::string schema_data = R"({
            "fields": [
                {"name": "timestamp", "type": "float64"},
                {"name": "com_position", "type": "float64[3]"},
                {"name": "com_linear_velocity", "type": "float64[3]"},
                {"name": "external_forces", "type": "float64[3]"}
            ]
        })";
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
        channel.messageEncoding = "binary";
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

void DebugLogger::log(const BaseState& base_state) { pimpl_->log(base_state); }
void DebugLogger::log(const CentroidalState& centroidal_state) { pimpl_->log(centroidal_state); }
void DebugLogger::log(const ImuMeasurement& imu_measurement) { pimpl_->log(imu_measurement); }
void DebugLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {
    pimpl_->log(joints_measurement);
}
void DebugLogger::log(const ContactState& contact_state) { pimpl_->log(contact_state); }
void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {
    pimpl_->log(ft_measurement);
}

}  // namespace serow