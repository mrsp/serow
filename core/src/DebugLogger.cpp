#include "DebugLogger.hpp"

struct BinaryVector3d {
    double x;
    double y;
    double z;
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
        double position;
        double velocity;
    } joints[];
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
    BinaryContactData contacts[];
};

namespace serow {

DebugLogger::DebugLogger(const std::string& filename) {
    // Open the file
    file_writer_ = std::make_unique<mcap::FileWriter>();
    if (!file_writer_->open(filename).ok()) {
        throw std::runtime_error("Failed to open MCAP file for writing");
    }

    // Create the logger
    writer_ = std::make_unique<mcap::McapWriter>();
    mcap::McapWriterOptions options("ros2");  // Use ROS2 profile
    writer_->open(*file_writer_, options);

    // Define schema for Vector3d
    mcap::Schema vector3d_schema;
    vector3d_schema.name = "BinaryVector3d";
    vector3d_schema.encoding = "binary";
    std::string vector3d_schema_data = R"({
        "fields": [
            {"name": "x", "type": "float64"},
            {"name": "y", "type": "float64"},
            {"name": "z", "type": "float64"}
        ]
    })";
    vector3d_schema.data =
        mcap::ByteArray(reinterpret_cast<const std::byte*>(vector3d_schema_data.data()),
                        reinterpret_cast<const std::byte*>(vector3d_schema_data.data() +
                                                           vector3d_schema_data.size()));
    writer_->addSchema(vector3d_schema);

    // Define schema for ImuMeasurement
    mcap::Schema imu_schema;
    imu_schema.name = "BinaryImuMeasurement";
    imu_schema.encoding = "binary";
    std::string imu_schema_data = R"({
        "fields": [
            {"name": "timestamp", "type": "float64"},
            {"name": "linear_acceleration", "type": "float64[3]"},
            {"name": "angular_velocity", "type": "float64[3]"}
        ]
    })";
    imu_schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(imu_schema_data.data()),
        reinterpret_cast<const std::byte*>(imu_schema_data.data() + imu_schema_data.size()));
    writer_->addSchema(imu_schema);

    // Define schema for JointMeasurement
    mcap::Schema joint_schema;
    joint_schema.name = "BinaryJointMeasurement";
    joint_schema.encoding = "binary";
    std::string joint_schema_data = R"({
        "fields": [
            {"name": "timestamp", "type": "float64"},
            {"name": "num_joints", "type": "uint32"},
            {"name": "joints", "type": "array", "items": {
                "fields": [
                    {"name": "position", "type": "float64"},
                    {"name": "velocity", "type": "float64"}
                ]
            }}
        ]
    })";
    joint_schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(joint_schema_data.data()),
        reinterpret_cast<const std::byte*>(joint_schema_data.data() + joint_schema_data.size()));
    writer_->addSchema(joint_schema);

    // Define schema for ContactState
    mcap::Schema contact_schema;
    contact_schema.name = "BinaryContactState";
    contact_schema.encoding = "binary";
    std::string contact_schema_data = R"({
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
    contact_schema.data =
        mcap::ByteArray(reinterpret_cast<const std::byte*>(contact_schema_data.data()),
                        reinterpret_cast<const std::byte*>(contact_schema_data.data() +
                                                           contact_schema_data.size()));
    writer_->addSchema(contact_schema);

    // Add channels
    mcap::Channel vector3d_channel;
    vector3d_channel.schemaId = 0;  // First schema (Vector3d)
    vector3d_channel.messageEncoding = "binary";
    writer_->addChannel(vector3d_channel);

    mcap::Channel imu_channel;
    imu_channel.schemaId = 1;  // Second schema (ImuMeasurement)
    imu_channel.messageEncoding = "binary";
    writer_->addChannel(imu_channel);

    mcap::Channel joint_channel;
    joint_channel.schemaId = 2;  // Third schema (JointMeasurement)
    joint_channel.messageEncoding = "binary";
    writer_->addChannel(joint_channel);

    mcap::Channel contact_channel;
    contact_channel.schemaId = 3;  // Fourth schema (ContactState)
    contact_channel.messageEncoding = "binary";
    writer_->addChannel(contact_channel);
}

void DebugLogger::log(const BaseState& base_state) {
    // Create a message with the schema and data
    mcap::Message message;
    message.channelId = 0;  // base_position channel
    message.sequence = base_position_sequence_++;
    message.logTime = base_state.timestamp;
    message.publishTime = base_state.timestamp;

    // Convert the base state to our binary format
    BinaryVector3d data;
    data.x = base_state.base_position.x();
    data.y = base_state.base_position.y();
    data.z = base_state.base_position.z();

    message.dataSize = sizeof(BinaryVector3d);
    message.data = reinterpret_cast<const std::byte*>(&data);
    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write message" << std::endl;
    }
}

void DebugLogger::log(const CentroidalState& centroidal_state) {}

void DebugLogger::log(const ContactState& contact_state) {
    if (contact_state.contacts_status.empty()) {
        return;
    }

    // Calculate total size needed for the message
    const size_t num_contacts = contact_state.contacts_status.size();
    const size_t header_size = sizeof(double) + sizeof(uint32_t);  // timestamp + num_contacts
    const size_t total_size = header_size + (num_contacts * sizeof(BinaryContactData));

    // Allocate memory for the message
    std::vector<std::byte> buffer(total_size);
    auto* data = reinterpret_cast<BinaryContactState*>(buffer.data());

    // Set the timestamp and number of contacts
    data->timestamp = contact_state.timestamp;
    data->num_contacts = static_cast<uint32_t>(num_contacts);

    // Fill in contact data
    size_t contact_idx = 0;
    for (const auto& [contact_name, status] : contact_state.contacts_status) {
        auto& contact_data = data->contacts[contact_idx];

        // Copy contact name (truncate if too long)
        std::strncpy(contact_data.name, contact_name.c_str(), sizeof(contact_data.name) - 1);
        contact_data.name[sizeof(contact_data.name) - 1] = '\0';  // Ensure null termination

        // Status and probability
        contact_data.status = status;
        contact_data.probability = contact_state.contacts_probability.at(contact_name);

        // Force
        const auto& force = contact_state.contacts_force.at(contact_name);
        contact_data.force[0] = force.x();
        contact_data.force[1] = force.y();
        contact_data.force[2] = force.z();

        // Torque (if available)
        contact_data.has_torque = contact_state.contacts_torque.has_value();
        if (contact_data.has_torque) {
            const auto& torque = contact_state.contacts_torque.value().at(contact_name);
            contact_data.torque[0] = torque.x();
            contact_data.torque[1] = torque.y();
            contact_data.torque[2] = torque.z();
        } else {
            contact_data.torque[0] = 0.0;
            contact_data.torque[1] = 0.0;
            contact_data.torque[2] = 0.0;
        }

        contact_idx++;
    }

    // Create and write the message
    mcap::Message message;
    message.channelId = 3;  // contact_state channel
    message.sequence = contact_sequence_++;
    message.logTime = data->timestamp;
    message.publishTime = data->timestamp;
    message.dataSize = total_size;
    message.data = buffer.data();

    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write contact state message" << std::endl;
    }
}

void DebugLogger::log(const ImuMeasurement& imu_measurement) {
    mcap::Message message;
    message.channelId = 1;  // imu_measurement channel
    message.sequence = imu_sequence_++;
    message.logTime = imu_measurement.timestamp;
    message.publishTime = imu_measurement.timestamp;

    // Convert the IMU measurement to our binary format
    BinaryImuMeasurement data;
    data.timestamp = imu_measurement.timestamp;
    data.linear_acceleration[0] = imu_measurement.linear_acceleration.x();
    data.linear_acceleration[1] = imu_measurement.linear_acceleration.y();
    data.linear_acceleration[2] = imu_measurement.linear_acceleration.z();
    data.angular_velocity[0] = imu_measurement.angular_velocity.x();
    data.angular_velocity[1] = imu_measurement.angular_velocity.y();
    data.angular_velocity[2] = imu_measurement.angular_velocity.z();

    message.dataSize = sizeof(BinaryImuMeasurement);
    message.data = reinterpret_cast<const std::byte*>(&data);
    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write IMU message" << std::endl;
    }
}

void DebugLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {
    if (joints_measurement.empty()) {
        return;
    }

    // Calculate total size needed for the message
    const size_t num_joints = joints_measurement.size();
    const size_t header_size = sizeof(double) + sizeof(uint32_t);  // timestamp + num_joints
    const size_t joint_data_size = sizeof(BinaryJointMeasurement::JointData);
    const size_t total_size = header_size + (num_joints * joint_data_size);

    // Allocate memory for the message
    std::vector<std::byte> buffer(total_size);
    auto* data = reinterpret_cast<BinaryJointMeasurement*>(buffer.data());

    // Set the timestamp (using the first joint's timestamp)
    data->timestamp = joints_measurement.begin()->second.timestamp;
    data->num_joints = static_cast<uint32_t>(num_joints);

    // Fill in joint data
    size_t joint_idx = 0;
    for (const auto& [joint_name, measurement] : joints_measurement) {
        data->joints[joint_idx].position = measurement.position;
        data->joints[joint_idx].velocity =
            measurement.velocity.has_value() ? measurement.velocity.value() : 0.0;
        joint_idx++;
    }

    // Create and write the message
    mcap::Message message;
    message.channelId = 2;  // joint_measurement channel
    message.sequence = joint_sequence_++;
    message.logTime = data->timestamp;
    message.publishTime = data->timestamp;
    message.dataSize = total_size;
    message.data = buffer.data();

    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write joint measurements message" << std::endl;
    }
}

void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {}

DebugLogger::~DebugLogger() {
    if (writer_)
        writer_->close();
    writer_.reset();
    file_writer_.reset();
}

}  // namespace serow
