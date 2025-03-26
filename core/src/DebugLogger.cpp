#include "DebugLogger.hpp"

struct BinaryBaseState {
    double timestamp;
    double base_position[3];           // x, y, z
    double base_orientation[4];        // w, x, y, z (quaternion)
    double base_linear_velocity[3];    // vx, vy, vz
    double base_angular_velocity[3];   // wx, wy, wz
    double base_linear_acceleration[3]; // ax, ay, az
    double base_angular_acceleration[3]; // alpha_x, alpha_y, alpha_z
    double imu_linear_acceleration_bias[3]; // bias_x, bias_y, bias_z
    double imu_angular_velocity_bias[3];    // bias_wx, bias_wy, bias_wz
    uint32_t num_contacts;            // Number of contacts
    uint32_t num_feet;                // Number of feet
    // Dynamic arrays for contacts and feet data
    struct ContactData {
        char name[32];                // Contact name
        double position[3];           // x, y, z
        double orientation[4];        // w, x, y, z (quaternion)
    };
    struct FootData {
        char name[32];                // Foot name
        double position[3];           // x, y, z
        double orientation[4];        // w, x, y, z (quaternion)
        double linear_velocity[3];    // vx, vy, vz
        double angular_velocity[3];   // wx, wy, wz
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

struct BinaryForceTorqueMeasurement {
    double timestamp;
    double force[3];    // fx, fy, fz
    double cop[3];      // copx, copy, copz
    bool has_torque;    // flag to indicate if torque is present
    double torque[3];   // tx, ty, tz (optional)
};

struct BinaryCentroidalState {
    double timestamp;
    double com_position[3];           // x, y, z
    double com_linear_velocity[3];    // vx, vy, vz
    double external_forces[3];        // fx, fy, fz
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

    // Define schema for BaseState
    mcap::Schema base_state_schema;
    base_state_schema.name = "BinaryBaseState";
    base_state_schema.encoding = "binary";
    std::string base_state_schema_data = R"({
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
                    {"name": "name", "type": "string"},
                    {"name": "position", "type": "float64[3]"},
                    {"name": "orientation", "type": "float64[4]"}
                ]
            }},
            {"name": "feet", "type": "array", "items": {
                "fields": [
                    {"name": "name", "type": "string"},
                    {"name": "position", "type": "float64[3]"},
                    {"name": "orientation", "type": "float64[4]"},
                    {"name": "linear_velocity", "type": "float64[3]"},
                    {"name": "angular_velocity", "type": "float64[3]"}
                ]
            }}
        ]
    })";
    base_state_schema.data =
        mcap::ByteArray(reinterpret_cast<const std::byte*>(base_state_schema_data.data()),
                        reinterpret_cast<const std::byte*>(base_state_schema_data.data() +
                                                           base_state_schema_data.size()));
    writer_->addSchema(base_state_schema);

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

    // Define schema for ForceTorqueMeasurement
    mcap::Schema ft_schema;
    ft_schema.name = "BinaryForceTorqueMeasurement";
    ft_schema.encoding = "binary";
    std::string ft_schema_data = R"({
        "fields": [
            {"name": "timestamp", "type": "float64"},
            {"name": "force", "type": "float64[3]"},
            {"name": "cop", "type": "float64[3]"},
            {"name": "has_torque", "type": "bool"},
            {"name": "torque", "type": "float64[3]"}
        ]
    })";
    ft_schema.data =
        mcap::ByteArray(reinterpret_cast<const std::byte*>(ft_schema_data.data()),
                        reinterpret_cast<const std::byte*>(ft_schema_data.data() +
                                                           ft_schema_data.size()));
    writer_->addSchema(ft_schema);

    // Define schema for CentroidalState
    mcap::Schema centroidal_schema;
    centroidal_schema.name = "BinaryCentroidalState";
    centroidal_schema.encoding = "binary";
    std::string centroidal_schema_data = R"({
        "fields": [
            {"name": "timestamp", "type": "float64"},
            {"name": "com_position", "type": "float64[3]"},
            {"name": "com_linear_velocity", "type": "float64[3]"},
            {"name": "external_forces", "type": "float64[3]"}
        ]
    })";
    centroidal_schema.data =
        mcap::ByteArray(reinterpret_cast<const std::byte*>(centroidal_schema_data.data()),
                        reinterpret_cast<const std::byte*>(centroidal_schema_data.data() +
                                                           centroidal_schema_data.size()));
    writer_->addSchema(centroidal_schema);

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

    mcap::Channel ft_channel;
    ft_channel.schemaId = 4;  // Fifth schema (ForceTorqueMeasurement)
    ft_channel.messageEncoding = "binary";
    writer_->addChannel(ft_channel);

    mcap::Channel centroidal_channel;
    centroidal_channel.schemaId = 5;  // Sixth schema (CentroidalState)
    centroidal_channel.messageEncoding = "binary";
    writer_->addChannel(centroidal_channel);
}

void DebugLogger::log(const BaseState& base_state) {
    // Create the binary state
    BinaryBaseState data;
    data.timestamp = base_state.timestamp;
    data.num_contacts = static_cast<uint32_t>(base_state.contacts_position.size());
    data.num_feet = static_cast<uint32_t>(base_state.feet_position.size());

    // Base state data
    data.base_position[0] = base_state.base_position.x();
    data.base_position[1] = base_state.base_position.y();
    data.base_position[2] = base_state.base_position.z();

    data.base_orientation[0] = base_state.base_orientation.w();
    data.base_orientation[1] = base_state.base_orientation.x();
    data.base_orientation[2] = base_state.base_orientation.y();
    data.base_orientation[3] = base_state.base_orientation.z();

    data.base_linear_velocity[0] = base_state.base_linear_velocity.x();
    data.base_linear_velocity[1] = base_state.base_linear_velocity.y();
    data.base_linear_velocity[2] = base_state.base_linear_velocity.z();

    data.base_angular_velocity[0] = base_state.base_angular_velocity.x();
    data.base_angular_velocity[1] = base_state.base_angular_velocity.y();
    data.base_angular_velocity[2] = base_state.base_angular_velocity.z();

    data.base_linear_acceleration[0] = base_state.base_linear_acceleration.x();
    data.base_linear_acceleration[1] = base_state.base_linear_acceleration.y();
    data.base_linear_acceleration[2] = base_state.base_linear_acceleration.z();

    data.base_angular_acceleration[0] = base_state.base_angular_acceleration.x();
    data.base_angular_acceleration[1] = base_state.base_angular_acceleration.y();
    data.base_angular_acceleration[2] = base_state.base_angular_acceleration.z();

    data.imu_linear_acceleration_bias[0] = base_state.imu_linear_acceleration_bias.x();
    data.imu_linear_acceleration_bias[1] = base_state.imu_linear_acceleration_bias.y();
    data.imu_linear_acceleration_bias[2] = base_state.imu_linear_acceleration_bias.z();

    data.imu_angular_velocity_bias[0] = base_state.imu_angular_velocity_bias.x();
    data.imu_angular_velocity_bias[1] = base_state.imu_angular_velocity_bias.y();
    data.imu_angular_velocity_bias[2] = base_state.imu_angular_velocity_bias.z();

    // Fill in contact data
    data.contacts.reserve(data.num_contacts);
    for (const auto& [contact_name, position] : base_state.contacts_position) {
        BinaryBaseState::ContactData contact_data;

        // Copy contact name
        std::strncpy(contact_data.name, contact_name.c_str(), sizeof(contact_data.name) - 1);
        contact_data.name[sizeof(contact_data.name) - 1] = '\0';

        // Position
        contact_data.position[0] = position.x();
        contact_data.position[1] = position.y();
        contact_data.position[2] = position.z();

        // Orientation (if available)
        if (base_state.contacts_orientation.has_value()) {
            const auto& orientation = base_state.contacts_orientation.value().at(contact_name);
            contact_data.orientation[0] = orientation.w();
            contact_data.orientation[1] = orientation.x();
            contact_data.orientation[2] = orientation.y();
            contact_data.orientation[3] = orientation.z();
        } else {
            contact_data.orientation[0] = 1.0;
            contact_data.orientation[1] = 0.0;
            contact_data.orientation[2] = 0.0;
            contact_data.orientation[3] = 0.0;
        }

        data.contacts.push_back(contact_data);
    }

    // Fill in foot data
    data.feet.reserve(data.num_feet);
    for (const auto& [foot_name, position] : base_state.feet_position) {
        BinaryBaseState::FootData foot_data;

        // Copy foot name
        std::strncpy(foot_data.name, foot_name.c_str(), sizeof(foot_data.name) - 1);
        foot_data.name[sizeof(foot_data.name) - 1] = '\0';

        // Position
        foot_data.position[0] = position.x();
        foot_data.position[1] = position.y();
        foot_data.position[2] = position.z();

        // Orientation
        const auto& orientation = base_state.feet_orientation.at(foot_name);
        foot_data.orientation[0] = orientation.w();
        foot_data.orientation[1] = orientation.x();
        foot_data.orientation[2] = orientation.y();
        foot_data.orientation[3] = orientation.z();

        // Linear velocity
        const auto& linear_velocity = base_state.feet_linear_velocity.at(foot_name);
        foot_data.linear_velocity[0] = linear_velocity.x();
        foot_data.linear_velocity[1] = linear_velocity.y();
        foot_data.linear_velocity[2] = linear_velocity.z();

        // Angular velocity
        const auto& angular_velocity = base_state.feet_angular_velocity.at(foot_name);
        foot_data.angular_velocity[0] = angular_velocity.x();
        foot_data.angular_velocity[1] = angular_velocity.y();
        foot_data.angular_velocity[2] = angular_velocity.z();

        data.feet.push_back(foot_data);
    }

    // Create and write the message
    mcap::Message message;
    message.channelId = 0;  // base_state channel
    message.sequence = base_state_sequence_++;
    message.logTime = data.timestamp;
    message.publishTime = data.timestamp;
    message.dataSize = sizeof(BinaryBaseState) + 
                      (data.contacts.size() * sizeof(BinaryBaseState::ContactData)) +
                      (data.feet.size() * sizeof(BinaryBaseState::FootData));
    message.data = reinterpret_cast<const std::byte*>(&data);

    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write base state message" << std::endl;
    }
}

void DebugLogger::log(const CentroidalState& centroidal_state) {
    mcap::Message message;
    message.channelId = 5;  // centroidal_state channel
    message.sequence = centroidal_sequence_++;
    message.logTime = centroidal_state.timestamp;
    message.publishTime = centroidal_state.timestamp;

    // Convert the centroidal state to our binary format
    BinaryCentroidalState data;
    data.timestamp = centroidal_state.timestamp;

    // CoM position
    data.com_position[0] = centroidal_state.com_position.x();
    data.com_position[1] = centroidal_state.com_position.y();
    data.com_position[2] = centroidal_state.com_position.z();

    // CoM linear velocity
    data.com_linear_velocity[0] = centroidal_state.com_linear_velocity.x();
    data.com_linear_velocity[1] = centroidal_state.com_linear_velocity.y();
    data.com_linear_velocity[2] = centroidal_state.com_linear_velocity.z();

    // External forces
    data.external_forces[0] = centroidal_state.external_forces.x();
    data.external_forces[1] = centroidal_state.external_forces.y();
    data.external_forces[2] = centroidal_state.external_forces.z();

    message.dataSize = sizeof(BinaryCentroidalState);
    message.data = reinterpret_cast<const std::byte*>(&data);
    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write centroidal state message" << std::endl;
    }
}

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

void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {
    if (ft_measurement.empty()) {
        return;
    }

    // Calculate total size needed for the message
    const size_t num_measurements = ft_measurement.size();
    const size_t header_size = sizeof(double) + sizeof(uint32_t);  // timestamp + num_measurements
    const size_t measurement_size = sizeof(BinaryForceTorqueMeasurement);
    const size_t total_size = header_size + (num_measurements * measurement_size);

    // Allocate memory for the message
    std::vector<std::byte> buffer(total_size);
    auto* data = reinterpret_cast<BinaryForceTorqueMeasurement*>(buffer.data());

    // Set the timestamp (using the first measurement's timestamp)
    data->timestamp = ft_measurement.begin()->second.timestamp;

    // Fill in measurement data
    size_t measurement_idx = 0;
    for (const auto& [sensor_name, measurement] : ft_measurement) {
        auto& measurement_data = data[measurement_idx];

        // Force
        measurement_data.force[0] = measurement.force.x();
        measurement_data.force[1] = measurement.force.y();
        measurement_data.force[2] = measurement.force.z();

        // Center of pressure
        measurement_data.cop[0] = measurement.cop.x();
        measurement_data.cop[1] = measurement.cop.y();
        measurement_data.cop[2] = measurement.cop.z();

        // Torque (if available)
        measurement_data.has_torque = measurement.torque.has_value();
        if (measurement_data.has_torque) {
            const auto& torque = measurement.torque.value();
            measurement_data.torque[0] = torque.x();
            measurement_data.torque[1] = torque.y();
            measurement_data.torque[2] = torque.z();
        } else {
            measurement_data.torque[0] = 0.0;
            measurement_data.torque[1] = 0.0;
            measurement_data.torque[2] = 0.0;
        }

        measurement_idx++;
    }

    // Create and write the message
    mcap::Message message;
    message.channelId = 4;  // force_torque_measurement channel
    message.sequence = ft_sequence_++;
    message.logTime = data->timestamp;
    message.publishTime = data->timestamp;
    message.dataSize = total_size;
    message.data = buffer.data();

    auto status = writer_->write(message);
    if (status.code != mcap::StatusCode::Success) {
        std::cerr << "Failed to write force-torque measurement message" << std::endl;
    }
}

DebugLogger::~DebugLogger() {
    if (writer_)
        writer_->close();
    writer_.reset();
    file_writer_.reset();
}

}  // namespace serow
