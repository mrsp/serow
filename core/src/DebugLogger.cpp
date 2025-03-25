#include "DebugLogger.hpp"

struct BinaryVector3d {
    double x;
    double y;
    double z;
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
    
    // Define a simple schema for our binary format
    mcap::Schema schema;
    schema.name = "BinaryVector3d";
    schema.encoding = "binary";
    // Define the schema data as a binary format
    std::string schema_data = R"({
        "fields": [
            {"name": "x", "type": "float64"},
            {"name": "y", "type": "float64"},
            {"name": "z", "type": "float64"}
        ]
    })";
    schema.data = mcap::ByteArray(reinterpret_cast<const std::byte*>(schema_data.data()),
                                 reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    writer_->addSchema(schema);
}

void DebugLogger::log(const BaseState& base_state) {
    // Create a message with the schema and data
    mcap::Message message;
    message.channelId = 0; // We'll need to add a channel first
    message.sequence = 0;
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

void DebugLogger::log(const ContactState& contact_state) {}

void DebugLogger::log(const ImuMeasurement& imu_measurement) {}

void DebugLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {}

void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurement) {}

DebugLogger::~DebugLogger() {
    if (writer_) {
        writer_->close();
    }
    writer_.reset();
    file_writer_.reset();
}

}  // namespace serow
