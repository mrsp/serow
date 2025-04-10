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
#include "MeasurementLogger.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "ImuMeasurement_generated.h"
#include "KinematicMeasurement_generated.h"

namespace serow {

// Implementation class definition
class MeasurementLogger::Impl {
public:
    explicit Impl(const std::string& filename) {
        try {
            // First check if the file exists and is writable
            if (std::filesystem::exists(filename)) {
                if (!std::filesystem::is_regular_file(filename)) {
                    throw std::runtime_error("Log path exists but is not a regular file: " +
                                             filename);
                }
                // Check if we have write permissions
                auto perms = std::filesystem::status(filename).permissions();
                if ((perms & std::filesystem::perms::owner_write) == std::filesystem::perms::none) {
                    throw std::runtime_error("Log file is not writable: " + filename);
                }
            }

            // Open the file with error checking
            file_writer_ = std::make_unique<mcap::FileWriter>();
            auto open_status = file_writer_->open(filename);
            if (!open_status.ok()) {
                throw std::runtime_error("Failed to open MCAP file: " + open_status.message);
            }

            // Create the logger with ROS2 profile
            writer_ = std::make_unique<mcap::McapWriter>();
            // Configure MCAP options with explicit version
            mcap::McapWriterOptions options("ros2");

            writer_->open(*file_writer_, options);

            // Initialize schemas and channels
            initializeSchemas();
            initializeChannels();
        } catch (const std::exception& e) {
            std::cerr << "ProprioceptionLogger initialization error: " << e.what() << std::endl;
            throw;
        }
    }

    ~Impl() {
        try {
            if (writer_) {
                writer_->close();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error closing MCAP writer: " << e.what() << std::endl;
        }
    }

    // Split timestamp into seconds and nanoseconds
    inline void splitTimestamp(double timestamp, int64_t& sec, int32_t& nsec) const noexcept {
        sec = static_cast<int64_t>(timestamp);
        nsec = static_cast<int32_t>((timestamp - sec) * 1e9);
    }

    void log(const ImuMeasurement& imu_measurement) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(imu_measurement.timestamp, sec, nsec);

            // Create Time
            auto time = foxglove::Time(sec, nsec);

            // Create linear acceleration vector
            auto linear_acceleration = foxglove::CreateVector3(
                builder, imu_measurement.linear_acceleration.x(),
                imu_measurement.linear_acceleration.y(), imu_measurement.linear_acceleration.z());

            // Create angular velocity vector
            auto angular_velocity = foxglove::CreateVector3(
                builder, imu_measurement.angular_velocity.x(), imu_measurement.angular_velocity.y(),
                imu_measurement.angular_velocity.z());

            // Create orientation quaternion
            auto orientation = foxglove::CreateQuaternion(
                builder, imu_measurement.orientation.x(), imu_measurement.orientation.y(),
                imu_measurement.orientation.z(), imu_measurement.orientation.w());


            // Create linear acceleration covariance matrix
            auto linear_acceleration_cov = foxglove::CreateMatrix3(
                builder, imu_measurement.linear_acceleration_cov(0, 0), imu_measurement.linear_acceleration_cov(0, 1),
                imu_measurement.linear_acceleration_cov(0, 2), imu_measurement.linear_acceleration_cov(1, 0),
                imu_measurement.linear_acceleration_cov(1, 1), imu_measurement.linear_acceleration_cov(1, 2),
                imu_measurement.linear_acceleration_cov(2, 0), imu_measurement.linear_acceleration_cov(2, 1),
                imu_measurement.linear_acceleration_cov(2, 2));

            // Create angular velocity covariance matrix
            auto angular_velocity_cov = foxglove::CreateMatrix3(
                builder, imu_measurement.angular_velocity_cov(0, 0), imu_measurement.angular_velocity_cov(0, 1),
                imu_measurement.angular_velocity_cov(0, 2), imu_measurement.angular_velocity_cov(1, 0),
                imu_measurement.angular_velocity_cov(1, 1), imu_measurement.angular_velocity_cov(1, 2),
                imu_measurement.angular_velocity_cov(2, 0), imu_measurement.angular_velocity_cov(2, 1),
                imu_measurement.angular_velocity_cov(2, 2));

            // Create angular velocity bias covariance matrix
            auto angular_velocity_bias_cov = foxglove::CreateMatrix3(
                builder, imu_measurement.angular_velocity_bias_cov(0, 0), imu_measurement.angular_velocity_bias_cov(0, 1),
                imu_measurement.angular_velocity_bias_cov(0,     2), imu_measurement.angular_velocity_bias_cov(1, 0),
                imu_measurement.angular_velocity_bias_cov(1, 1), imu_measurement.angular_velocity_bias_cov(1, 2),
                imu_measurement.angular_velocity_bias_cov(2, 0), imu_measurement.angular_velocity_bias_cov(2, 1),
                imu_measurement.angular_velocity_bias_cov(2, 2));

            // Create linear acceleration bias covariance matrix
            auto linear_acceleration_bias_cov = foxglove::CreateMatrix3(
                builder, imu_measurement.linear_acceleration_bias_cov(0, 0), imu_measurement.linear_acceleration_bias_cov(0, 1),
                imu_measurement.linear_acceleration_bias_cov(0, 2), imu_measurement.linear_acceleration_bias_cov(1, 0),
                imu_measurement.linear_acceleration_bias_cov(1, 1), imu_measurement.linear_acceleration_bias_cov(1, 2),
                imu_measurement.linear_acceleration_bias_cov(2, 0), imu_measurement.linear_acceleration_bias_cov(2, 1),
                imu_measurement.linear_acceleration_bias_cov(2, 2));

            // Create angular acceleration vector
            auto angular_acceleration = foxglove::CreateVector3(
                builder, imu_measurement.angular_acceleration.x(), imu_measurement.angular_acceleration.y(),
                imu_measurement.angular_acceleration.z());

            // Create the root ImuState
            auto imu = foxglove::CreateImuMeasurement(builder,
                                                      &time,                // timestamp
                                                      linear_acceleration,  // linear_acceleration
                                                      angular_velocity,     // angular_velocity
                                                      orientation,          // orientation
                                                      linear_acceleration_cov, // linear_acceleration_cov
                                                      angular_velocity_cov, // angular_velocity_cov
                                                      angular_velocity_bias_cov, // angular_velocity_bias_cov
                                                      linear_acceleration_bias_cov, // linear_acceleration_bias_cov
                                                      angular_acceleration); // angular_acceleration

            // Finish the buffer
            builder.Finish(imu);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            if (!buffer || size == 0) {
                throw std::runtime_error("Invalid buffer state after Finish");
            }

            // Write the message
            writeMessage(1, imu_sequence_++, imu_measurement.timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging IMU measurement: " << e.what() << std::endl;
        }
    }

    void log(const KinematicMeasurement& kinematic_measurement) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            // Create entries for contacts_status
            std::vector<flatbuffers::Offset<foxglove::StringBoolEntry>> contacts_status_entries;
            for (const auto& [key, value] : kinematic_measurement.contacts_status) {
                auto key_offset = builder.CreateString(key);
                auto entry = foxglove::CreateStringBoolEntry(builder, key_offset, value);
                contacts_status_entries.push_back(entry);
            }
            auto contacts_status_vector = builder.CreateVector(contacts_status_entries);

            // Create entries for contacts_probability
            std::vector<flatbuffers::Offset<foxglove::StringDoubleEntry>> contacts_probability_entries;
            for (const auto& [key, value] : kinematic_measurement.contacts_probability) {
                auto key_offset = builder.CreateString(key);
                auto entry = foxglove::CreateStringDoubleEntry(builder, key_offset, value);
                contacts_probability_entries.push_back(entry);
            }
            auto contacts_probability_vector = builder.CreateVector(contacts_probability_entries);

            // Create entries for contacts_position
            std::vector<flatbuffers::Offset<foxglove::StringVector3Entry>> contacts_position_entries;
            for (const auto& [key, value] : kinematic_measurement.contacts_position) {
                auto key_offset = builder.CreateString(key);
                auto value_offset = foxglove::CreateVector3(builder, value.x(), value.y(), value.z());
                auto entry = foxglove::CreateStringVector3Entry(builder, key_offset, value_offset);
                contacts_position_entries.push_back(entry);
            }
            auto contacts_position_vector = builder.CreateVector(contacts_position_entries);

            // Create entries for base_to_foot_positions
            std::vector<flatbuffers::Offset<foxglove::StringVector3Entry>> base_to_foot_entries;
            for (const auto& [key, value] : kinematic_measurement.base_to_foot_positions) {
                auto key_offset = builder.CreateString(key);
                auto value_offset = foxglove::CreateVector3(builder, value.x(), value.y(), value.z());
                auto entry = foxglove::CreateStringVector3Entry(builder, key_offset, value_offset);
                base_to_foot_entries.push_back(entry);
            }
            auto base_to_foot_vector = builder.CreateVector(base_to_foot_entries);

            // Create entries for contacts_position_noise
            std::vector<flatbuffers::Offset<foxglove::StringMatrix3Entry>> contacts_position_noise_entries;
            for (const auto& [key, value] : kinematic_measurement.contacts_position_noise) {
                auto key_offset = builder.CreateString(key);
                auto matrix3_offset = foxglove::CreateMatrix3(builder,
                    value(0,0), value(0,1), value(0,2),
                    value(1,0), value(1,1), value(1,2),
                    value(2,0), value(2,1), value(2,2));
                auto entry = foxglove::CreateStringMatrix3Entry(builder, key_offset, matrix3_offset);
                contacts_position_noise_entries.push_back(entry);
            }
            auto contacts_position_noise_vector = builder.CreateVector(contacts_position_noise_entries);

            // Create entries for contacts_orientation if present
            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<foxglove::StringQuaternionEntry>>> contacts_orientation_vector;
            if (kinematic_measurement.contacts_orientation) {
                std::vector<flatbuffers::Offset<foxglove::StringQuaternionEntry>> contacts_orientation_entries;
                for (const auto& [key, value] : *kinematic_measurement.contacts_orientation) {
                    auto key_offset = builder.CreateString(key);
                    auto value_offset = foxglove::CreateQuaternion(builder, value.x(), value.y(), value.z(), value.w());
                    auto entry = foxglove::CreateStringQuaternionEntry(builder, key_offset, value_offset);
                    contacts_orientation_entries.push_back(entry);
                }
                contacts_orientation_vector = builder.CreateVector(contacts_orientation_entries);
            }

            // Create entries for contacts_orientation_noise if present
            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<foxglove::StringMatrix3Entry>>> contacts_orientation_noise_vector;
            if (kinematic_measurement.contacts_orientation_noise) {
                std::vector<flatbuffers::Offset<foxglove::StringMatrix3Entry>> contacts_orientation_noise_entries;
                for (const auto& [key, value] : *kinematic_measurement.contacts_orientation_noise) {
                    auto key_offset = builder.CreateString(key);
                    auto matrix3_offset = foxglove::CreateMatrix3(builder,
                        value(0,0), value(0,1), value(0,2),
                        value(1,0), value(1,1), value(1,2),
                        value(2,0), value(2,1), value(2,2));
                    auto entry = foxglove::CreateStringMatrix3Entry(builder, key_offset, matrix3_offset);
                    contacts_orientation_noise_entries.push_back(entry);
                }
                contacts_orientation_noise_vector = builder.CreateVector(contacts_orientation_noise_entries);
            }

            // Create Vector3 fields
            auto base_linear_velocity = foxglove::CreateVector3(builder, 
                kinematic_measurement.base_linear_velocity.x(),
                kinematic_measurement.base_linear_velocity.y(),
                kinematic_measurement.base_linear_velocity.z());

            auto com_angular_momentum_derivative = foxglove::CreateVector3(builder,
                kinematic_measurement.com_angular_momentum_derivative.x(),
                kinematic_measurement.com_angular_momentum_derivative.y(),
                kinematic_measurement.com_angular_momentum_derivative.z());

            auto com_position = foxglove::CreateVector3(builder,
                kinematic_measurement.com_position.x(),
                kinematic_measurement.com_position.y(),
                kinematic_measurement.com_position.z());

            auto com_linear_acceleration = foxglove::CreateVector3(builder,
                kinematic_measurement.com_linear_acceleration.x(),
                kinematic_measurement.com_linear_acceleration.y(),
                kinematic_measurement.com_linear_acceleration.z());

            // Create Quaternion fields
            auto base_orientation = foxglove::CreateQuaternion(builder,
                kinematic_measurement.base_orientation.x(),
                kinematic_measurement.base_orientation.y(),
                kinematic_measurement.base_orientation.z(),
                kinematic_measurement.base_orientation.w());

            // Create Matrix3 fields
            auto create_matrix3 = [&builder](const Eigen::Matrix3d& matrix) {
                return foxglove::CreateMatrix3(builder,
                    matrix(0,0), matrix(0,1), matrix(0,2),
                    matrix(1,0), matrix(1,1), matrix(1,2),
                    matrix(2,0), matrix(2,1), matrix(2,2));
            };

            auto base_linear_velocity_cov = create_matrix3(kinematic_measurement.base_linear_velocity_cov);
            auto base_orientation_cov = create_matrix3(kinematic_measurement.base_orientation_cov);
            auto position_slip_cov = create_matrix3(kinematic_measurement.position_slip_cov);
            auto orientation_slip_cov = create_matrix3(kinematic_measurement.orientation_slip_cov);
            auto position_cov = create_matrix3(kinematic_measurement.position_cov);
            auto orientation_cov = create_matrix3(kinematic_measurement.orientation_cov);
            auto com_position_process_cov = create_matrix3(kinematic_measurement.com_position_process_cov);
            auto com_linear_velocity_process_cov = create_matrix3(kinematic_measurement.com_linear_velocity_process_cov);
            auto external_forces_process_cov = create_matrix3(kinematic_measurement.external_forces_process_cov);
            auto com_position_cov = create_matrix3(kinematic_measurement.com_position_cov);
            auto com_linear_acceleration_cov = create_matrix3(kinematic_measurement.com_linear_acceleration_cov);

            // Create the main KinematicMeasurement object
            auto measurement = foxglove::CreateKinematicMeasurement(
                builder,
                kinematic_measurement.timestamp,
                base_linear_velocity,
                base_orientation,
                contacts_status_vector,
                contacts_probability_vector,
                contacts_position_vector,
                base_to_foot_vector,
                contacts_position_noise_vector,
                contacts_orientation_vector,
                contacts_orientation_noise_vector,
                com_angular_momentum_derivative,
                com_position,
                com_linear_acceleration,
                base_linear_velocity_cov,
                base_orientation_cov,
                position_slip_cov,
                orientation_slip_cov,
                position_cov,
                orientation_cov,
                com_position_process_cov,
                com_linear_velocity_process_cov,
                external_forces_process_cov,
                com_position_cov,
                com_linear_acceleration_cov
            );

            builder.Finish(measurement);

            // Write the message
            writeMessage(2, kin_sequence_++, kinematic_measurement.timestamp,
                        reinterpret_cast<const std::byte*>(builder.GetBufferPointer()),
                        builder.GetSize());
        }
        catch (const std::exception& e) {
            std::cerr << "Error logging kinematic measurement: " << e.what() << std::endl;
        }
    }

private:
    // Optimized message writing with reuse of message object
    void writeMessage(uint16_t channel_id, uint64_t sequence, double timestamp,
                      const std::byte* data, size_t data_size) noexcept {
        try {
            // Update message object with new values
            mcap::Message message;
            message.channelId = channel_id;
            message.sequence = sequence;

            // Ensure timestamp is in nanoseconds and consistent
            // Convert to nanoseconds using std::chrono for precision
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(timestamp));
            uint64_t ns_timestamp = ns.count();
            message.logTime = ns_timestamp;
            message.publishTime = ns_timestamp;  // Use same timestamp for both

            message.dataSize = data_size;
            message.data = data;

            // Protect the writer with a mutex
            std::lock_guard<std::mutex> lock(writer_mutex_);

            // Write the message without additional error checking
            auto status = writer_->write(message);
            if (status.code != mcap::StatusCode::Success) {
                std::cerr << "Failed to write message for channel " << channel_id << ": "
                          << status.message << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in writeMessage: " << e.what() << std::endl;
        }
    }

    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        schemas.reserve(2);
        schemas.push_back(createSchema("ImuMeasurement"));
        schemas.push_back(createSchema("KinematicMeasurement"));

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.reserve(2);
        channels.push_back(createChannel(1, "/imu"));
        channels.push_back(createChannel(2, "/kin"));

        for (auto& channel : channels) {
            writer_->addChannel(channel);
        }
    }

    mcap::Channel createChannel(uint16_t id, const std::string& topic) {
        mcap::Channel channel;
        channel.id = id;
        channel.topic = topic;
        channel.schemaId = id;
        channel.messageEncoding = "flatbuffer";
        return channel;
    }

    // Constants
    static constexpr size_t INITIAL_BUILDER_SIZE = 4096;

    // Sequence counters
    uint64_t kin_sequence_ = 0;
    uint64_t imu_sequence_ = 0;
    double last_timestamp_{-1.0};

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
    std::mutex writer_mutex_;
}; 

// Public interface implementation
MeasurementLogger::MeasurementLogger(const std::string& filename)
    : pimpl_(std::make_unique<Impl>(filename)) {}

MeasurementLogger::~MeasurementLogger() = default;

void MeasurementLogger::log(const ImuMeasurement& imu_measurement) {
    pimpl_->log(imu_measurement);
}

void MeasurementLogger::log(const KinematicMeasurement& kinematic_measurement) {
    pimpl_->log(kinematic_measurement);
}

}  // namespace serow
