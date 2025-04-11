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

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(kinematic_measurement.timestamp, sec, nsec);

            auto timestamp = foxglove::Time(sec, nsec);

            // Create all vector fields
            auto base_linear_velocity = foxglove::CreateVector3(builder, 
                kinematic_measurement.base_linear_velocity.x(),
                kinematic_measurement.base_linear_velocity.y(),
                kinematic_measurement.base_linear_velocity.z());

            auto base_orientation = foxglove::CreateQuaternion(builder,
                kinematic_measurement.base_orientation.w(),
                kinematic_measurement.base_orientation.x(),
                kinematic_measurement.base_orientation.y(),
                kinematic_measurement.base_orientation.z());

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

            // Create all matrix fields
            auto base_linear_velocity_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.base_linear_velocity_cov(0,0), kinematic_measurement.base_linear_velocity_cov(0,1), kinematic_measurement.base_linear_velocity_cov(0,2),
                kinematic_measurement.base_linear_velocity_cov(1,0), kinematic_measurement.base_linear_velocity_cov(1,1), kinematic_measurement.base_linear_velocity_cov(1,2),
                kinematic_measurement.base_linear_velocity_cov(2,0), kinematic_measurement.base_linear_velocity_cov(2,1), kinematic_measurement.base_linear_velocity_cov(2,2));

            auto base_orientation_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.base_orientation_cov(0,0), kinematic_measurement.base_orientation_cov(0,1), kinematic_measurement.base_orientation_cov(0,2),
                kinematic_measurement.base_orientation_cov(1,0), kinematic_measurement.base_orientation_cov(1,1), kinematic_measurement.base_orientation_cov(1,2),
                kinematic_measurement.base_orientation_cov(2,0), kinematic_measurement.base_orientation_cov(2,1), kinematic_measurement.base_orientation_cov(2,2));

            auto position_slip_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.position_slip_cov(0,0), kinematic_measurement.position_slip_cov(0,1), kinematic_measurement.position_slip_cov(0,2),
                kinematic_measurement.position_slip_cov(1,0), kinematic_measurement.position_slip_cov(1,1), kinematic_measurement.position_slip_cov(1,2),
                kinematic_measurement.position_slip_cov(2,0), kinematic_measurement.position_slip_cov(2,1), kinematic_measurement.position_slip_cov(2,2));

            auto orientation_slip_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.orientation_slip_cov(0,0), kinematic_measurement.orientation_slip_cov(0,1), kinematic_measurement.orientation_slip_cov(0,2),
                kinematic_measurement.orientation_slip_cov(1,0), kinematic_measurement.orientation_slip_cov(1,1), kinematic_measurement.orientation_slip_cov(1,2),
                kinematic_measurement.orientation_slip_cov(2,0), kinematic_measurement.orientation_slip_cov(2,1), kinematic_measurement.orientation_slip_cov(2,2));

            auto position_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.position_cov(0,0), kinematic_measurement.position_cov(0,1), kinematic_measurement.position_cov(0,2),
                kinematic_measurement.position_cov(1,0), kinematic_measurement.position_cov(1,1), kinematic_measurement.position_cov(1,2),
                kinematic_measurement.position_cov(2,0), kinematic_measurement.position_cov(2,1), kinematic_measurement.position_cov(2,2));

            auto orientation_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.orientation_cov(0,0), kinematic_measurement.orientation_cov(0,1), kinematic_measurement.orientation_cov(0,2),
                kinematic_measurement.orientation_cov(1,0), kinematic_measurement.orientation_cov(1,1), kinematic_measurement.orientation_cov(1,2),
                kinematic_measurement.orientation_cov(2,0), kinematic_measurement.orientation_cov(2,1), kinematic_measurement.orientation_cov(2,2));

            auto com_position_process_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.com_position_process_cov(0,0), kinematic_measurement.com_position_process_cov(0,1), kinematic_measurement.com_position_process_cov(0,2),
                kinematic_measurement.com_position_process_cov(1,0), kinematic_measurement.com_position_process_cov(1,1), kinematic_measurement.com_position_process_cov(1,2),
                kinematic_measurement.com_position_process_cov(2,0), kinematic_measurement.com_position_process_cov(2,1), kinematic_measurement.com_position_process_cov(2,2));

            auto com_linear_velocity_process_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.com_linear_velocity_process_cov(0,0), kinematic_measurement.com_linear_velocity_process_cov(0,1), kinematic_measurement.com_linear_velocity_process_cov(0,2),
                kinematic_measurement.com_linear_velocity_process_cov(1,0), kinematic_measurement.com_linear_velocity_process_cov(1,1), kinematic_measurement.com_linear_velocity_process_cov(1,2),
                kinematic_measurement.com_linear_velocity_process_cov(2,0), kinematic_measurement.com_linear_velocity_process_cov(2,1), kinematic_measurement.com_linear_velocity_process_cov(2,2));

            auto external_forces_process_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.external_forces_process_cov(0,0), kinematic_measurement.external_forces_process_cov(0,1), kinematic_measurement.external_forces_process_cov(0,2),
                kinematic_measurement.external_forces_process_cov(1,0), kinematic_measurement.external_forces_process_cov(1,1), kinematic_measurement.external_forces_process_cov(1,2),
                kinematic_measurement.external_forces_process_cov(2,0), kinematic_measurement.external_forces_process_cov(2,1), kinematic_measurement.external_forces_process_cov(2,2));

            auto com_position_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.com_position_cov(0,0), kinematic_measurement.com_position_cov(0,1), kinematic_measurement.com_position_cov(0,2),
                kinematic_measurement.com_position_cov(1,0), kinematic_measurement.com_position_cov(1,1), kinematic_measurement.com_position_cov(1,2),
                kinematic_measurement.com_position_cov(2,0), kinematic_measurement.com_position_cov(2,1), kinematic_measurement.com_position_cov(2,2));

            auto com_linear_acceleration_cov = foxglove::CreateMatrix3(builder,
                kinematic_measurement.com_linear_acceleration_cov(0,0), kinematic_measurement.com_linear_acceleration_cov(0,1), kinematic_measurement.com_linear_acceleration_cov(0,2),
                kinematic_measurement.com_linear_acceleration_cov(1,0), kinematic_measurement.com_linear_acceleration_cov(1,1), kinematic_measurement.com_linear_acceleration_cov(1,2),
                kinematic_measurement.com_linear_acceleration_cov(2,0), kinematic_measurement.com_linear_acceleration_cov(2,1), kinematic_measurement.com_linear_acceleration_cov(2,2));

            // Create contact names vector
            std::vector<flatbuffers::Offset<flatbuffers::String>> contact_names_vec;
            for (const auto& [frame_name, _] : kinematic_measurement.contacts_position) {
                contact_names_vec.push_back(builder.CreateString(frame_name));
            }
            auto contact_names = builder.CreateVector(contact_names_vec);

            // Create contact status vector
            std::vector<bool> contacts_status_vec;
            for (const auto& [_, status] : kinematic_measurement.contacts_status) {
                contacts_status_vec.push_back(status);
            }
            auto contacts_status = builder.CreateVector(contacts_status_vec);

            // Create contact probability vector
            std::vector<double> contacts_probability_vec;
            for (const auto& [_, probability] : kinematic_measurement.contacts_probability) {
                contacts_probability_vec.push_back(probability);
            }
            auto contacts_probability = builder.CreateVector(contacts_probability_vec);

            // Create contact position vector
            std::vector<flatbuffers::Offset<foxglove::Vector3>> contacts_position_vec;
            for (const auto& [_, position] : kinematic_measurement.contacts_position) {
                contacts_position_vec.push_back(foxglove::CreateVector3(builder, position.x(), position.y(), position.z()));
            }
            auto contacts_position = builder.CreateVector(contacts_position_vec);

            // Create base to foot position vector
            std::vector<flatbuffers::Offset<foxglove::Vector3>> base_to_foot_vec;
            for (const auto& [_, position] : kinematic_measurement.base_to_foot_positions) {
                base_to_foot_vec.push_back(foxglove::CreateVector3(builder, position.x(), position.y(), position.z()));
            }
            auto base_to_foot = builder.CreateVector(base_to_foot_vec);

            // Create contact position noise vector
            std::vector<flatbuffers::Offset<foxglove::Matrix3>> contacts_position_noise_vec;
            for (const auto& [_, noise] : kinematic_measurement.contacts_position_noise) {
                contacts_position_noise_vec.push_back(foxglove::CreateMatrix3(builder,
                    noise(0,0), noise(0,1), noise(0,2),
                    noise(1,0), noise(1,1), noise(1,2),
                    noise(2,0), noise(2,1), noise(2,2)));
            }
            auto contacts_position_noise = builder.CreateVector(contacts_position_noise_vec);

            // Create contact orientation vector if present
            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<foxglove::Quaternion>>> contacts_orientation;
            if (kinematic_measurement.contacts_orientation) {
                std::vector<flatbuffers::Offset<foxglove::Quaternion>> contacts_orientation_vec;
                for (const auto& [_, orientation] : *kinematic_measurement.contacts_orientation) {
                    contacts_orientation_vec.push_back(foxglove::CreateQuaternion(builder, 
                        orientation.w(), orientation.x(), orientation.y(), orientation.z()));
                }
                contacts_orientation = builder.CreateVector(contacts_orientation_vec);
            }

            // Create contact orientation noise vector if present
            flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<foxglove::Matrix3>>> contacts_orientation_noise;
            if (kinematic_measurement.contacts_orientation_noise) {
                std::vector<flatbuffers::Offset<foxglove::Matrix3>> contacts_orientation_noise_vec;
                for (const auto& [_, noise] : *kinematic_measurement.contacts_orientation_noise) {
                    contacts_orientation_noise_vec.push_back(foxglove::CreateMatrix3(builder,
                        noise(0,0), noise(0,1), noise(0,2),
                        noise(1,0), noise(1,1), noise(1,2),
                        noise(2,0), noise(2,1), noise(2,2)));
                }
                contacts_orientation_noise = builder.CreateVector(contacts_orientation_noise_vec);
            }

            // Create the KinematicMeasurement message
            auto measurement = foxglove::CreateKinematicMeasurement(
                builder,
                &timestamp,
                base_linear_velocity,
                base_orientation,
                contact_names,
                contacts_status,
                contacts_probability,
                contacts_position,
                base_to_foot,
                contacts_position_noise,
                contacts_orientation,
                contacts_orientation_noise,
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

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            writeMessage(2, kin_sequence_++, kinematic_measurement.timestamp,
                        reinterpret_cast<const std::byte*>(buffer), size);
            last_timestamp_ = kinematic_measurement.timestamp;
        } catch (const std::exception& e) {
            std::cerr << "Error logging Kinematic Measurement: " << e.what() << std::endl;
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
