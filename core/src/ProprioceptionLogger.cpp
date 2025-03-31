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
#include "ProprioceptionLogger.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include "BaseState_generated.h"
#include "CentroidalState_generated.h"
#include "ContactState_generated.h"
#include "FrameTransform_generated.h"
#include "FrameTransforms_generated.h"
#include "ImuState_generated.h"
#include "Time_generated.h"
#include "Vector3_generated.h"

namespace serow {

// Implementation class definition
class ProprioceptionLogger::Impl {
public:
    explicit Impl(const std::string& filename) {
        try {
            // Open the file with error checking
            file_writer_ = std::make_unique<mcap::FileWriter>();
            auto open_status = file_writer_->open(filename);
            if (!open_status.ok()) {
                throw std::runtime_error("Failed to open MCAP file: " + open_status.message);
            }

            // Create the logger with ROS2 profile
            writer_ = std::make_unique<mcap::McapWriter>();
            mcap::McapWriterOptions options("");
            writer_->open(*file_writer_, options);

            // Initialize schemas and channels
            initializeSchemas();
            initializeChannels();
            
            // Pre-allocate builders for frequent operations
            builders_.reserve(6);
            for (int i = 0; i < 6; i++) {
                builders_.emplace_back(flatbuffers::FlatBufferBuilder{INITIAL_BUILDER_SIZE});
            }
            
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

    // Fast timestamp conversion
    inline uint64_t convertToNanoseconds(double timestamp) const noexcept {
        return static_cast<uint64_t>(timestamp * 1e9);
    }
    
    // Split timestamp into seconds and nanoseconds
    inline void splitTimestamp(double timestamp, int64_t& sec, int32_t& nsec) const noexcept {
        sec = static_cast<int64_t>(timestamp);
        nsec = static_cast<int32_t>((timestamp - sec) * 1e9);
    }

    void log(const ImuMeasurement& imu_measurement) {
        try {
            auto& builder = builders_[0];
            builder.Reset();

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

            // Create the root ImuState
            auto imu_state = foxglove::CreateImuState(builder,
                                                      &time,                // timestamp
                                                      linear_acceleration,  // linear_acceleration
                                                      angular_velocity,     // angular_velocity
                                                      orientation);         // orientation

            // Finish the buffer
            builder.Finish(imu_state);

            // Write the message
            writeMessage(1, imu_sequence_++, imu_measurement.timestamp,
                         reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), 
                         builder.GetSize());
        } catch (const std::exception& e) {
            std::cerr << "Error logging IMU state: " << e.what() << std::endl;
        }
    }


    void log(const ContactState& contact_state) {
        try {
            // Create FlatBuffers builder
            auto& builder = builders_[1];
            builder.Reset();

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(contact_state.timestamp, sec, nsec);

            // Create Time
            auto time = foxglove::Time(sec, nsec);

            // Preallocate vectors for better performance
            const size_t contact_count = contact_state.contacts_status.size();
            std::vector<flatbuffers::Offset<flatbuffers::String>> contact_names;
            std::vector<flatbuffers::Offset<foxglove::Contact>> contacts;
            contact_names.reserve(contact_count);
            contacts.reserve(contact_count);

            // Create contact names vector
            for (const auto& [contact_name, _] : contact_state.contacts_status) {
                contact_names.push_back(builder.CreateString(contact_name));
            }
            auto contact_names_vec = builder.CreateVector(contact_names);

            // Create contacts vector
            for (const auto& [contact_name, status] : contact_state.contacts_status) {
                // Create force vector
                auto force = foxglove::CreateVector3(
                    builder, contact_state.contacts_force.at(contact_name).x(),
                    contact_state.contacts_force.at(contact_name).y(),
                    contact_state.contacts_force.at(contact_name).z());

                // Create torque vector if available
                flatbuffers::Offset<foxglove::Vector3> torque = 0;
                if (contact_state.contacts_torque.has_value()) {
                    const auto& torque_map = contact_state.contacts_torque.value();
                    if (torque_map.count(contact_name) > 0) {
                        torque = foxglove::CreateVector3(builder, torque_map.at(contact_name).x(),
                                                         torque_map.at(contact_name).y(),
                                                         torque_map.at(contact_name).z());
                    }
                }

                // Create contact
                auto contact = foxglove::CreateContact(
                    builder,
                    status,                                               // status
                    contact_state.contacts_probability.at(contact_name),  // probability
                    force,                                                // force
                    torque);  // torque (will be null if not set)

                contacts.push_back(contact);
            }
            auto contacts_vec = builder.CreateVector(contacts);

            // Create the root ContactState
            auto contact_state_fb =
                foxglove::CreateContactState(builder,
                                             &time,              // timestamp
                                             contact_names_vec,  // contact_names
                                             contacts_vec);      // contacts

            // Finish the buffer
            builder.Finish(contact_state_fb);

            // Write the message
            writeMessage(2, contact_sequence_++, contact_state.timestamp,
                         reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), 
                         builder.GetSize());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Contact State: " << e.what() << std::endl;
        }
    }

    void log(const CentroidalState& centroidal_state) {
        try {
            auto& builder = builders_[2];
            builder.Reset();

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(centroidal_state.timestamp, sec, nsec);

            auto timestamp = foxglove::Time(sec, nsec);
            auto com_position = foxglove::CreateVector3(builder, centroidal_state.com_position.x(),
                                                        centroidal_state.com_position.y(),
                                                        centroidal_state.com_position.z());
            auto com_linear_velocity = foxglove::CreateVector3(
                builder, centroidal_state.com_linear_velocity.x(),
                centroidal_state.com_linear_velocity.y(), centroidal_state.com_linear_velocity.z());
            auto external_forces = foxglove::CreateVector3(
                builder, centroidal_state.external_forces.x(), centroidal_state.external_forces.y(),
                centroidal_state.external_forces.z());
            auto cop_position = foxglove::CreateVector3(builder, centroidal_state.cop_position.x(),
                                                        centroidal_state.cop_position.y(),
                                                        centroidal_state.cop_position.z());
            auto com_linear_acceleration =
                foxglove::CreateVector3(builder, centroidal_state.com_linear_acceleration.x(),
                                        centroidal_state.com_linear_acceleration.y(),
                                        centroidal_state.com_linear_acceleration.z());
            auto angular_momentum = foxglove::CreateVector3(
                builder, centroidal_state.angular_momentum.x(),
                centroidal_state.angular_momentum.y(), centroidal_state.angular_momentum.z());
            auto angular_momentum_derivative =
                foxglove::CreateVector3(builder, centroidal_state.angular_momentum_derivative.x(),
                                        centroidal_state.angular_momentum_derivative.y(),
                                        centroidal_state.angular_momentum_derivative.z());

            auto centroidal_state_fb = foxglove::CreateCentroidalState(
                builder, &timestamp, com_position, com_linear_velocity, external_forces,
                cop_position, com_linear_acceleration, angular_momentum,
                angular_momentum_derivative);

            builder.Finish(centroidal_state_fb);

            writeMessage(3, centroidal_sequence_++, centroidal_state.timestamp,
                         reinterpret_cast<const std::byte*>(builder.GetBufferPointer()),
                         builder.GetSize());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Centroidal State: " << e.what() << std::endl;
        }
    }

    void log(const BaseState& base_state) {
        try {
            auto& builder = builders_[3];
            builder.Reset();
            
            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(base_state.timestamp, sec, nsec);

            auto timestamp = foxglove::Time(sec, nsec);

            auto base_position =
                foxglove::CreateVector3(builder, base_state.base_position.x(),
                                        base_state.base_position.y(), base_state.base_position.z());

            auto base_orientation = foxglove::CreateQuaternion(
                builder, base_state.base_orientation.w(), base_state.base_orientation.x(),
                base_state.base_orientation.y(), base_state.base_orientation.z());

            auto base_linear_velocity = foxglove::CreateVector3(
                builder, base_state.base_linear_velocity.x(), base_state.base_linear_velocity.y(),
                base_state.base_linear_velocity.z());

            auto base_angular_velocity = foxglove::CreateVector3(
                builder, base_state.base_angular_velocity.x(), base_state.base_angular_velocity.y(),
                base_state.base_angular_velocity.z());

            auto base_linear_acceleration = foxglove::CreateVector3(
                builder, base_state.base_linear_acceleration.x(),
                base_state.base_linear_acceleration.y(), base_state.base_linear_acceleration.z());

            auto base_angular_acceleration = foxglove::CreateVector3(
                builder, base_state.base_angular_acceleration.x(),
                base_state.base_angular_acceleration.y(), base_state.base_angular_acceleration.z());

            auto imu_angular_velocity_bias = foxglove::CreateVector3(
                builder, base_state.imu_angular_velocity_bias.x(),
                base_state.imu_angular_velocity_bias.y(), base_state.imu_angular_velocity_bias.z());

            auto imu_linear_acceleration_bias =
                foxglove::CreateVector3(builder, base_state.imu_linear_acceleration_bias.x(),
                                        base_state.imu_linear_acceleration_bias.y(),
                                        base_state.imu_linear_acceleration_bias.z());

            auto base_state_fb = foxglove::CreateBaseState(
                builder, &timestamp, base_position, base_orientation, base_linear_velocity,
                base_angular_velocity, base_linear_acceleration, base_angular_acceleration,
                imu_angular_velocity_bias, imu_linear_acceleration_bias);

            builder.Finish(base_state_fb);
            
            writeMessage(4, base_sequence_++, base_state.timestamp,
                         reinterpret_cast<const std::byte*>(builder.GetBufferPointer()),
                         builder.GetSize());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Base State: " << e.what() << std::endl;
        }
    }

    void log(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
             double timestamp) {
        try {
            auto& builder = builders_[4];
            builder.Reset();  

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            // Create parent and child frame ids
            auto parent_frame = builder.CreateString("world");
            auto child_frame = builder.CreateString("base");
            
            // Create timestamp
            auto timestamp_fb = foxglove::Time(sec, nsec);
            
            // Create translation
            auto translation =
                foxglove::CreateVector3(builder, position.x(), position.y(), position.z());
            // Create rotation
            auto rotation = foxglove::CreateQuaternion(builder, orientation.x(), orientation.y(),
                                                       orientation.z(), orientation.w());
            // Create the root message
            foxglove::FrameTransformBuilder tf_builder(builder);
            tf_builder.add_translation(translation);
            tf_builder.add_rotation(rotation);
            tf_builder.add_timestamp(&timestamp_fb);
            tf_builder.add_parent_frame_id(std::move(parent_frame));
            tf_builder.add_child_frame_id(std::move(child_frame));
            auto tf = tf_builder.Finish();
            builder.Finish(tf);

            // Get the serialized data
            writeMessage(5, tf_sequence_++, timestamp, 
                         reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), 
                         builder.GetSize());
        } catch (const std::exception& e) {
            std::cerr << "Error logging basetransform: " << e.what() << std::endl;
        }
    }

    void log(const std::map<std::string, Eigen::Vector3d>& positions,
             const std::map<std::string, Eigen::Quaterniond>& orientations, double timestamp) {
        try {
            auto& builder = builders_[5];
            builder.Reset();

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            // Create timestamp
            auto timestamp_fb = foxglove::Time(sec, nsec);

            // Create transforms vector - preallocate capacity
            std::vector<flatbuffers::Offset<foxglove::FrameTransform>> transforms_vector;
            transforms_vector.reserve(positions.size());

            // Reuse cached world_frame string
            auto parent_frame = builder.CreateString("world");

            for (const auto& [frame_id, position] : positions) {
                // Create frame_id strings
                auto child_frame = builder.CreateString(frame_id);

                // Create translation
                auto translation =
                    foxglove::CreateVector3(builder, position.x(), position.y(), position.z());

                // Create rotation
                const auto& quaternion = orientations.at(frame_id);
                auto rotation = foxglove::CreateQuaternion(builder, quaternion.x(), quaternion.y(),
                                                           quaternion.z(), quaternion.w());

                // Create transform
                auto transform = foxglove::CreateFrameTransform(
                    builder, &timestamp_fb, parent_frame, child_frame, translation, rotation);

                transforms_vector.push_back(transform);
            }

            // Create transforms vector
            auto transforms = builder.CreateVector(transforms_vector);

            // Create the root message
            foxglove::FrameTransformsBuilder tf_builder(builder);
            tf_builder.add_transforms(transforms);
            auto tf_array = tf_builder.Finish();

            // Finish the buffer
            builder.Finish(tf_array);

            // Get the serialized data
            writeMessage(6, leg_tf_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(builder.GetBufferPointer()), 
                         builder.GetSize());
        } catch (const std::exception& e) {
            std::cerr << "Error logging feet transforms: " << e.what() << std::endl;
        }
    }

private:
    // Optimized message writing with reuse of message object
    void writeMessage(uint16_t channel_id, uint64_t sequence, double timestamp,
                      const std::byte* data, size_t data_size) noexcept {
        // Update message object with new values
        mcap::Message message;
        message.channelId = channel_id;
        message.sequence = sequence;
        message.logTime = convertToNanoseconds(timestamp);
        message.publishTime = message.logTime;
        message.dataSize = data_size;
        message.data = data;

        // Write the message without additional error checking
        auto status = writer_->write(message);
        if (status.code != mcap::StatusCode::Success) {
            std::cerr << "Failed to write message for channel " << channel_id 
                      << ": " << status.message << std::endl;
        }
    }

    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        schemas.reserve(6);
        schemas.push_back(createSchema("ImuState"));
        schemas.push_back(createSchema("ContactState"));
        schemas.push_back(createSchema("CentroidalState"));
        schemas.push_back(createSchema("BaseState"));
        schemas.push_back(createSchema("FrameTransform"));
        schemas.push_back(createSchema("FrameTransforms"));

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.reserve(6);
        channels.push_back(createChannel(1, "/imu_state"));
        channels.push_back(createChannel(2, "/contact_state"));
        channels.push_back(createChannel(3, "/centroidal_state"));
        channels.push_back(createChannel(4, "/base_state"));
        channels.push_back(createChannel(5, "/odom"));
        channels.push_back(createChannel(6, "/legs"));

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
    static constexpr size_t INITIAL_BUILDER_SIZE = 1024;
    
    // Sequence counters
    uint64_t base_sequence_ = 0;
    uint64_t centroidal_sequence_ = 0;
    uint64_t contact_sequence_ = 0;
    uint64_t imu_sequence_ = 0;
    uint64_t tf_sequence_ = 0;
    uint64_t leg_tf_sequence_ = 0;

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
    
    
    // Pool of reusable flatbuffer builders
    std::vector<flatbuffers::FlatBufferBuilder> builders_;
};  // namespace serow

// Public interface implementation
ProprioceptionLogger::ProprioceptionLogger(const std::string& filename)
    : pimpl_(std::make_unique<Impl>(filename)) {}

ProprioceptionLogger::~ProprioceptionLogger() = default;

void ProprioceptionLogger::log(const BaseState& base_state) {
    pimpl_->log(base_state);
}

void ProprioceptionLogger::log(const CentroidalState& centroidal_state) {
    pimpl_->log(centroidal_state);
}

void ProprioceptionLogger::log(const ImuMeasurement& imu_measurement) {
    pimpl_->log(imu_measurement);
}

void ProprioceptionLogger::log(const ContactState& contact_state) {
    pimpl_->log(contact_state);
}

void ProprioceptionLogger::log(const Eigen::Vector3d& position,
                               const Eigen::Quaterniond& orientation, double timestamp) {
    pimpl_->log(position, orientation, timestamp);
}

void ProprioceptionLogger::log(const std::map<std::string, Eigen::Vector3d>& positions,
                               const std::map<std::string, Eigen::Quaterniond>& orientations,
                               double timestamp) {
    pimpl_->log(positions, orientations, timestamp);
}

}  // namespace serow
