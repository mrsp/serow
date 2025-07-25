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
#include "ImuMeasurement_generated.h"
#include "JointState_generated.h"
#include "Time_generated.h"
#include "Vector3_generated.h"

namespace serow {

// Implementation class definition
class ProprioceptionLogger::Impl {
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

    void setStartTime(double timestamp) {
        start_time_ = timestamp;
    }

    // Split timestamp into seconds and nanoseconds
    inline void splitTimestamp(double timestamp, int64_t& sec, int32_t& nsec) const noexcept {
        sec = static_cast<int64_t>(timestamp);
        nsec = static_cast<int32_t>((timestamp - sec) * 1e9);
    }

    void log(const ImuMeasurement& imu_measurement) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = imu_measurement.timestamp;
            }
            const double timestamp = imu_measurement.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cerr << "Timestamp is negative: " << timestamp << std::endl;
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

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

            // Create the root ImuMeasurement
            auto imu = foxglove::CreateImuMeasurement(builder,
                                                      &time,                // timestamp
                                                      linear_acceleration,  // linear_acceleration
                                                      angular_velocity,     // angular_velocity
                                                      orientation);         // orientation

            // Finish the buffer
            builder.Finish(imu);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            if (!buffer || size == 0) {
                throw std::runtime_error("Invalid buffer state after Finish");
            }

            // Write the message
            writeMessage(1, imu_sequence_++, timestamp, reinterpret_cast<const std::byte*>(buffer),
                         size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging IMU measurement: " << e.what() << std::endl;
        }
    }

    void log(const ContactState& contact_state) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = contact_state.timestamp;
            }
            const double timestamp = contact_state.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cerr << "Timestamp is negative: " << timestamp << std::endl;
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

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

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            // Write the message
            writeMessage(2, contact_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging Contact State: " << e.what() << std::endl;
        }
    }

    void log(const CentroidalState& centroidal_state) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = centroidal_state.timestamp;
            }
            const double timestamp = centroidal_state.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cerr << "Timestamp is negative: " << timestamp << std::endl;
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            auto time = foxglove::Time(sec, nsec);
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
                builder, &time, com_position, com_linear_velocity, external_forces, cop_position,
                com_linear_acceleration, angular_momentum, angular_momentum_derivative);

            builder.Finish(centroidal_state_fb);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            writeMessage(3, centroidal_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging Centroidal State: " << e.what() << std::endl;
        }
    }

    void log(const BaseState& base_state) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = base_state.timestamp;
            }
            const double timestamp = base_state.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cerr << "Timestamp is negative: " << timestamp << std::endl;
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            auto time = foxglove::Time(sec, nsec);

            // Create contact names vector
            std::vector<flatbuffers::Offset<flatbuffers::String>> contact_names_vec;
            for (const auto& [frame_name, _] : base_state.contacts_position) {
                contact_names_vec.push_back(builder.CreateString(frame_name));
            }
            auto contact_names = builder.CreateVector(contact_names_vec);

            // Create all vector fields
            auto base_position =
                foxglove::CreateVector3(builder, base_state.base_position.x(),
                                        base_state.base_position.y(), base_state.base_position.z());

            auto base_orientation = foxglove::CreateQuaternion(
                builder, base_state.base_orientation.x(), base_state.base_orientation.y(),
                base_state.base_orientation.z(), base_state.base_orientation.w());

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

            auto imu_linear_acceleration_bias =
                foxglove::CreateVector3(builder, base_state.imu_linear_acceleration_bias.x(),
                                        base_state.imu_linear_acceleration_bias.y(),
                                        base_state.imu_linear_acceleration_bias.z());

            auto imu_angular_velocity_bias = foxglove::CreateVector3(
                builder, base_state.imu_angular_velocity_bias.x(),
                base_state.imu_angular_velocity_bias.y(), base_state.imu_angular_velocity_bias.z());

            // Create all matrix fields
            auto base_position_cov = foxglove::CreateMatrix3(
                builder, base_state.base_position_cov(0, 0), base_state.base_position_cov(0, 1),
                base_state.base_position_cov(0, 2), base_state.base_position_cov(1, 0),
                base_state.base_position_cov(1, 1), base_state.base_position_cov(1, 2),
                base_state.base_position_cov(2, 0), base_state.base_position_cov(2, 1),
                base_state.base_position_cov(2, 2));

            auto base_orientation_cov = foxglove::CreateMatrix3(
                builder, base_state.base_orientation_cov(0, 0),
                base_state.base_orientation_cov(0, 1), base_state.base_orientation_cov(0, 2),
                base_state.base_orientation_cov(1, 0), base_state.base_orientation_cov(1, 1),
                base_state.base_orientation_cov(1, 2), base_state.base_orientation_cov(2, 0),
                base_state.base_orientation_cov(2, 1), base_state.base_orientation_cov(2, 2));

            auto base_linear_velocity_cov =
                foxglove::CreateMatrix3(builder, base_state.base_linear_velocity_cov(0, 0),
                                        base_state.base_linear_velocity_cov(0, 1),
                                        base_state.base_linear_velocity_cov(0, 2),
                                        base_state.base_linear_velocity_cov(1, 0),
                                        base_state.base_linear_velocity_cov(1, 1),
                                        base_state.base_linear_velocity_cov(1, 2),
                                        base_state.base_linear_velocity_cov(2, 0),
                                        base_state.base_linear_velocity_cov(2, 1),
                                        base_state.base_linear_velocity_cov(2, 2));

            auto base_angular_velocity_cov =
                foxglove::CreateMatrix3(builder, base_state.base_angular_velocity_cov(0, 0),
                                        base_state.base_angular_velocity_cov(0, 1),
                                        base_state.base_angular_velocity_cov(0, 2),
                                        base_state.base_angular_velocity_cov(1, 0),
                                        base_state.base_angular_velocity_cov(1, 1),
                                        base_state.base_angular_velocity_cov(1, 2),
                                        base_state.base_angular_velocity_cov(2, 0),
                                        base_state.base_angular_velocity_cov(2, 1),
                                        base_state.base_angular_velocity_cov(2, 2));

            auto imu_linear_acceleration_bias_cov =
                foxglove::CreateMatrix3(builder, base_state.imu_linear_acceleration_bias_cov(0, 0),
                                        base_state.imu_linear_acceleration_bias_cov(0, 1),
                                        base_state.imu_linear_acceleration_bias_cov(0, 2),
                                        base_state.imu_linear_acceleration_bias_cov(1, 0),
                                        base_state.imu_linear_acceleration_bias_cov(1, 1),
                                        base_state.imu_linear_acceleration_bias_cov(1, 2),
                                        base_state.imu_linear_acceleration_bias_cov(2, 0),
                                        base_state.imu_linear_acceleration_bias_cov(2, 1),
                                        base_state.imu_linear_acceleration_bias_cov(2, 2));

            auto imu_angular_velocity_bias_cov =
                foxglove::CreateMatrix3(builder, base_state.imu_angular_velocity_bias_cov(0, 0),
                                        base_state.imu_angular_velocity_bias_cov(0, 1),
                                        base_state.imu_angular_velocity_bias_cov(0, 2),
                                        base_state.imu_angular_velocity_bias_cov(1, 0),
                                        base_state.imu_angular_velocity_bias_cov(1, 1),
                                        base_state.imu_angular_velocity_bias_cov(1, 2),
                                        base_state.imu_angular_velocity_bias_cov(2, 0),
                                        base_state.imu_angular_velocity_bias_cov(2, 1),
                                        base_state.imu_angular_velocity_bias_cov(2, 2));

            // Create contact position vectors
            std::vector<flatbuffers::Offset<foxglove::Vector3>> contacts_position_vec;
            for (const auto& [_, position] : base_state.contacts_position) {
                contacts_position_vec.push_back(
                    foxglove::CreateVector3(builder, position.x(), position.y(), position.z()));
            }
            auto contacts_position = builder.CreateVector(contacts_position_vec);

            // Create contact orientation vectors
            std::vector<flatbuffers::Offset<foxglove::Quaternion>> contacts_orientation_vec;
            if (base_state.contacts_orientation) {
                for (const auto& [_, orientation] : *base_state.contacts_orientation) {
                    contacts_orientation_vec.push_back(
                        foxglove::CreateQuaternion(builder, orientation.x(), orientation.y(),
                                                   orientation.z(), orientation.w()));
                }
            }
            auto contacts_orientation = builder.CreateVector(contacts_orientation_vec);

            // Create contact position covariance vectors
            std::vector<flatbuffers::Offset<foxglove::Matrix3>> contacts_position_cov_vec;
            for (const auto& [_, cov] : base_state.contacts_position_cov) {
                contacts_position_cov_vec.push_back(
                    foxglove::CreateMatrix3(builder, cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 0),
                                            cov(1, 1), cov(1, 2), cov(2, 0), cov(2, 1), cov(2, 2)));
            }
            auto contacts_position_cov = builder.CreateVector(contacts_position_cov_vec);

            // Create contact orientation covariance vectors
            std::vector<flatbuffers::Offset<foxglove::Matrix3>> contacts_orientation_cov_vec;
            if (base_state.contacts_orientation_cov) {
                for (const auto& [_, cov] : *base_state.contacts_orientation_cov) {
                    contacts_orientation_cov_vec.push_back(foxglove::CreateMatrix3(
                        builder, cov(0, 0), cov(0, 1), cov(0, 2), cov(1, 0), cov(1, 1), cov(1, 2),
                        cov(2, 0), cov(2, 1), cov(2, 2)));
                }
            }
            auto contacts_orientation_cov = builder.CreateVector(contacts_orientation_cov_vec);

            // Create feet position vectors
            std::vector<flatbuffers::Offset<foxglove::Vector3>> feet_position_vec;
            for (const auto& [_, position] : base_state.feet_position) {
                feet_position_vec.push_back(
                    foxglove::CreateVector3(builder, position.x(), position.y(), position.z()));
            }
            auto feet_position = builder.CreateVector(feet_position_vec);

            // Create feet orientation vectors
            std::vector<flatbuffers::Offset<foxglove::Quaternion>> feet_orientation_vec;
            for (const auto& [_, orientation] : base_state.feet_orientation) {
                feet_orientation_vec.push_back(foxglove::CreateQuaternion(
                    builder, orientation.x(), orientation.y(), orientation.z(), orientation.w()));
            }
            auto feet_orientation = builder.CreateVector(feet_orientation_vec);

            // Create feet linear velocity vectors
            std::vector<flatbuffers::Offset<foxglove::Vector3>> feet_linear_velocity_vec;
            for (const auto& [_, velocity] : base_state.feet_linear_velocity) {
                feet_linear_velocity_vec.push_back(
                    foxglove::CreateVector3(builder, velocity.x(), velocity.y(), velocity.z()));
            }
            auto feet_linear_velocity = builder.CreateVector(feet_linear_velocity_vec);

            // Create feet angular velocity vectors
            std::vector<flatbuffers::Offset<foxglove::Vector3>> feet_angular_velocity_vec;
            for (const auto& [_, velocity] : base_state.feet_angular_velocity) {
                feet_angular_velocity_vec.push_back(
                    foxglove::CreateVector3(builder, velocity.x(), velocity.y(), velocity.z()));
            }
            auto feet_angular_velocity = builder.CreateVector(feet_angular_velocity_vec);

            // Create the BaseState message with fields in the correct order according to IDs
            auto base_state_fb =
                foxglove::CreateBaseState(builder,
                                          &time,                             // id: 0
                                          contact_names,                     // id: 1
                                          base_position,                     // id: 2
                                          base_orientation,                  // id: 3
                                          base_linear_velocity,              // id: 4
                                          base_angular_velocity,             // id: 5
                                          base_linear_acceleration,          // id: 6
                                          base_angular_acceleration,         // id: 7
                                          imu_linear_acceleration_bias,      // id: 8
                                          imu_angular_velocity_bias,         // id: 9
                                          base_position_cov,                 // id: 10
                                          base_orientation_cov,              // id: 11
                                          base_linear_velocity_cov,          // id: 12
                                          base_angular_velocity_cov,         // id: 13
                                          imu_linear_acceleration_bias_cov,  // id: 14
                                          imu_angular_velocity_bias_cov,     // id: 15
                                          contacts_position,                 // id: 16
                                          contacts_orientation,              // id: 17
                                          contacts_position_cov,             // id: 18
                                          contacts_orientation_cov,          // id: 19
                                          feet_position,                     // id: 20
                                          feet_orientation,                  // id: 21
                                          feet_linear_velocity,              // id: 22
                                          feet_angular_velocity              // id: 23
                );

            builder.Finish(base_state_fb);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            writeMessage(4, base_sequence_++, timestamp, reinterpret_cast<const std::byte*>(buffer),
                         size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging Base State: " << e.what() << std::endl;
        }
    }

    void log(const std::map<std::string, Eigen::Isometry3d>& frame_tfs, double ts) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = ts;
            }
            const double timestamp = ts - start_time_.value();

            if (timestamp < 0) {
                std::cerr << "Timestamp is negative: " << timestamp << std::endl;
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            // Create timestamp
            auto time = foxglove::Time(sec, nsec);

            // Create transforms vector - preallocate capacity
            std::vector<flatbuffers::Offset<foxglove::FrameTransform>> transforms_vector;
            transforms_vector.reserve(frame_tfs.size());

            // Reuse cached world_frame string
            auto parent_frame = builder.CreateString("world");

            for (const auto& [frame_id, tf] : frame_tfs) {
                // Create frame_id strings
                auto child_frame = builder.CreateString(frame_id);

                // Create translation
                auto translation = foxglove::CreateVector3(
                    builder, tf.translation().x(), tf.translation().y(), tf.translation().z());

                // Create rotation
                const auto& quaternion = Eigen::Quaterniond(tf.linear());
                auto rotation = foxglove::CreateQuaternion(builder, quaternion.x(), quaternion.y(),
                                                           quaternion.z(), quaternion.w());

                // Create transform
                auto transform = foxglove::CreateFrameTransform(builder, &time, parent_frame,
                                                                child_frame, translation, rotation);

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

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            // Get the serialized data
            writeMessage(5, tfs_sequence_++, timestamp, reinterpret_cast<const std::byte*>(buffer),
                         size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging feet transforms: " << e.what() << std::endl;
        }
    }

    void log(const JointState& joint_state) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = joint_state.timestamp;
            }
            const double timestamp = joint_state.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cerr << "Timestamp is negative: " << timestamp << std::endl;
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            auto time = foxglove::Time(sec, nsec);

            // Create joint state vectors
            std::vector<flatbuffers::Offset<flatbuffers::String>> joint_names_vec;
            std::vector<double> joint_positions_vec;
            std::vector<double> joint_velocities_vec;

            // Pre-allocate vectors for better performance
            joint_names_vec.reserve(joint_state.joints_position.size());
            joint_positions_vec.reserve(joint_state.joints_position.size());
            joint_velocities_vec.reserve(joint_state.joints_position.size());

            // Build the vectors
            for (const auto& [name, position] : joint_state.joints_position) {
                joint_names_vec.push_back(builder.CreateString(name));
                joint_positions_vec.push_back(position);
                joint_velocities_vec.push_back(joint_state.joints_velocity.at(name));
            }

            // Create the vectors in the builder
            auto names_offset = builder.CreateVector(joint_names_vec);
            auto positions_offset = builder.CreateVector(joint_positions_vec);
            auto velocities_offset = builder.CreateVector(joint_velocities_vec);

            // Create the root message
            foxglove::JointStateBuilder joint_states_builder(builder);
            joint_states_builder.add_timestamp(&time);
            joint_states_builder.add_names(names_offset);
            joint_states_builder.add_positions(positions_offset);
            joint_states_builder.add_velocities(velocities_offset);
            auto joint_states_array = joint_states_builder.Finish();

            // Finish the buffer
            builder.Finish(joint_states_array);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            // Write the message
            writeMessage(6, joint_states_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging joint state: " << e.what() << std::endl;
        }
    }

    bool isInitialized() const {
        return start_time_.has_value();
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
        schemas.reserve(5);
        schemas.push_back(createSchema("ImuMeasurement"));
        schemas.push_back(createSchema("ContactState"));
        schemas.push_back(createSchema("CentroidalState"));
        schemas.push_back(createSchema("BaseState"));
        schemas.push_back(createSchema("FrameTransforms"));
        schemas.push_back(createSchema("JointState"));

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.reserve(5);
        channels.push_back(createChannel(1, "/imu"));
        channels.push_back(createChannel(2, "/contact_state"));
        channels.push_back(createChannel(3, "/centroidal_state"));
        channels.push_back(createChannel(4, "/base_state"));
        channels.push_back(createChannel(5, "/tf"));
        channels.push_back(createChannel(6, "/joint_state"));

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
    uint64_t base_sequence_ = 0;
    uint64_t centroidal_sequence_ = 0;
    uint64_t contact_sequence_ = 0;
    uint64_t imu_sequence_ = 0;
    uint64_t tfs_sequence_ = 0;
    uint64_t joint_states_sequence_ = 0;
    std::optional<double> start_time_;

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
    std::mutex writer_mutex_;
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

void ProprioceptionLogger::log(const std::map<std::string, Eigen::Isometry3d>& frame_tfs,
                               double timestamp) {
    pimpl_->log(frame_tfs, timestamp);
}

void ProprioceptionLogger::log(const JointState& joint_state) {
    pimpl_->log(joint_state);
}

void ProprioceptionLogger::setStartTime(double timestamp) {
    pimpl_->setStartTime(timestamp);
}

bool ProprioceptionLogger::isInitialized() const {
    return pimpl_->isInitialized();
}

}  // namespace serow
