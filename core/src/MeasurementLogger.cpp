/**
 * Copyright (C) Stylianos Piperakis, Ownage Dynamics L.P.
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

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "ForceTorqueMeasurements_generated.h"
#include "FrameTransform_generated.h"
#include "ImuMeasurement_generated.h"
#include "JointMeasurements_generated.h"

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
            std::cerr << "ProprioceptionLogger initialization error: " << e.what() << '\n';
            throw;
        }
    }

    ~Impl() {
        try {
            if (writer_) {
                writer_->close();
            }
        } catch (const std::exception& e) {
            std::cerr << "Error closing MCAP writer: " << e.what() << '\n';
        }
    }

    // Use same canonical ns as writeMessage (round + round-trip) so flatbuffer time and index match
    inline void splitTimestamp(double timestamp, int64_t& sec, int32_t& nsec) const noexcept {
        uint64_t ns = static_cast<uint64_t>(std::round(timestamp * 1e9));
        ns = static_cast<uint64_t>(static_cast<double>(ns) / 1e9 * 1e9);
        sec = static_cast<int64_t>(ns / 1000000000);
        nsec = static_cast<int32_t>(ns % 1000000000);
    }

    void log(const ImuMeasurement& imu_measurement) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = imu_measurement.timestamp;
            }
            const double timestamp = imu_measurement.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cout << "[SEROW/MeasurementLogger]: IMU Timestamp is negative " << timestamp
                          << " returning without logging" << '\n';
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

            // Create linear acceleration covariance matrix
            auto linear_acceleration_cov =
                foxglove::CreateMatrix3(builder, imu_measurement.linear_acceleration_cov(0, 0),
                                        imu_measurement.linear_acceleration_cov(0, 1),
                                        imu_measurement.linear_acceleration_cov(0, 2),
                                        imu_measurement.linear_acceleration_cov(1, 0),
                                        imu_measurement.linear_acceleration_cov(1, 1),
                                        imu_measurement.linear_acceleration_cov(1, 2),
                                        imu_measurement.linear_acceleration_cov(2, 0),
                                        imu_measurement.linear_acceleration_cov(2, 1),
                                        imu_measurement.linear_acceleration_cov(2, 2));

            // Create angular velocity covariance matrix
            auto angular_velocity_cov =
                foxglove::CreateMatrix3(builder, imu_measurement.angular_velocity_cov(0, 0),
                                        imu_measurement.angular_velocity_cov(0, 1),
                                        imu_measurement.angular_velocity_cov(0, 2),
                                        imu_measurement.angular_velocity_cov(1, 0),
                                        imu_measurement.angular_velocity_cov(1, 1),
                                        imu_measurement.angular_velocity_cov(1, 2),
                                        imu_measurement.angular_velocity_cov(2, 0),
                                        imu_measurement.angular_velocity_cov(2, 1),
                                        imu_measurement.angular_velocity_cov(2, 2));

            // Create angular velocity bias covariance matrix
            auto angular_velocity_bias_cov =
                foxglove::CreateMatrix3(builder, imu_measurement.angular_velocity_bias_cov(0, 0),
                                        imu_measurement.angular_velocity_bias_cov(0, 1),
                                        imu_measurement.angular_velocity_bias_cov(0, 2),
                                        imu_measurement.angular_velocity_bias_cov(1, 0),
                                        imu_measurement.angular_velocity_bias_cov(1, 1),
                                        imu_measurement.angular_velocity_bias_cov(1, 2),
                                        imu_measurement.angular_velocity_bias_cov(2, 0),
                                        imu_measurement.angular_velocity_bias_cov(2, 1),
                                        imu_measurement.angular_velocity_bias_cov(2, 2));

            // Create linear acceleration bias covariance matrix
            auto linear_acceleration_bias_cov =
                foxglove::CreateMatrix3(builder, imu_measurement.linear_acceleration_bias_cov(0, 0),
                                        imu_measurement.linear_acceleration_bias_cov(0, 1),
                                        imu_measurement.linear_acceleration_bias_cov(0, 2),
                                        imu_measurement.linear_acceleration_bias_cov(1, 0),
                                        imu_measurement.linear_acceleration_bias_cov(1, 1),
                                        imu_measurement.linear_acceleration_bias_cov(1, 2),
                                        imu_measurement.linear_acceleration_bias_cov(2, 0),
                                        imu_measurement.linear_acceleration_bias_cov(2, 1),
                                        imu_measurement.linear_acceleration_bias_cov(2, 2));

            // Create angular acceleration vector
            auto angular_acceleration = foxglove::CreateVector3(
                builder, imu_measurement.angular_acceleration.x(),
                imu_measurement.angular_acceleration.y(), imu_measurement.angular_acceleration.z());

            // Create the root ImuState
            auto imu = foxglove::CreateImuMeasurement(
                builder,
                &time,                         // timestamp
                linear_acceleration,           // linear_acceleration
                angular_velocity,              // angular_velocity
                orientation,                   // orientation
                linear_acceleration_cov,       // linear_acceleration_cov
                angular_velocity_cov,          // angular_velocity_cov
                angular_velocity_bias_cov,     // angular_velocity_bias_cov
                linear_acceleration_bias_cov,  // linear_acceleration_bias_cov
                angular_acceleration);         // angular_acceleration

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
            std::cerr << "Error logging IMU measurement: " << e.what() << '\n';
        }
    }

    void log(const BasePoseGroundTruth& base_pose_ground_truth) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = base_pose_ground_truth.timestamp;
            }
            const double timestamp = base_pose_ground_truth.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cout
                    << "[SEROW/MeasurementLogger]: Base Pose Ground Truth Timestamp is negative "
                    << timestamp << " returning without logging" << '\n';
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            // Create timestamp
            auto time = foxglove::Time(sec, nsec);

            // Reuse cached world_frame string
            auto parent_frame = builder.CreateString("world");

            // Create frame_id strings
            auto child_frame = builder.CreateString("base");

            // Create translation
            auto translation = foxglove::CreateVector3(builder, base_pose_ground_truth.position.x(),
                                                       base_pose_ground_truth.position.y(),
                                                       base_pose_ground_truth.position.z());

            // Create rotation
            const auto& quaternion = base_pose_ground_truth.orientation;
            auto rotation = foxglove::CreateQuaternion(builder, quaternion.x(), quaternion.y(),
                                                       quaternion.z(), quaternion.w());

            // Create transform
            auto transform = foxglove::CreateFrameTransform(builder, &time, parent_frame,
                                                            child_frame, translation, rotation);

            // Finish the buffer
            builder.Finish(transform);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            // Get the serialized data
            writeMessage(4, base_pose_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging base pose ground truth: " << e.what() << '\n';
        }
    }

    void log(const std::map<std::string, JointMeasurement>& joints) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = joints.begin()->second.timestamp;
            }
            const double timestamp = joints.begin()->second.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cout << "[SEROW/MeasurementLogger]: Joint Measurements Timestamp is negative "
                          << timestamp << " returning without logging" << '\n';
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            // Create timestamp
            auto time = foxglove::Time(sec, nsec);

            // Create joint names vector
            std::vector<flatbuffers::Offset<flatbuffers::String>> joint_names_vec;
            for (const auto& [joint_name, _] : joints) {
                joint_names_vec.push_back(builder.CreateString(joint_name));
            }
            auto joint_names = builder.CreateVector(joint_names_vec);

            // Create joint positions vector
            std::vector<double> joint_positions_vec;
            for (const auto& [_, measurement] : joints) {
                joint_positions_vec.push_back(measurement.position);
            }
            auto joint_positions = builder.CreateVector(joint_positions_vec);

            // Create joint velocities vector
            std::vector<double> joint_velocities_vec;
            for (const auto& [_, measurement] : joints) {
                if (measurement.velocity.has_value()) {
                    joint_velocities_vec.push_back(measurement.velocity.value());
                } else {
                    joint_velocities_vec.push_back(0.0);
                }
            }
            auto joint_velocities = builder.CreateVector(joint_velocities_vec);

            // Create the joint measurement message
            auto measurement = foxglove::CreateJointMeasurements(builder, &time, joint_names,
                                                                 joint_positions, joint_velocities);

            builder.Finish(measurement);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            // Get the serialized data
            writeMessage(2, joints_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging joint measurement: " << e.what() << '\n';
        }
    }

    void log(const std::map<std::string, ForceTorqueMeasurement>& ft) {
        try {
            flatbuffers::FlatBufferBuilder builder(INITIAL_BUILDER_SIZE);

            if (!start_time_.has_value()) {
                start_time_ = ft.begin()->second.timestamp;
            }
            const double timestamp = ft.begin()->second.timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cout << "[SEROW/MeasurementLogger]: Force Torque Timestamp is negative "
                          << timestamp << " returning without logging" << '\n';
                return;
            }

            // Convert timestamp to sec and nsec
            int64_t sec;
            int32_t nsec;
            splitTimestamp(timestamp, sec, nsec);

            // Create timestamp
            auto time = foxglove::Time(sec, nsec);

            // Create frame names vector
            std::vector<flatbuffers::Offset<flatbuffers::String>> frame_names_vec;
            for (const auto& [frame_name, _] : ft) {
                frame_names_vec.push_back(builder.CreateString(frame_name));
            }
            auto frame_names = builder.CreateVector(frame_names_vec);

            // Create forces vector
            std::vector<flatbuffers::Offset<foxglove::Vector3>> forces_vec;
            for (const auto& [_, measurement] : ft) {
                forces_vec.push_back(foxglove::CreateVector3(
                    builder, measurement.force.x(), measurement.force.y(), measurement.force.z()));
            }
            auto forces = builder.CreateVector(forces_vec);

            // Create torques vector
            std::vector<flatbuffers::Offset<foxglove::Vector3>> torques_vec;
            for (const auto& [_, measurement] : ft) {
                if (measurement.torque.has_value()) {
                    torques_vec.push_back(foxglove::CreateVector3(
                        builder, measurement.torque.value().x(), measurement.torque.value().y(),
                        measurement.torque.value().z()));
                } else {
                    torques_vec.push_back(foxglove::CreateVector3(builder, 0.0, 0.0, 0.0));
                }
            }
            auto torques = builder.CreateVector(torques_vec);

            // Create the force torque measurement message
            auto measurement = foxglove::CreateForceTorqueMeasurements(builder, &time, frame_names,
                                                                       forces, torques);

            builder.Finish(measurement);

            // Get the buffer pointer and size before any potential modifications
            const uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();

            // Get the serialized data
            writeMessage(3, ft_sequence_++, timestamp, reinterpret_cast<const std::byte*>(buffer),
                         size);
        } catch (const std::exception& e) {
            std::cerr << "Error logging force torque measurement: " << e.what() << '\n';
        }
    }

    void setStartTime(double timestamp) {
        start_time_ = timestamp;
    }

    bool isInitialized() const {
        return start_time_.has_value();
    }

private:
    // Optimized message writing with reuse of message object
    void writeMessage(uint16_t channel_id, uint64_t sequence, double timestamp,
                      const std::byte* data, size_t data_size) noexcept {
        if (data_size == 0 || data == nullptr) {
            return;
        }
        try {
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(timestamp));
            uint64_t ns_timestamp = ns.count();

            mcap::Message message;
            message.channelId = channel_id;
            message.sequence = sequence;
            message.dataSize = data_size;
            message.data = data;

            std::lock_guard<std::mutex> lock(writer_mutex_);
            // Keep MCAP record log_time non-decreasing; flatbuffer payload time unchanged.
            ns_timestamp = std::max(ns_timestamp, last_log_time_ns_);
            last_log_time_ns_ = ns_timestamp;
            message.logTime = ns_timestamp;
            message.publishTime = ns_timestamp;

            auto status = writer_->write(message);
            if (status.code != mcap::StatusCode::Success) {
                std::cerr << "Failed to write message for channel " << channel_id << ": "
                          << status.message << '\n';
            }
        } catch (const std::exception& e) {
            std::cerr << "Error in writeMessage: " << e.what() << '\n';
        }
    }

    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        schemas.reserve(4);
        schemas.push_back(createSchema("ImuMeasurement"));
        schemas.push_back(createSchema("JointMeasurements"));
        schemas.push_back(createSchema("ForceTorqueMeasurements"));
        schemas.push_back(createSchema("FrameTransform"));

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.reserve(4);
        channels.push_back(createChannel(1, "/imu"));
        channels.push_back(createChannel(2, "/joints"));
        channels.push_back(createChannel(3, "/ft"));
        channels.push_back(createChannel(4, "/base_pose_ground_truth"));

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
    uint64_t imu_sequence_ = 0;
    uint64_t joints_sequence_ = 0;
    uint64_t ft_sequence_ = 0;
    uint64_t base_pose_sequence_ = 0;
    std::optional<double> start_time_;
    uint64_t last_log_time_ns_{0};

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

// void MeasurementLogger::log(const KinematicMeasurement& kinematic_measurement) {
//     pimpl_->log(kinematic_measurement);
// }

void MeasurementLogger::log(const BasePoseGroundTruth& base_pose_ground_truth) {
    pimpl_->log(base_pose_ground_truth);
}

void MeasurementLogger::log(const std::map<std::string, JointMeasurement>& joints) {
    pimpl_->log(joints);
}

void MeasurementLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft) {
    pimpl_->log(ft);
}

void MeasurementLogger::setStartTime(double timestamp) {
    pimpl_->setStartTime(timestamp);
}

bool MeasurementLogger::isInitialized() const {
    return pimpl_->isInitialized();
}

}  // namespace serow
