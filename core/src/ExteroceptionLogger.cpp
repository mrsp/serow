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
#include <chrono>
#include <iomanip>
#include <iostream>

#include "ExteroceptionLogger.hpp"
#include "PointCloud_generated.h"

namespace serow {

// Implementation class definition
class ExteroceptionLogger::Impl {
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
            // Configure MCAP options
            mcap::McapWriterOptions options("");
            writer_->open(*file_writer_, options);

            // Initialize schemas and channels
            initializeSchemas();
            initializeChannels();
        } catch (const std::exception& e) {
            std::cerr << "ExteroceptionLogger initialization error: " << e.what() << std::endl;
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

    // Utility function to convert timestamp to nanoseconds
    uint64_t convertToNanoseconds(double timestamp) {
        // Ensure timestamp is converted to nanoseconds
        return static_cast<uint64_t>(timestamp * 1e9);
    }

    void log(const LocalMapState& local_map_state) {
        try {
            const double timestamp = local_map_state.timestamp;
            const auto& local_map = local_map_state.data;

            // Create a FlatBuffer builder
            flatbuffers::FlatBufferBuilder builder;

            std::vector<float> point_data;
            point_data.reserve(local_map.size() * 3);

            for (size_t i = 0; i < local_map.size(); i++) {
                const auto& point = local_map[i];
                point_data.push_back(point[0]);
                point_data.push_back(point[1]);
                point_data.push_back(point[2]);
            }

            // Create the timestamp
            auto ts = foxglove::Time(
                static_cast<int64_t>(timestamp),
                static_cast<int32_t>((timestamp - static_cast<int64_t>(timestamp)) * 1e9));

            // Create the pose (identity transform)
            auto position = foxglove::CreateVector3(builder, 0.0, 0.0, 0.0);
            auto orientation = foxglove::CreateQuaternion(builder, 0.0, 0.0, 0.0, 1.0);
            auto pose = foxglove::CreatePose(builder, position, orientation);

            // Create the frame_id string
            auto frame_id = builder.CreateString("world");

            // Create fields for point data
            std::vector<flatbuffers::Offset<foxglove::PackedElementField>> fields;
            fields.push_back(foxglove::CreatePackedElementField(builder, builder.CreateString("x"),
                                                                0, foxglove::NumericType_FLOAT32));
            fields.push_back(foxglove::CreatePackedElementField(builder, builder.CreateString("y"),
                                                                4, foxglove::NumericType_FLOAT32));
            fields.push_back(foxglove::CreatePackedElementField(builder, builder.CreateString("z"),
                                                                8, foxglove::NumericType_FLOAT32));
            auto fields_vec = builder.CreateVector(fields);

            // Convert float data to uint8_t for the data field
            std::vector<uint8_t> binary_data;
            binary_data.resize(point_data.size() * sizeof(float));
            std::memcpy(binary_data.data(), point_data.data(), binary_data.size());
            auto data_vec = builder.CreateVector(binary_data);

            // Create the PointCloud
            foxglove::PointCloudBuilder pc_builder(builder);
            pc_builder.add_timestamp(&ts);
            pc_builder.add_frame_id(frame_id);
            pc_builder.add_pose(pose);
            pc_builder.add_point_stride(12);  // 3 floats * 4 bytes
            pc_builder.add_fields(fields_vec);
            pc_builder.add_data(data_vec);
            auto pointcloud = pc_builder.Finish();

            // Finish the buffer
            builder.Finish(pointcloud);

            // Get the serialized data
            uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();
            writeMessage(1, local_map_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
            last_local_map_timestamp_ = timestamp;
        } catch (const std::exception& e) {
            std::cerr << "Error logging local map: " << e.what() << std::endl;
        }
    }

    double getLastLocalMapTimestamp() const {
        return last_local_map_timestamp_;
    }

private:
    void writeMessage(uint16_t channel_id, uint64_t sequence, double timestamp,
                      const std::byte* data, size_t data_size) {
        try {
            mcap::Message message;
            message.channelId = channel_id;
            message.sequence = sequence;

            // Precise nanosecond timestamp
            uint64_t ns_timestamp = convertToNanoseconds(timestamp);
            message.logTime = ns_timestamp;
            message.publishTime = ns_timestamp;

            message.dataSize = data_size;
            message.data = data;

            auto status = writer_->write(message);
            if (status.code != mcap::StatusCode::Success) {
                throw std::runtime_error("Failed to write message for channel " +
                                         std::to_string(channel_id) + ": " + status.message);
            }
        } catch (const std::exception& e) {
            std::cerr << "MCAP write error: " << e.what() << std::endl;
        }
    }

    // Reuse the original schema and channel initialization methods
    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        schemas.push_back(createSchema("PointCloud"));

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.push_back(createChannel(1, "/elevation-map"));

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

    // Sequence counters
    uint64_t local_map_sequence_{};
    double last_local_map_timestamp_{-1.0};

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
};

// Public interface implementation
ExteroceptionLogger::ExteroceptionLogger(const std::string& filename)
    : pimpl_(std::make_unique<Impl>(filename)) {}

ExteroceptionLogger::~ExteroceptionLogger() = default;

void ExteroceptionLogger::log(const LocalMapState& local_map_state) {
    pimpl_->log(local_map_state);
}

double ExteroceptionLogger::getLastLocalMapTimestamp() const {
    return pimpl_->getLastLocalMapTimestamp();
}

}  // namespace serow
