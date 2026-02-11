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
#include "ExteroceptionLogger.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "Schemas.hpp"

#include "Grid_generated.h"
#include "PackedElementField_generated.h"
#include "Pose_generated.h"
#include "Quaternion_generated.h"
#include "Time_generated.h"
#include "Vector2_generated.h"
#include "Vector3_generated.h"

namespace serow {

// Implementation class definition
class ExteroceptionLogger::Impl {
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

    void setStartTime(double timestamp) {
        start_time_ = timestamp;
    }

    void log(const std::vector<float>& elevation, const std::vector<float>& variance,
             double timestamp) {
        try {
            if (!start_time_.has_value()) {
                start_time_ = timestamp;
            }
            timestamp = timestamp - start_time_.value();

            if (timestamp < 0) {
                std::cout << "[SEROW/ExteroceptionLogger]: Local Map Timestamp is negative "
                          << timestamp << " returning without logging" << std::endl;
                return;
            }

            if (timestamp <= last_timestamp_) {
                return;
            }

            // Ensure both vectors have the same size
            if (elevation.size() != variance.size()) {
                std::cerr << "Elevation and variance data size mismatch" << std::endl;
                return;
            }

            // Create a FlatBuffer builder
            flatbuffers::FlatBufferBuilder builder;

            // Create the timestamp
            auto time = foxglove::Time(
                static_cast<int64_t>(timestamp),
                static_cast<int32_t>((timestamp - static_cast<int64_t>(timestamp)) * 1e9));

            // Create the pose (grid origin)
            const double origin_x = grid_origin_x_ - (grid_width_ * grid_resolution_ / 2.0);
            const double origin_y = grid_origin_y_ - (grid_height_ * grid_resolution_ / 2.0);
            auto position = foxglove::CreateVector3(builder, origin_x, origin_y, 0.0);
            auto orientation = foxglove::CreateQuaternion(builder, 0.0, 0.0, 0.0, 1.0);
            auto pose = foxglove::CreatePose(builder, position, orientation);
            auto cell_size = foxglove::CreateVector2(builder, grid_resolution_, grid_resolution_);

            // Create the frame_id string
            auto frame_id = builder.CreateString("world");

            // Create fields for the data (elevation and variance)
            std::vector<flatbuffers::Offset<foxglove::PackedElementField>> field_offsets;
            // Elevation field at offset 0
            auto elevation_field_name = builder.CreateString("elevation");
            auto elevation_field = foxglove::CreatePackedElementField(
                builder, elevation_field_name, 0, foxglove::NumericType_FLOAT32);
            field_offsets.push_back(elevation_field);

            // Variance field at offset 4 (after the 4-byte elevation float)
            auto variance_field_name = builder.CreateString("variance");
            auto variance_field = foxglove::CreatePackedElementField(
                builder, variance_field_name, 4, foxglove::NumericType_FLOAT32);
            field_offsets.push_back(variance_field);

            auto fields = builder.CreateVector(field_offsets);

            // Interleave elevation and variance data
            std::vector<uint8_t> interleaved_data;
            interleaved_data.reserve(elevation.size() * 2 * sizeof(float));

            for (size_t i = 0; i < elevation.size(); ++i) {
                // Add elevation bytes
                const uint8_t* elev_bytes = reinterpret_cast<const uint8_t*>(&elevation[i]);
                interleaved_data.insert(interleaved_data.end(), elev_bytes,
                                        elev_bytes + sizeof(float));

                // Add variance bytes
                const uint8_t* var_bytes = reinterpret_cast<const uint8_t*>(&variance[i]);
                interleaved_data.insert(interleaved_data.end(), var_bytes,
                                        var_bytes + sizeof(float));
            }

            auto data_vec = builder.CreateVector(interleaved_data);

            // Calculate strides - now each cell has 8 bytes (elevation + variance)
            uint32_t cell_stride = 8;  // 8 bytes per cell (2 float32s)
            uint32_t column_count = grid_width_;
            uint32_t row_stride = column_count * cell_stride;

            // Create the Grid
            auto grid = foxglove::CreateGrid(builder, &time, frame_id, pose, column_count,
                                             cell_size, row_stride, cell_stride, fields, data_vec);

            // Finish the buffer
            builder.Finish(grid);

            // Get the serialized data
            uint8_t* buffer = builder.GetBufferPointer();
            size_t size = builder.GetSize();
            writeMessage(1, local_map_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(buffer), size);
            last_timestamp_ = timestamp;
        } catch (const std::exception& e) {
            std::cerr << "Error logging local map: " << e.what() << std::endl;
        }
    }

    double getLastTimestamp() const {
        return last_timestamp_;
    }

    bool isInitialized() const {
        return start_time_.has_value();
    }

    void setGridParameters(double resolution, uint32_t width, uint32_t height, double origin_x,
                           double origin_y) {
        grid_resolution_ = resolution;
        grid_width_ = width;
        grid_height_ = height;
        grid_origin_x_ = origin_x;
        grid_origin_y_ = origin_y;
    }

private:
    // Grid parameters
    double grid_resolution_{0.01};  // 1cm resolution by default
    uint32_t grid_width_{1000};
    uint32_t grid_height_{1000};
    double grid_origin_x_{0.0};
    double grid_origin_y_{0.0};

    void writeMessage(uint16_t channel_id, uint64_t sequence, double timestamp,
                      const std::byte* data, size_t data_size) {
        if (data_size == 0 || data == nullptr) {
            return; 
        }
        try {
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

            auto status = writer_->write(message);
            if (status.code != mcap::StatusCode::Success) {
                throw std::runtime_error("Failed to write message for channel " +
                                         std::to_string(channel_id) + ": " + status.message);
            }
        } catch (const std::exception& e) {
            std::cerr << "MCAP write error: " << e.what() << std::endl;
        }
    }

    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        schemas.push_back(createSchema("Grid"));

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
    double last_timestamp_{-1.0};
    std::optional<double> start_time_;
    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
};

// Public interface implementation
ExteroceptionLogger::ExteroceptionLogger(const std::string& filename)
    : pimpl_(std::make_unique<Impl>(filename)) {}

ExteroceptionLogger::~ExteroceptionLogger() = default;

void ExteroceptionLogger::log(const std::vector<float>& elevation,
                              const std::vector<float>& variance, double timestamp) {
    pimpl_->log(elevation, variance, timestamp);
}

double ExteroceptionLogger::getLastTimestamp() const {
    return pimpl_->getLastTimestamp();
}

void ExteroceptionLogger::setStartTime(double timestamp) {
    pimpl_->setStartTime(timestamp);
}

bool ExteroceptionLogger::isInitialized() const {
    return pimpl_->isInitialized();
}

void ExteroceptionLogger::setGridParameters(double resolution, uint32_t width, uint32_t height,
                                            double origin_x, double origin_y) {
    pimpl_->setGridParameters(resolution, width, height, origin_x, origin_y);
}

}  // namespace serow
