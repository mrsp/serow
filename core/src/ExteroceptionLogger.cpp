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
#include <nlohmann/json.hpp>

#include "ExteroceptionLogger.hpp"

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

    // Utility function to convert timestamp to nanoseconds
    uint64_t convertToNanoseconds(double timestamp) {
        // Ensure timestamp is converted to nanoseconds
        return static_cast<uint64_t>(timestamp * 1e9);
    }

    void log(const LocalMapState& local_map_state) {
        try {
            const double timestamp = local_map_state.timestamp;
            const auto& local_map = local_map_state.data;

            // Serialize PointCloud to JSON according to schema
            nlohmann::json json_data;

            // Add timestamp according to schema
            json_data["timestamp"] = {
                {"sec", static_cast<int64_t>(timestamp)},
                {"nsec",
                 static_cast<int32_t>((timestamp - static_cast<int64_t>(timestamp)) * 1e9)}};

            // Add frame_id
            json_data["frame_id"] = "world";

            // Add pose (identity transform)
            json_data["pose"] = {{"position", {{"x", 0.0}, {"y", 0.0}, {"z", 0.0}}},
                                 {"orientation", {{"x", 0.0}, {"y", 0.0}, {"z", 0.0}, {"w", 1.0}}}};

            // Add point stride (3 floats = 12 bytes)
            json_data["point_stride"] = 12;

            // Add fields definition
            json_data["fields"] = {{
                                       {"name", "x"}, {"offset", 0}, {"type", 7}  // FLOAT32
                                   },
                                   {
                                       {"name", "y"}, {"offset", 4}, {"type", 7}  // FLOAT32
                                   },
                                   {
                                       {"name", "z"}, {"offset", 8}, {"type", 7}  // FLOAT32
                                   }};

            // Pack point data into binary buffer
            std::vector<float> point_data;
            point_data.reserve(local_map.size() * 3);
            for (const auto& point : local_map) {
                point_data.push_back(point[0]);
                point_data.push_back(point[1]);
                point_data.push_back(point[2]);
            }

            // Convert binary data to base64
            const char* base64_chars =
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::string base64_data;
            const uint8_t* binary_data = reinterpret_cast<const uint8_t*>(point_data.data());
            size_t binary_size = point_data.size() * sizeof(float);

            base64_data.reserve((binary_size + 2) / 3 * 4);

            for (size_t i = 0; i < binary_size; i += 3) {
                uint32_t n = binary_data[i] << 16;
                if (i + 1 < binary_size)
                    n |= binary_data[i + 1] << 8;
                if (i + 2 < binary_size)
                    n |= binary_data[i + 2];

                base64_data += base64_chars[(n >> 18) & 63];
                base64_data += base64_chars[(n >> 12) & 63];
                base64_data += (i + 1 < binary_size) ? base64_chars[(n >> 6) & 63] : '=';
                base64_data += (i + 2 < binary_size) ? base64_chars[n & 63] : '=';
            }

            json_data["data"] = base64_data;

            std::string json_str = json_data.dump(-1, ' ', true);
            writeMessage(1, local_map_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
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
        schemas.push_back(createPointCloudSchema());

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
        channel.messageEncoding = "json";
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
