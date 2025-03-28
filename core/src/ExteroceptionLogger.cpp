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
#include "ExteroceptionLogger.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>

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

            nlohmann::json json_data;
            json_data["timestamp"] = timestamp;
            json_data["map_size"] = map_size;

            // Create array of points
            nlohmann::json points = nlohmann::json::array();

            for (size_t i = 0; i < map_size; ++i) {
                const std::array<float, 3>& point = local_map[i];
                points.push_back({static_cast<double>(point[0]), static_cast<double>(point[1]),
                                  static_cast<double>(point[2])});
            }

            json_data["points"] = points;

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
        schemas.push_back(createLocalMapSchema());

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.push_back(createChannel(1, "/local-map"));

        for (auto& channel : channels) {
            writer_->addChannel(channel);
        }
    }

    mcap::Schema createLocalMapSchema() {
        mcap::Schema schema;
        schema.name = "LocalMap";
        schema.encoding = "jsonschema";
        std::string schema_data =
            "{"
            "    \"$schema\": \"http://json-schema.org/draft-07/schema#\","
            "    \"type\": \"object\","
            "    \"properties\": {"
            "        \"timestamp\": {"
            "            \"type\": \"number\","
            "            \"description\": \"Timestamp of the local map data\""
            "        },"
            "        \"map_size\": {"
            "            \"type\": \"integer\","
            "            \"description\": \"Size of the local map grid\""
            "        },"
            "        \"points\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"array\","
            "                \"items\": {"
            "                    \"type\": \"number\""
            "                },"
            "                \"minItems\": 3,"
            "                \"maxItems\": 3,"
            "                \"description\": \"[x, y, z] coordinates of a point in the map\""
            "            },"
            "            \"description\": \"Array of 3D points in the local map\""
            "        }"
            "    },"
            "    \"required\": [\"timestamp\", \"map_size\", \"points\"]"
            "}";

        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
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
    uint64_t local_map_sequence_ = 0;
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
