#pragma once

#include <filesystem>
#include <string>

#include "mcap/mcap.hpp"

namespace serow {

// Create a schema for PointCloud
static inline mcap::Schema createPointCloudSchema() {
       // Load the .bfbs file
    std::string bfbs_path = "PointCloud.bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / "PointCloud.bfbs";
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open PointCloud.bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error("PointCloud.bfbs is empty");
    }

    // Define SceneEntity schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove.PointCloud";
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

// Create a schema for tfs
static inline mcap::Schema createTFSchema() {
       // Load the .bfbs file
    std::string bfbs_path = "FrameTransform.bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / "FrameTransform.bfbs";
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open FrameTransform.bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error("FrameTransform.bfbs is empty");
    }

    // Define SceneEntity schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove.FrameTransform";
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}



// Create a schema for an array of tfs
static inline mcap::Schema createFrameTransformsSchema() {
       // Load the .bfbs file
    std::string bfbs_path = "FrameTransforms.bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / "FrameTransforms.bfbs";
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open FrameTransforms.bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error("FrameTransforms.bfbs is empty");
    }

    // Define SceneEntity schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove.FrameTransforms";
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

static inline mcap::Schema createBaseStateSchema() {
    // Load the .bfbs file
    std::string bfbs_path = "BaseState.bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / "BaseState.bfbs";
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open BaseState.bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error("BaseState.bfbs is empty");
    }

    // Define SceneEntity schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove.BaseState";
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

static inline mcap::Schema createCentroidalStateSchema() {
    // Load the .bfbs file
    std::string bfbs_path = "CentroidalState.bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / "CentroidalState.bfbs";
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open CentroidalState.bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error("CentroidalState.bfbs is empty");
    }

    // Define SceneEntity schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove.CentroidalState";
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}


static inline mcap::Schema createContactStateSchema() {
    mcap::Schema schema;
    schema.name = "ContactState";
    schema.encoding = "jsonschema";
    std::string schema_data =
        "{"
        "    \"type\": \"object\","
        "    \"properties\": {"
        "        \"timestamp\": {"
        "            \"type\": \"number\","
        "            \"description\": \"Timestamp of the state (s)\""
        "        },"
        "        \"num_contacts\": {"
        "            \"type\": \"integer\","
        "            \"description\": \"Number of contact points\""
        "        },"
        "        \"contact_names\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"string\""
        "            },"
        "            \"description\": \"Array of contact point names\""
        "        },"
        "        \"contacts\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"object\","
        "                \"properties\": {"
        "                    \"status\": {"
        "                        \"type\": \"boolean\","
        "                        \"description\": \"Contact status (true if in contact)\""
        "                    },"
        "                    \"probability\": {"
        "                        \"type\": \"number\","
        "                        \"description\": \"Contact probability (0-1)\""
        "                    },"
        "                    \"force\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 3,"
        "                        \"maxItems\": 3,"
        "                        \"description\": \"3D Contact force in world frame "
        "coordinates (N)\""
        "                    },"
        "                    \"torque\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 3,"
        "                        \"maxItems\": 3,"
        "                        \"description\": \"3D Contact torque in world frame "
        "coordinates (Nm)\""
        "                    }"
        "                },"
        "                \"required\": [\"status\", \"probability\", \"force\"]"
        "            }"
        "        }"
        "    },"
        "    \"required\": [\"timestamp\", \"num_contacts\", \"contact_names\", \"contacts\"]"
        "}";

    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

static inline mcap::Schema createImuMeasurementSchema() {
    mcap::Schema schema;
    schema.name = "ImuMeasurement";
    schema.encoding = "jsonschema";
    std::string schema_data =
        "{"
        "    \"type\": \"object\","
        "    \"properties\": {"
        "        \"timestamp\": {"
        "            \"type\": \"number\","
        "            \"description\": \"Timestamp of the measurement (s)\""
        "        },"
        "        \"linear_acceleration\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"Linear acceleration in x, y, z (m/s^2)\""
        "        },"
        "        \"angular_velocity\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"Angular velocity in x, y, z (rad/s)\""
        "        }"
        "    },"
        "    \"required\": [\"timestamp\", \"linear_acceleration\", \"angular_velocity\"]"
        "}";

    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

static inline mcap::Schema createSceneEntitySchema() {
    // Load the .bfbs file
    std::string bfbs_path = "SceneEntity.bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / "SceneEntity.bfbs";
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open SceneEntity.bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error("SceneEntity.bfbs is empty");
    }

    // Define SceneEntity schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove.SceneEntity";
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

}  // namespace serow
