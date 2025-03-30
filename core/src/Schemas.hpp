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
    mcap::Schema schema;
    schema.name = "BaseState";
    schema.encoding = "jsonschema";
    std::string schema_data =
        "{"
        "    \"type\": \"object\","
        "    \"properties\": {"
        "        \"timestamp\": {"
        "            \"type\": \"number\","
        "            \"description\": \"Timestamp of the state (s)\""
        "        },"
        "        \"base_position\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Base position in world frame coordinates (m)\""
        "        },"
        "        \"base_orientation\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 4,"
        "            \"maxItems\": 4,"
        "            \"description\": \"Base orientation quaternion [w, x, y, z] in world "
        "frame coordinates\""
        "        },"
        "        \"base_linear_velocity\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Base linear velocity in world frame coordinates "
        "(m/s)\""
        "        },"
        "        \"base_angular_velocity\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Base angular velocity in world frame coordinates "
        "(rad/s)\""
        "        },"
        "        \"base_linear_acceleration\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Base linear acceleration in world frame coordinates "
        "(m/s^2)\""
        "        },"
        "        \"base_angular_acceleration\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Base angular acceleration in world frame "
        "coordinates (rad/s^2)\""
        "        },"
        "        \"imu_linear_acceleration_bias\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D IMU linear acceleration bias in IMU frame "
        "coordinates (m/s^2)\""
        "        },"
        "        \"imu_angular_velocity_bias\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D IMU angular velocity bias in IMU frame coordinates "
        "(rad/s)\""
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
        "                    \"name\": {"
        "                        \"type\": \"string\","
        "                        \"description\": \"Contact point name\""
        "                    },"
        "                    \"position\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 3,"
        "                        \"maxItems\": 3,"
        "                        \"description\": \"3D contact position in world frame "
        "coordinates (m)\""
        "                    },"
        "                    \"foot_position\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 3,"
        "                        \"maxItems\": 3,"
        "                        \"description\": \"3D foot position in world frame "
        "coordinates (m)\""
        "                    },"
        "                    \"foot_orientation\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 4,"
        "                        \"maxItems\": 4,"
        "                        \"description\": \"Foot orientation quaternion [w, x, y, z] "
        "in world frame coordinates\""
        "                    },"
        "                    \"foot_linear_velocity\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 3,"
        "                        \"maxItems\": 3,"
        "                        \"description\": \"3D foot linear velocity in world frame "
        "coordinates (m/s)\""
        "                    },"
        "                    \"foot_angular_velocity\": {"
        "                        \"type\": \"array\","
        "                        \"items\": {"
        "                            \"type\": \"number\""
        "                        },"
        "                        \"minItems\": 3,"
        "                        \"maxItems\": 3,"
        "                        \"description\": \"3D foot angular velocity in world frame "
        "coordinates (rad/s)\""
        "                    }"
        "                },"
        "                \"required\": [\"name\", \"position\"]"
        "            },"
        "            \"description\": \"Array of contact data objects\""
        "        }"
        "    },"
        "    \"required\": [\"timestamp\", \"base_position\", \"base_orientation\", "
        "\"base_linear_velocity\", \"base_angular_velocity\", \"base_linear_acceleration\", "
        "\"base_angular_acceleration\", \"imu_linear_acceleration_bias\", "
        "\"imu_angular_velocity_bias\", \"num_contacts\", \"contact_names\", \"contacts\"]"
        "}";

    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

static inline mcap::Schema createCentroidalStateSchema() {
    mcap::Schema schema;
    schema.name = "CentroidalState";
    schema.encoding = "jsonschema";
    std::string schema_data =
        "{"
        "    \"type\": \"object\","
        "    \"properties\": {"
        "        \"timestamp\": {"
        "            \"type\": \"number\","
        "            \"description\": \"Timestamp of the state (s)\""
        "        },"
        "        \"com_position\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D CoM position in world frame coordinates (m)\""
        "        },"
        "        \"com_linear_velocity\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D CoM linear velocity in world frame coordinates "
        "(m/s)\""
        "        },"
        "        \"external_forces\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D External forces at the CoM in world frame "
        "coordinates (N)\""
        "        },"
        "        \"cop_position\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D COP position in world frame coordinates (m)\""
        "        },"
        "        \"com_linear_acceleration\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D CoM linear acceleration in world frame coordinates "
        "(m/s^2)\""
        "        },"
        "        \"angular_momentum\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Angular momentum around the CoM in world frame "
        "coordinates (kg m^2/s)\""
        "        },"
        "        \"angular_momentum_derivative\": {"
        "            \"type\": \"array\","
        "            \"items\": {"
        "                \"type\": \"number\""
        "            },"
        "            \"minItems\": 3,"
        "            \"maxItems\": 3,"
        "            \"description\": \"3D Angular momentum derivative around the CoM in world "
        "frame coordinates (Nm)\""
        "        }"
        "    },"
        "    \"required\": [\"timestamp\", \"com_position\", \"com_linear_velocity\", "
        "\"external_forces\", \"cop_position\", \"com_linear_acceleration\", "
        "\"angular_momentum\", \"angular_momentum_derivative\"]"
        "}";

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
