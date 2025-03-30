#pragma once

#include <filesystem>
#include <string>

#include "mcap/mcap.hpp"

namespace serow {

static inline mcap::Schema createSchema(const std::string& schema_name) {
    // Load the .bfbs file
    std::string bfbs_path = schema_name + ".bfbs";

    // Add the source directory path if defined
#ifdef SCHEMAS_DIR
    bfbs_path = std::filesystem::path(SCHEMAS_DIR) / (schema_name + ".bfbs");
#endif
    std::ifstream schema_file(bfbs_path, std::ios::binary);
    if (!schema_file.is_open()) {
        throw std::runtime_error("Failed to open " + schema_name + ".bfbs at " + bfbs_path);
    }

    // Read the entire file into a vector
    std::vector<uint8_t> schema_data((std::istreambuf_iterator<char>(schema_file)),
                                     std::istreambuf_iterator<char>());
    schema_file.close();

    if (schema_data.empty()) {
        throw std::runtime_error(schema_name + ".bfbs is empty");
    }

    // Create schema using .bfbs data
    mcap::Schema schema;
    schema.name = "foxglove." + schema_name;
    schema.encoding = "flatbuffer";
    schema.data = mcap::ByteArray(
        reinterpret_cast<const std::byte*>(schema_data.data()),
        reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
    return schema;
}

}  // namespace serow
