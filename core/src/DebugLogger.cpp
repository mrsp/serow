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
#include "DebugLogger.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <iomanip>
#include <chrono>

namespace serow {

// Implementation class definition
class DebugLogger::Impl {
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
            std::cerr << "DebugLogger initialization error: " << e.what() << std::endl;
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

    void log(const ImuMeasurement& imu_measurement) {
        try {
            // Create JSON message with explicit type handling
            nlohmann::json json_data = {
                {"timestamp", imu_measurement.timestamp},
                {"linear_acceleration", {
                    imu_measurement.linear_acceleration.x(),
                    imu_measurement.linear_acceleration.y(),
                    imu_measurement.linear_acceleration.z()
                }},
                {"angular_velocity", {
                    imu_measurement.angular_velocity.x(),
                    imu_measurement.angular_velocity.y(),
                    imu_measurement.angular_velocity.z()
                }}
            };

            // Convert JSON to string with consistent formatting
            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(1, imu_sequence_++, imu_measurement.timestamp, 
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging IMU measurement: " << e.what() << std::endl;
        }
    }

    void log(const std::map<std::string, JointMeasurement>& joints_measurement) {
        if (joints_measurement.empty()) return;

        try {
            nlohmann::json json_data = {
                {"timestamp", joints_measurement.begin()->second.timestamp},
                {"num_joints", joints_measurement.size()},
                {"joint_names", nlohmann::json::array()},
                {"joints", nlohmann::json::array()}
            };

            // Populate joint names and measurements
            for (const auto& [joint_name, _] : joints_measurement) {
                json_data["joint_names"].push_back(joint_name);
            }

            for (const auto& [_, measurement] : joints_measurement) {
                nlohmann::json joint_data = {
                    {"position", measurement.position}
                };
                if (measurement.velocity.has_value()) {
                    joint_data["velocity"] = measurement.velocity.value();
                }
                json_data["joints"].push_back(joint_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(2, joint_sequence_++, json_data["timestamp"],
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Joint measurement: " << e.what() << std::endl;
        }
    }

    void log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurements) {
        if (ft_measurements.empty()) return;

        try {
            nlohmann::json json_data = {
                {"timestamp", ft_measurements.begin()->second.timestamp},
                {"num_sensors", ft_measurements.size()},
                {"sensor_names", nlohmann::json::array()},
                {"sensors", nlohmann::json::array()}
            };

            // Populate sensor names
            for (const auto& [sensor_name, _] : ft_measurements) {
                json_data["sensor_names"].push_back(sensor_name);
            }

            // Populate sensor measurements
            for (const auto& [_, measurement] : ft_measurements) {
                nlohmann::json sensor_data = {
                    {"force", {
                        measurement.force.x(), 
                        measurement.force.y(), 
                        measurement.force.z()
                    }},
                    {"cop", {
                        measurement.cop.x(), 
                        measurement.cop.y(), 
                        measurement.cop.z()
                    }}
                };

                if (measurement.torque.has_value()) {
                    sensor_data["torque"] = {
                        measurement.torque.value().x(),
                        measurement.torque.value().y(),
                        measurement.torque.value().z()
                    };
                }

                json_data["sensors"].push_back(sensor_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(3, ft_sequence_++, json_data["timestamp"],
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Force-Torque measurement: " << e.what() << std::endl;
        }
    }

    void log(const ContactState& contact_state) {
        try {
            nlohmann::json json_data = {
                {"timestamp", contact_state.timestamp},
                {"num_contacts", contact_state.contacts_status.size()},
                {"contact_names", nlohmann::json::array()},
                {"contacts", nlohmann::json::array()}
            };

            // Populate contact names
            for (const auto& [contact_name, _] : contact_state.contacts_status) {
                json_data["contact_names"].push_back(contact_name);
            }

            // Populate contact details
            for (const auto& [contact_name, status] : contact_state.contacts_status) {
                nlohmann::json contact_data = {
                    {"status", status},
                    {"probability", contact_state.contacts_probability.at(contact_name)},
                    {"force", {
                        contact_state.contacts_force.at(contact_name).x(),
                        contact_state.contacts_force.at(contact_name).y(),
                        contact_state.contacts_force.at(contact_name).z()
                    }}
                };

                if (contact_state.contacts_torque.has_value()) {
                    const auto& torque = contact_state.contacts_torque.value();
                    if (torque.count(contact_name) > 0) {
                        contact_data["torque"] = {
                            torque.at(contact_name).x(),
                            torque.at(contact_name).y(),
                            torque.at(contact_name).z()
                        };
                    }
                }

                json_data["contacts"].push_back(contact_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(4, contact_sequence_++, contact_state.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Contact State: " << e.what() << std::endl;
        }
    }

    void log(const CentroidalState& centroidal_state) {
        try {
            nlohmann::json json_data = {
                {"timestamp", centroidal_state.timestamp},
                {"com_position", {
                    centroidal_state.com_position.x(),
                    centroidal_state.com_position.y(),
                    centroidal_state.com_position.z()
                }},
                {"com_linear_velocity", {
                    centroidal_state.com_linear_velocity.x(),
                    centroidal_state.com_linear_velocity.y(),
                    centroidal_state.com_linear_velocity.z()
                }},
                {"external_forces", {
                    centroidal_state.external_forces.x(),
                    centroidal_state.external_forces.y(),
                    centroidal_state.external_forces.z()
                }},
                {"cop_position", {
                    centroidal_state.cop_position.x(),
                    centroidal_state.cop_position.y(),
                    centroidal_state.cop_position.z()
                }},
                {"com_linear_acceleration", {
                    centroidal_state.com_linear_acceleration.x(),
                    centroidal_state.com_linear_acceleration.y(),
                    centroidal_state.com_linear_acceleration.z()
                }},
                {"angular_momentum", {
                    centroidal_state.angular_momentum.x(),
                    centroidal_state.angular_momentum.y(),
                    centroidal_state.angular_momentum.z()
                }},
                {"angular_momentum_derivative", {
                    centroidal_state.angular_momentum_derivative.x(),
                    centroidal_state.angular_momentum_derivative.y(),
                    centroidal_state.angular_momentum_derivative.z()
                }}
            };

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(5, centroidal_sequence_++, centroidal_state.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Centroidal State: " << e.what() << std::endl;
        }
    }

    void log(const BaseState& base_state) {
        try {
            nlohmann::json json_data = {
                {"timestamp", base_state.timestamp},
                {"base_position", {
                    base_state.base_position.x(),
                    base_state.base_position.y(),
                    base_state.base_position.z()
                }},
                {"base_orientation", {
                    base_state.base_orientation.w(),
                    base_state.base_orientation.x(),
                    base_state.base_orientation.y(),
                    base_state.base_orientation.z()
                }},
                {"base_linear_velocity", {
                    base_state.base_linear_velocity.x(),
                    base_state.base_linear_velocity.y(),
                    base_state.base_linear_velocity.z()
                }},
                {"base_angular_velocity", {
                    base_state.base_angular_velocity.x(),
                    base_state.base_angular_velocity.y(),
                    base_state.base_angular_velocity.z()
                }},
                {"base_linear_acceleration", {
                    base_state.base_linear_acceleration.x(),
                    base_state.base_linear_acceleration.y(),
                    base_state.base_linear_acceleration.z()
                }},
                {"base_angular_acceleration", {
                    base_state.base_angular_acceleration.x(),
                    base_state.base_angular_acceleration.y(),
                    base_state.base_angular_acceleration.z()
                }},
                {"imu_linear_acceleration_bias", {
                    base_state.imu_linear_acceleration_bias.x(),
                    base_state.imu_linear_acceleration_bias.y(),
                    base_state.imu_linear_acceleration_bias.z()
                }},
                {"imu_angular_velocity_bias", {
                    base_state.imu_angular_velocity_bias.x(),
                    base_state.imu_angular_velocity_bias.y(),
                    base_state.imu_angular_velocity_bias.z()
                }},
                {"num_contacts", base_state.contacts_position.size()},
                {"contact_names", nlohmann::json::array()},
                {"contacts", nlohmann::json::array()}
            };

            // Populate contact data
            for (const auto& [name, pos] : base_state.contacts_position) {
                json_data["contact_names"].push_back(name);
                
                nlohmann::json contact_data = {
                    {"name", name},
                    {"position", {
                        pos.x(),
                        pos.y(),
                        pos.z()
                    }}
                };

                // Add optional foot position if available
                if (base_state.feet_position.find(name) != base_state.feet_position.end()) {
                    const auto& foot_pos = base_state.feet_position.at(name);
                    contact_data["foot_position"] = {
                        foot_pos.x(),
                        foot_pos.y(),
                        foot_pos.z()
                    };
                }

                // Add optional foot orientation if available
                if (base_state.feet_orientation.find(name) != base_state.feet_orientation.end()) {
                    const auto& foot_ori = base_state.feet_orientation.at(name);
                    contact_data["foot_orientation"] = {
                        foot_ori.w(),
                        foot_ori.x(),
                        foot_ori.y(),
                        foot_ori.z()
                    };
                }

                // Add optional foot linear velocity if available
                if (base_state.feet_linear_velocity.find(name) != base_state.feet_linear_velocity.end()) {
                    const auto& foot_vel = base_state.feet_linear_velocity.at(name);
                    contact_data["foot_linear_velocity"] = {
                        foot_vel.x(),
                        foot_vel.y(),
                        foot_vel.z()
                    };
                }

                // Add optional foot angular velocity if available
                if (base_state.feet_angular_velocity.find(name) != base_state.feet_angular_velocity.end()) {
                    const auto& foot_ang_vel = base_state.feet_angular_velocity.at(name);
                    contact_data["foot_angular_velocity"] = {
                        foot_ang_vel.x(),
                        foot_ang_vel.y(),
                        foot_ang_vel.z()
                    };
                }

                json_data["contacts"].push_back(contact_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(6, base_state_sequence_++, base_state.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Base State: " << e.what() << std::endl;
        }
    }

    void log(const std::pair<double, std::array<std::array<float, 3>, map_size>>& local_map_state) {
        try {
            const double timestamp = local_map_state.first;
            const auto& local_map = local_map_state.second;
            
            nlohmann::json json_data;
            json_data["map_size"] = map_size;
            
            // Create array of points
            nlohmann::json points = nlohmann::json::array();
            
            for (size_t i = 0; i < 1000; ++i) {
                    const std::array<float, 3>& point = local_map[i];
                    points.push_back({
                        static_cast<double>(point[0]),
                        static_cast<double>(point[1]),
                        static_cast<double>(point[2])
                    });
            }
            
            json_data["points"] = points;
            
            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(7, local_map_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), 
                         json_str.size());
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
                                         std::to_string(channel_id) + 
                                         ": " + status.message);
            }
        } catch (const std::exception& e) {
            std::cerr << "MCAP write error: " << e.what() << std::endl;
        }
    }

    // Reuse the original schema and channel initialization methods
    void initializeSchemas() {
        std::vector<mcap::Schema> schemas;
        schemas.push_back(createImuMeasurementSchema());
        schemas.push_back(createJointMeasurementSchema());
        schemas.push_back(createForceTorqueMeasurementSchema());
        schemas.push_back(createContactStateSchema());
        schemas.push_back(createCentroidalStateSchema());
        schemas.push_back(createBaseStateSchema());
        schemas.push_back(createLocalMapSchema());

        for (auto& schema : schemas) {
            writer_->addSchema(schema);
        }
    }

    void initializeChannels() {
        std::vector<mcap::Channel> channels;
        channels.push_back(createChannel(1, "ImuMeasurement"));
        channels.push_back(createChannel(2, "JointMeasurement"));
        channels.push_back(createChannel(3, "ForceTorqueMeasurement"));
        channels.push_back(createChannel(4, "ContactState"));
        channels.push_back(createChannel(5, "CentroidalState"));
        channels.push_back(createChannel(6, "BaseState"));
        channels.push_back(createChannel(7, "LocalMap"));

        for (auto& channel : channels) {
            writer_->addChannel(channel);
        }
    }

 mcap::Schema createImuMeasurementSchema() {
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

    mcap::Schema createJointMeasurementSchema() {
        mcap::Schema schema;
        schema.name = "JointMeasurement";
        schema.encoding = "jsonschema";
        std::string schema_data =
            "{"
            "    \"type\": \"object\","
            "    \"properties\": {"
            "        \"timestamp\": {"
            "            \"type\": \"number\","
            "            \"description\": \"Timestamp of the measurement (s)\""
            "        },"
            "        \"num_joints\": {"
            "            \"type\": \"integer\","
            "            \"description\": \"Number of joints in the measurement\""
            "        },"
            "        \"joint_names\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"string\""
            "            },"
            "            \"description\": \"Array of joint names\""
            "        },"
            "        \"joints\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"object\","
            "                \"properties\": {"
            "                    \"position\": {"
            "                        \"type\": \"number\","
            "                        \"description\": \"Joint position (rad)\""
            "                    },"
            "                    \"velocity\": {"
            "                        \"type\": \"number\","
            "                        \"description\": \"Joint velocity (rad/s)\""
            "                    }"
            "                },"
            "                \"required\": [\"position\"]"
            "            }"
            "        }"
            "    },"
            "    \"required\": [\"timestamp\", \"num_joints\", \"joint_names\", \"joints\"]"
            "}";

        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createForceTorqueMeasurementSchema() {
        mcap::Schema schema;
        schema.name = "ForceTorqueMeasurement";
        schema.encoding = "jsonschema";
        std::string schema_data =
            "{"
            "    \"type\": \"object\","
            "    \"properties\": {"
            "        \"timestamp\": {"
            "            \"type\": \"number\","
            "            \"description\": \"Timestamp of the measurement (s)\""
            "        },"
            "        \"num_sensors\": {"
            "            \"type\": \"integer\","
            "            \"description\": \"Number of force-torque sensors in the measurement\""
            "        },"
            "        \"sensor_names\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"string\""
            "            },"
            "            \"description\": \"Array of force-torque sensor names\""
            "        },"
            "        \"sensors\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"object\","
            "                \"properties\": {"
            "                    \"force\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"Force measured by force-torque sensor (N)\""
            "                    },"
            "                    \"cop\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"Center of pressure (COP) measured by force-torque sensor (m)\""
            "                    },"
            "                    \"torque\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"Optional torque measured by force-torque sensor (Nm)\""
            "                    }"
            "                },"
            "                \"required\": [\"force\", \"cop\"]"
            "            }"
            "        }"
            "    },"
            "    \"required\": [\"timestamp\", \"num_sensors\", \"sensor_names\", \"sensors\"]"
            "}";

        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createContactStateSchema() {
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
            "                        \"description\": \"3D Contact force in world frame coordinates (N)\""
            "                    },"
            "                    \"torque\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"3D Contact torque in world frame coordinates (Nm)\""
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

    mcap::Schema createCentroidalStateSchema() {
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
            "            \"description\": \"3D CoM linear velocity in world frame coordinates (m/s)\""
            "        },"
            "        \"external_forces\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D External forces at the CoM in world frame coordinates (N)\""
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
            "            \"description\": \"3D CoM linear acceleration in world frame coordinates (m/s^2)\""
            "        },"
            "        \"angular_momentum\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D Angular momentum around the CoM in world frame coordinates (kg m^2/s)\""
            "        },"
            "        \"angular_momentum_derivative\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D Angular momentum derivative around the CoM in world frame coordinates (Nm)\""
            "        }"
            "    },"
            "    \"required\": [\"timestamp\", \"com_position\", \"com_linear_velocity\", \"external_forces\", \"cop_position\", \"com_linear_acceleration\", \"angular_momentum\", \"angular_momentum_derivative\"]"
            "}";

        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createBaseStateSchema() {
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
            "            \"description\": \"Base orientation quaternion [w, x, y, z] in world frame coordinates\""
            "        },"
            "        \"base_linear_velocity\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D Base linear velocity in world frame coordinates (m/s)\""
            "        },"
            "        \"base_angular_velocity\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D Base angular velocity in world frame coordinates (rad/s)\""
            "        },"
            "        \"base_linear_acceleration\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D Base linear acceleration in world frame coordinates (m/s^2)\""
            "        },"
            "        \"base_angular_acceleration\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D Base angular acceleration in world frame coordinates (rad/s^2)\""
            "        },"
            "        \"imu_linear_acceleration_bias\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D IMU linear acceleration bias in IMU frame coordinates (m/s^2)\""
            "        },"
            "        \"imu_angular_velocity_bias\": {"
            "            \"type\": \"array\","
            "            \"items\": {"
            "                \"type\": \"number\""
            "            },"
            "            \"minItems\": 3,"
            "            \"maxItems\": 3,"
            "            \"description\": \"3D IMU angular velocity bias in IMU frame coordinates (rad/s)\""
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
            "                        \"description\": \"3D contact position in world frame coordinates (m)\""
            "                    },"
            "                    \"foot_position\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"3D foot position in world frame coordinates (m)\""
            "                    },"
            "                    \"foot_orientation\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 4,"
            "                        \"maxItems\": 4,"
            "                        \"description\": \"Foot orientation quaternion [w, x, y, z] in world frame coordinates\""
            "                    },"
            "                    \"foot_linear_velocity\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"3D foot linear velocity in world frame coordinates (m/s)\""
            "                    },"
            "                    \"foot_angular_velocity\": {"
            "                        \"type\": \"array\","
            "                        \"items\": {"
            "                            \"type\": \"number\""
            "                        },"
            "                        \"minItems\": 3,"
            "                        \"maxItems\": 3,"
            "                        \"description\": \"3D foot angular velocity in world frame coordinates (rad/s)\""
            "                    }"
            "                },"
            "                \"required\": [\"name\", \"position\"]"
            "            },"
            "            \"description\": \"Array of contact data objects\""
            "        }"
            "    },"
            "    \"required\": [\"timestamp\", \"base_position\", \"base_orientation\", \"base_linear_velocity\", \"base_angular_velocity\", \"base_linear_acceleration\", \"base_angular_acceleration\", \"imu_linear_acceleration_bias\", \"imu_angular_velocity_bias\", \"num_contacts\", \"contact_names\", \"contacts\"]"
            "}";

        schema.data = mcap::ByteArray(
            reinterpret_cast<const std::byte*>(schema_data.data()),
            reinterpret_cast<const std::byte*>(schema_data.data() + schema_data.size()));
        return schema;
    }

    mcap::Schema createLocalMapSchema() {
        mcap::Schema schema;
        schema.name = "LocalMap";
        schema.encoding = "jsonschema";
        std::string schema_data =
            "{"
            "    \"type\": \"object\","
            "    \"properties\": {"
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
    uint64_t base_state_sequence_ = 0;
    uint64_t centroidal_sequence_ = 0;
    uint64_t contact_sequence_ = 0;
    uint64_t imu_sequence_ = 0;
    uint64_t joint_sequence_ = 0;
    uint64_t ft_sequence_ = 0;
    uint64_t local_map_sequence_ = 0;
    double last_local_map_timestamp_{-1.0};

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
};

// Public interface implementation
DebugLogger::DebugLogger(const std::string& filename) : pimpl_(std::make_unique<Impl>(filename)) {}

DebugLogger::~DebugLogger() = default;

void DebugLogger::log(const BaseState& base_state) { pimpl_->log(base_state); }

void DebugLogger::log(const CentroidalState& centroidal_state) { pimpl_->log(centroidal_state); }

void DebugLogger::log(const ImuMeasurement& imu_measurement) {
    pimpl_->log(imu_measurement);
}

void DebugLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {
    pimpl_->log(joints_measurement);
}

void DebugLogger::log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurements) {
    pimpl_->log(ft_measurements);
}

void DebugLogger::log(const ContactState& contact_state) {
    pimpl_->log(contact_state);
}

void DebugLogger::log(const std::pair<double, std::array<std::array<float, 3>, map_size>>& local_map_state) {
    pimpl_->log(local_map_state);
}

double DebugLogger::getLastLocalMapTimestamp() const {
    return pimpl_->getLastLocalMapTimestamp();
}

}  // namespace serow
