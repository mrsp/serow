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
#include <nlohmann/json.hpp>

namespace serow {

// Implementation class definition
class ProprioceptionLogger::Impl {
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

    // Utility function to convert timestamp to nanoseconds
    uint64_t convertToNanoseconds(double timestamp) {
        // Ensure timestamp is converted to nanoseconds
        return static_cast<uint64_t>(timestamp * 1e9);
    }

    void log(const std::map<std::string, Eigen::Vector3d>& positions,
             const std::map<std::string, Eigen::Quaterniond>& orientations, double timestamp) {
        nlohmann::json json_data = {{"timestamp", timestamp},
                                    {"transforms", nlohmann::json::array()}};

        for (const auto& [frame_id, position] : positions) {
            nlohmann::json transform = {
                {"timestamp",
                 {{"sec", static_cast<int>(timestamp)},
                  {"nsec", static_cast<int>((timestamp - static_cast<int>(timestamp)) * 1e9)}}},
                {"parent_frame_id", "world"},
                {"child_frame_id", frame_id},
                {"translation", {{"x", position.x()}, {"y", position.y()}, {"z", position.z()}}},
                {"rotation",
                 {{"x", orientations.at(frame_id).x()},
                  {"y", orientations.at(frame_id).y()},
                  {"z", orientations.at(frame_id).z()},
                  {"w", orientations.at(frame_id).w()}}}};
            json_data["transforms"].push_back(transform);
        }

        std::string json_str = json_data.dump(-1, ' ', true);

        writeMessage(8, leg_tf_sequence_++, timestamp,
                     reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
    }

    void log(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation,
             double timestamp) {
        try {
            nlohmann::json json_data = {
                {"timestamp",
                 {{"sec", static_cast<int>(timestamp)},
                  {"nsec", static_cast<int>((timestamp - static_cast<int>(timestamp)) * 1e9)}}},
                {"parent_frame_id", "world"},
                {"child_frame_id", "base_link"},
                {"translation", {{"x", position.x()}, {"y", position.y()}, {"z", position.z()}}},
                {"rotation",
                 {{"x", orientation.x()},
                  {"y", orientation.y()},
                  {"z", orientation.z()},
                  {"w", orientation.w()}}}};

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(7, tf_sequence_++, timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Proprioception: " << e.what() << std::endl;
        }
    }

    void log(const ImuMeasurement& imu_measurement) {
        try {
            // Create JSON message with explicit type handling
            nlohmann::json json_data = {
                {"timestamp", imu_measurement.timestamp},
                {"linear_acceleration",
                 {imu_measurement.linear_acceleration.x(), imu_measurement.linear_acceleration.y(),
                  imu_measurement.linear_acceleration.z()}},
                {"angular_velocity",
                 {imu_measurement.angular_velocity.x(), imu_measurement.angular_velocity.y(),
                  imu_measurement.angular_velocity.z()}}};

            // Convert JSON to string with consistent formatting
            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(1, imu_sequence_++, imu_measurement.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging IMU measurement: " << e.what() << std::endl;
        }
    }

    void log(const std::map<std::string, JointMeasurement>& joints_measurement) {
        if (joints_measurement.empty())
            return;

        try {
            nlohmann::json json_data = {{"timestamp", joints_measurement.begin()->second.timestamp},
                                        {"num_joints", joints_measurement.size()},
                                        {"joint_names", nlohmann::json::array()},
                                        {"joints", nlohmann::json::array()}};

            // Populate joint names and measurements
            for (const auto& [joint_name, _] : joints_measurement) {
                json_data["joint_names"].push_back(joint_name);
            }

            for (const auto& [_, measurement] : joints_measurement) {
                nlohmann::json joint_data = {{"position", measurement.position}};
                if (measurement.velocity.has_value()) {
                    joint_data["velocity"] = measurement.velocity.value();
                }
                json_data["joints"].push_back(joint_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(2, joint_sequence_++, json_data["timestamp"],
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Joint measurement: " << e.what() << std::endl;
        }
    }

    void log(const std::map<std::string, ForceTorqueMeasurement>& ft_measurements) {
        if (ft_measurements.empty())
            return;

        try {
            nlohmann::json json_data = {{"timestamp", ft_measurements.begin()->second.timestamp},
                                        {"num_sensors", ft_measurements.size()},
                                        {"sensor_names", nlohmann::json::array()},
                                        {"sensors", nlohmann::json::array()}};

            // Populate sensor names
            for (const auto& [sensor_name, _] : ft_measurements) {
                json_data["sensor_names"].push_back(sensor_name);
            }

            // Populate sensor measurements
            for (const auto& [_, measurement] : ft_measurements) {
                nlohmann::json sensor_data = {
                    {"force",
                     {measurement.force.x(), measurement.force.y(), measurement.force.z()}},
                    {"cop", {measurement.cop.x(), measurement.cop.y(), measurement.cop.z()}}};

                if (measurement.torque.has_value()) {
                    sensor_data["torque"] = {measurement.torque.value().x(),
                                             measurement.torque.value().y(),
                                             measurement.torque.value().z()};
                }

                json_data["sensors"].push_back(sensor_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(3, ft_sequence_++, json_data["timestamp"],
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Force-Torque measurement: " << e.what() << std::endl;
        }
    }

    void log(const ContactState& contact_state) {
        try {
            nlohmann::json json_data = {{"timestamp", contact_state.timestamp},
                                        {"num_contacts", contact_state.contacts_status.size()},
                                        {"contact_names", nlohmann::json::array()},
                                        {"contacts", nlohmann::json::array()}};

            // Populate contact names
            for (const auto& [contact_name, _] : contact_state.contacts_status) {
                json_data["contact_names"].push_back(contact_name);
            }

            // Populate contact details
            for (const auto& [contact_name, status] : contact_state.contacts_status) {
                nlohmann::json contact_data = {
                    {"status", status},
                    {"probability", contact_state.contacts_probability.at(contact_name)},
                    {"force",
                     {contact_state.contacts_force.at(contact_name).x(),
                      contact_state.contacts_force.at(contact_name).y(),
                      contact_state.contacts_force.at(contact_name).z()}}};

                if (contact_state.contacts_torque.has_value()) {
                    const auto& torque = contact_state.contacts_torque.value();
                    if (torque.count(contact_name) > 0) {
                        contact_data["torque"] = {torque.at(contact_name).x(),
                                                  torque.at(contact_name).y(),
                                                  torque.at(contact_name).z()};
                    }
                }

                json_data["contacts"].push_back(contact_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(4, contact_sequence_++, contact_state.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Contact State: " << e.what() << std::endl;
        }
    }

    void log(const CentroidalState& centroidal_state) {
        try {
            nlohmann::json json_data = {
                {"timestamp", centroidal_state.timestamp},
                {"com_position",
                 {centroidal_state.com_position.x(), centroidal_state.com_position.y(),
                  centroidal_state.com_position.z()}},
                {"com_linear_velocity",
                 {centroidal_state.com_linear_velocity.x(),
                  centroidal_state.com_linear_velocity.y(),
                  centroidal_state.com_linear_velocity.z()}},
                {"external_forces",
                 {centroidal_state.external_forces.x(), centroidal_state.external_forces.y(),
                  centroidal_state.external_forces.z()}},
                {"cop_position",
                 {centroidal_state.cop_position.x(), centroidal_state.cop_position.y(),
                  centroidal_state.cop_position.z()}},
                {"com_linear_acceleration",
                 {centroidal_state.com_linear_acceleration.x(),
                  centroidal_state.com_linear_acceleration.y(),
                  centroidal_state.com_linear_acceleration.z()}},
                {"angular_momentum",
                 {centroidal_state.angular_momentum.x(), centroidal_state.angular_momentum.y(),
                  centroidal_state.angular_momentum.z()}},
                {"angular_momentum_derivative",
                 {centroidal_state.angular_momentum_derivative.x(),
                  centroidal_state.angular_momentum_derivative.y(),
                  centroidal_state.angular_momentum_derivative.z()}}};

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(5, centroidal_sequence_++, centroidal_state.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Centroidal State: " << e.what() << std::endl;
        }
    }

    void log(const BaseState& base_state) {
        try {
            nlohmann::json json_data = {
                {"timestamp", base_state.timestamp},
                {"base_position",
                 {base_state.base_position.x(), base_state.base_position.y(),
                  base_state.base_position.z()}},
                {"base_orientation",
                 {base_state.base_orientation.w(), base_state.base_orientation.x(),
                  base_state.base_orientation.y(), base_state.base_orientation.z()}},
                {"base_linear_velocity",
                 {base_state.base_linear_velocity.x(), base_state.base_linear_velocity.y(),
                  base_state.base_linear_velocity.z()}},
                {"base_angular_velocity",
                 {base_state.base_angular_velocity.x(), base_state.base_angular_velocity.y(),
                  base_state.base_angular_velocity.z()}},
                {"base_linear_acceleration",
                 {base_state.base_linear_acceleration.x(), base_state.base_linear_acceleration.y(),
                  base_state.base_linear_acceleration.z()}},
                {"base_angular_acceleration",
                 {base_state.base_angular_acceleration.x(),
                  base_state.base_angular_acceleration.y(),
                  base_state.base_angular_acceleration.z()}},
                {"imu_linear_acceleration_bias",
                 {base_state.imu_linear_acceleration_bias.x(),
                  base_state.imu_linear_acceleration_bias.y(),
                  base_state.imu_linear_acceleration_bias.z()}},
                {"imu_angular_velocity_bias",
                 {base_state.imu_angular_velocity_bias.x(),
                  base_state.imu_angular_velocity_bias.y(),
                  base_state.imu_angular_velocity_bias.z()}},
                {"num_contacts", base_state.contacts_position.size()},
                {"contact_names", nlohmann::json::array()},
                {"contacts", nlohmann::json::array()}};

            // Populate contact data
            for (const auto& [name, pos] : base_state.contacts_position) {
                json_data["contact_names"].push_back(name);

                nlohmann::json contact_data = {{"name", name},
                                               {"position", {pos.x(), pos.y(), pos.z()}}};

                // Add optional foot position if available
                if (base_state.feet_position.find(name) != base_state.feet_position.end()) {
                    const auto& foot_pos = base_state.feet_position.at(name);
                    contact_data["foot_position"] = {foot_pos.x(), foot_pos.y(), foot_pos.z()};
                }

                // Add optional foot orientation if available
                if (base_state.feet_orientation.find(name) != base_state.feet_orientation.end()) {
                    const auto& foot_ori = base_state.feet_orientation.at(name);
                    contact_data["foot_orientation"] = {foot_ori.w(), foot_ori.x(), foot_ori.y(),
                                                        foot_ori.z()};
                }

                // Add optional foot linear velocity if available
                if (base_state.feet_linear_velocity.find(name) !=
                    base_state.feet_linear_velocity.end()) {
                    const auto& foot_vel = base_state.feet_linear_velocity.at(name);
                    contact_data["foot_linear_velocity"] = {foot_vel.x(), foot_vel.y(),
                                                            foot_vel.z()};
                }

                // Add optional foot angular velocity if available
                if (base_state.feet_angular_velocity.find(name) !=
                    base_state.feet_angular_velocity.end()) {
                    const auto& foot_ang_vel = base_state.feet_angular_velocity.at(name);
                    contact_data["foot_angular_velocity"] = {foot_ang_vel.x(), foot_ang_vel.y(),
                                                             foot_ang_vel.z()};
                }

                json_data["contacts"].push_back(contact_data);
            }

            std::string json_str = json_data.dump(-1, ' ', true);

            writeMessage(6, base_state_sequence_++, base_state.timestamp,
                         reinterpret_cast<const std::byte*>(json_str.data()), json_str.size());
        } catch (const std::exception& e) {
            std::cerr << "Error logging Base State: " << e.what() << std::endl;
        }
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
        schemas.push_back(createImuMeasurementSchema());
        schemas.push_back(createJointMeasurementSchema());
        schemas.push_back(createForceTorqueMeasurementSchema());
        schemas.push_back(createContactStateSchema());
        schemas.push_back(createCentroidalStateSchema());
        schemas.push_back(createBaseStateSchema());
        schemas.push_back(createTFSchema());
        schemas.push_back(createFrameTransformsSchema());

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
        channels.push_back(createChannel(7, "/odom"));
        channels.push_back(createChannel(8, "/legs"));

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
    uint64_t base_state_sequence_ = 0;
    uint64_t centroidal_sequence_ = 0;
    uint64_t contact_sequence_ = 0;
    uint64_t imu_sequence_ = 0;
    uint64_t joint_sequence_ = 0;
    uint64_t ft_sequence_ = 0;
    uint64_t tf_sequence_ = 0;
    uint64_t leg_tf_sequence_ = 0;

    // MCAP writing components
    std::unique_ptr<mcap::FileWriter> file_writer_;
    std::unique_ptr<mcap::McapWriter> writer_;
};

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

void ProprioceptionLogger::log(const std::map<std::string, JointMeasurement>& joints_measurement) {
    pimpl_->log(joints_measurement);
}

void ProprioceptionLogger::log(
    const std::map<std::string, ForceTorqueMeasurement>& ft_measurements) {
    pimpl_->log(ft_measurements);
}

void ProprioceptionLogger::log(const ContactState& contact_state) {
    pimpl_->log(contact_state);
}

void ProprioceptionLogger::log(const Eigen::Vector3d& position,
                               const Eigen::Quaterniond& orientation, double timestamp) {
    pimpl_->log(position, orientation, timestamp);
}

void ProprioceptionLogger::log(const std::map<std::string, Eigen::Vector3d>& positions,
                               const std::map<std::string, Eigen::Quaterniond>& orientations,
                               double timestamp) {
    pimpl_->log(positions, orientations, timestamp);
}

}  // namespace serow
