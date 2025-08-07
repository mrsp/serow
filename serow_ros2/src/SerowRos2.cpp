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
#include "SerowRos2.hpp"

#include <filesystem>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "yaml-cpp/yaml.h"

using std::placeholders::_1;
namespace fs = std::filesystem;

SerowRos2::SerowRos2() : Node("serow_ros2_driver") {
    // Get config file parameter
    auto config_file_param = this->declare_parameter<std::string>("config_file", "");

    if (config_file_param.empty()) {
        RCLCPP_ERROR(this->get_logger(),
                     "No config file specified. Use --ros-args -p config_file:=<path>");
        throw std::runtime_error("No config file specified");
    }

    // Check if the file exists
    if (!fs::exists(config_file_param)) {
        RCLCPP_ERROR(this->get_logger(), "Config file '%s' does not exist",
                     config_file_param.c_str());
        throw std::runtime_error("Config file does not exist: " + config_file_param);
    }

    // Load configuration from YAML file
    YAML::Node config;
    try {
        RCLCPP_INFO(this->get_logger(), "Loading configuration from: %s",
                    config_file_param.c_str());
        config = YAML::LoadFile(config_file_param);
    } catch (const YAML::BadFile& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load YAML file: %s", e.what());
        RCLCPP_ERROR(this->get_logger(), "File path: %s", config_file_param.c_str());
        throw;
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error parsing YAML file: %s", e.what());
        throw;
    }

    // Extract configuration values
    const std::string robot_name = config["robot_name"].as<std::string>();
    const std::string joint_state_topic = config["topics"]["joint_states"].as<std::string>();
    const std::string base_imu_topic = config["topics"]["imu"].as<std::string>();
    std::vector<std::string> force_torque_state_topics;
    for (const auto& topic : config["topics"]["force_torque_states"]) {
        force_torque_state_topics.push_back(topic.as<std::string>());
    }
    const std::string serow_config = config["serow_config"].as<std::string>();

    RCLCPP_INFO(this->get_logger(), "Robot name: %s", robot_name.c_str());
    RCLCPP_INFO(this->get_logger(), "Serow config file: %s", serow_config.c_str());

    // Initialize SERoW
    try {
        serow_.initialize(serow_config);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize SERoW: %s", e.what());
        throw;
    }

    // Create subscribers
    RCLCPP_INFO(this->get_logger(), "Creating joint state subscription on topic: %s",
                joint_state_topic.c_str());
    joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
        joint_state_topic, 10, std::bind(&SerowRos2::jointStateCallback, this, _1));

    RCLCPP_INFO(this->get_logger(), "Creating base imu subscription on topic: %s",
                base_imu_topic.c_str());
    base_imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
        base_imu_topic, 10, std::bind(&SerowRos2::baseImuCallback, this, _1));

    // Dynamically create a wrench callback one for each limb
    for (const auto& ft_topic : force_torque_state_topics) {
        auto force_torque_state_topic_callback =
            [this](const geometry_msgs::msg::WrenchStamped& msg) {
                this->ft_data_[msg.header.frame_id] = msg;
            };
        force_torque_state_topic_callbacks_.push_back(std::move(force_torque_state_topic_callback));
        RCLCPP_INFO(this->get_logger(), "Creating force torque state subscription on topic: %s",
                    ft_topic.c_str());
        auto ft_subscription = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            ft_topic, 1, force_torque_state_topic_callbacks_.back());
        force_torque_state_subscriptions_.push_back(std::move(ft_subscription));
    }

    // Create a timer for processing data instead of the blocking loop, max 1000Hz processing
    // rate (if applicable)
    timer_ =
        this->create_wall_timer(std::chrono::milliseconds(1), std::bind(&SerowRos2::run, this));

    const auto& state = serow_.getState(true);
    if (!state) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get state during initialization");
        throw std::runtime_error("Failed to get state during initialization");
    }

    com_position_publisher_ =
        this->create_publisher<geometry_msgs::msg::PointStamped>("/serow/com/position", 10);
    cop_position_publisher_ =
        this->create_publisher<geometry_msgs::msg::PointStamped>("/serow/cop/position", 10);
    com_momentum_publisher_ =
        this->create_publisher<geometry_msgs::msg::TwistStamped>("/serow/com/momentum", 10);
    com_momentum_rate_publisher_ =
        this->create_publisher<geometry_msgs::msg::TwistStamped>("/serow/com/momentum_rate", 10);
    com_external_wrench_publisher_ =
        this->create_publisher<geometry_msgs::msg::WrenchStamped>("/serow/com/external_wrench", 10);
    for (const auto& contact_frame : state->getContactsFrame()) {
        foot_odom_publishers_[contact_frame] = this->create_publisher<nav_msgs::msg::Odometry>(
            "/serow/" + contact_frame + "/odom", 10);
        foot_wrench_publishers_[contact_frame] =
            this->create_publisher<geometry_msgs::msg::WrenchStamped>(
                "/serow/" + contact_frame + "/contact/wrench", 10);
        foot_contact_probability_publishers_[contact_frame] =
            this->create_publisher<std_msgs::msg::Float64>(
                "/serow/" + contact_frame + "/contact/probability", 10);
    }
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/serow/" + state->getBaseFrame() + "/odom", 10);
    joint_state_publisher_ =
        this->create_publisher<sensor_msgs::msg::JointState>("/serow/joint_states", 10);

    // Start the publishing thread
    publishing_thread_ = std::thread(&SerowRos2::publish, this);

    RCLCPP_INFO(this->get_logger(), "SERoW was initialized successfully");
}

SerowRos2::~SerowRos2() {
    // Signal the publishing thread to stop
    {
        std::lock_guard<std::mutex> lock(publish_queue_mutex_);
        shutdown_requested_ = true;
    }
    publish_condition_.notify_one();

    // Wait for the publishing thread to finish
    if (publishing_thread_.joinable()) {
        publishing_thread_.join();
    }
}

void SerowRos2::run() {
    if (joint_state_data_.has_value() && base_imu_data_.has_value()) {
        // Create the joint measurements
        const auto& joint_state_data = joint_state_data_.value();
        std::map<std::string, serow::JointMeasurement> joint_measurements;
        for (unsigned int i = 0; i < joint_state_data.name.size(); i++) {
            serow::JointMeasurement joint{};
            joint.timestamp = static_cast<double>(joint_state_data.header.stamp.sec) +
                static_cast<double>(joint_state_data.header.stamp.nanosec) * 1e-9;
            joint.position = joint_state_data.position[i];
            if (!joint_state_data.velocity.empty()) {
                joint.velocity = joint_state_data.velocity[i];
            }
            joint_measurements[joint_state_data.name[i]] = std::move(joint);
        }

        // Create the base imu measurement
        const auto& imu_data = base_imu_data_.value();
        serow::ImuMeasurement imu_measurement{};
        imu_measurement.timestamp = static_cast<double>(imu_data.header.stamp.sec) +
            static_cast<double>(imu_data.header.stamp.nanosec) * 1e-9;
        imu_measurement.linear_acceleration =
            Eigen::Vector3d(imu_data.linear_acceleration.x, imu_data.linear_acceleration.y,
                            imu_data.linear_acceleration.z);
        imu_measurement.angular_velocity = Eigen::Vector3d(
            imu_data.angular_velocity.x, imu_data.angular_velocity.y, imu_data.angular_velocity.z);
        // Clear the joint and imu measurements
        joint_state_data_.reset();
        base_imu_data_.reset();

        // Create the force torque measurements
        std::map<std::string, serow::ForceTorqueMeasurement> ft_measurements;
        if (ft_data_.size() == force_torque_state_subscriptions_.size()) {
            // Create the leg F/T measurement
            for (auto& [key, value] : ft_data_) {
                serow::ForceTorqueMeasurement ft{};
                const auto& ft_data = value;
                ft.timestamp = static_cast<double>(ft_data.header.stamp.sec) +
                    static_cast<double>(ft_data.header.stamp.nanosec) * 1e-9;
                ft.force = Eigen::Vector3d(ft_data.wrench.force.x, ft_data.wrench.force.y,
                                           ft_data.wrench.force.z);
                ft.torque.emplace(Eigen::Vector3d(ft_data.wrench.torque.x, ft_data.wrench.torque.y,
                                                  ft_data.wrench.torque.z));
                ft_measurements[key] = std::move(ft);
            }
            ft_data_.clear();
        }

        serow_.filter(imu_measurement, joint_measurements,
                      ft_measurements.size() == force_torque_state_subscriptions_.size()
                          ? std::make_optional(ft_measurements)
                          : std::nullopt);

        const auto& state = serow_.getState();
        if (state) {
            // Queue the state for publishing instead of publishing directly
            {
                std::lock_guard<std::mutex> lock(publish_queue_mutex_);
                publish_queue_.push(state.value());

                // Keep only the latest state to avoid queue buildup
                while (publish_queue_.size() > 1) {
                    publish_queue_.pop();
                }
            }
            publish_condition_.notify_one();
        }
    }
}

void SerowRos2::publish() {
    while (true) {
        std::unique_lock<std::mutex> lock(publish_queue_mutex_);
        publish_condition_.wait(lock,
                                [this] { return !publish_queue_.empty() || shutdown_requested_; });

        if (shutdown_requested_) {
            break;
        }

        // Process all queued states
        while (!publish_queue_.empty()) {
            auto state = publish_queue_.front();
            publish_queue_.pop();
            lock.unlock();

            // Publish all state data
            publishJointState(state);
            publishBaseState(state);
            publishCentroidState(state);
            publishContactState(state);

            lock.lock();
        }
    }
}

void SerowRos2::publishJointState(const serow::State& state) {
    auto joint_state_msg = sensor_msgs::msg::JointState();
    const auto& joint_positions = state.getJointPositions();
    const auto& joint_velocities = state.getJointVelocities();

    // Get the timestamp from the state
    const double timestamp = state.getTimestamp("joint");
    // Get seconds and nanoseconds from the timestamp
    const auto seconds = static_cast<int32_t>(timestamp);
    const auto nanoseconds = static_cast<int32_t>((timestamp - seconds) * 1e9);
    joint_state_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
    joint_state_msg.header.frame_id = "";

    // resize the joint state message to the number of joints
    joint_state_msg.name.resize(joint_positions.size());
    joint_state_msg.position.resize(joint_positions.size());
    joint_state_msg.velocity.resize(joint_velocities.size());

    // fill the joint state message
    size_t i = 0;
    for (const auto& [name, position] : joint_positions) {
        joint_state_msg.name[i] = name;
        joint_state_msg.position[i] = position;
        joint_state_msg.velocity[i] = joint_velocities.at(name);
        i++;
    }
    joint_state_publisher_->publish(joint_state_msg);
}

void SerowRos2::publishBaseState(const serow::State& state) {
    auto odom_msg = nav_msgs::msg::Odometry();

    // Get the timestamp from the state
    const double timestamp = state.getTimestamp("base");
    // Get seconds and nanoseconds from the timestamp
    const auto seconds = static_cast<int32_t>(timestamp);
    const auto nanoseconds = static_cast<int32_t>((timestamp - seconds) * 1e9);
    const Eigen::Vector3d& base_position = state.getBasePosition();
    const Eigen::Quaterniond& base_orientation = state.getBaseOrientation();
    const Eigen::Vector3d& base_linear_velocity = state.getBaseLinearVelocity();
    const Eigen::Vector3d& base_angular_velocity = state.getBaseAngularVelocity();
    const Eigen::Matrix<double, 6, 6> base_pose_cov = state.getBasePoseCov();
    const Eigen::Matrix<double, 6, 6> base_velocity_cov = state.getBaseVelocityCov();

    odom_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
    odom_msg.header.frame_id = "world";
    odom_msg.child_frame_id = state.getBaseFrame();
    odom_msg.pose.pose.position.x = base_position.x();
    odom_msg.pose.pose.position.y = base_position.y();
    odom_msg.pose.pose.position.z = base_position.z();
    odom_msg.pose.pose.orientation.x = base_orientation.x();
    odom_msg.pose.pose.orientation.y = base_orientation.y();
    odom_msg.pose.pose.orientation.z = base_orientation.z();
    odom_msg.pose.pose.orientation.w = base_orientation.w();
    odom_msg.twist.twist.linear.x = base_linear_velocity.x();
    odom_msg.twist.twist.linear.y = base_linear_velocity.y();
    odom_msg.twist.twist.linear.z = base_linear_velocity.z();
    odom_msg.twist.twist.angular.x = base_angular_velocity.x();
    odom_msg.twist.twist.angular.y = base_angular_velocity.y();
    odom_msg.twist.twist.angular.z = base_angular_velocity.z();

    for (size_t i = 0; i < 6; i++) {
        for (size_t j = 0; j < 6; j++) {
            odom_msg.pose.covariance[i * 6 + j] = base_pose_cov(i, j);
            odom_msg.twist.covariance[i * 6 + j] = base_velocity_cov(i, j);
        }
    }
    odom_publisher_->publish(odom_msg);

    const auto contact_frames = state.getContactsFrame();
    for (const auto& contact_frame : contact_frames) {
        const Eigen::Vector3d& foot_position = state.getFootPosition(contact_frame);
        const Eigen::Quaterniond& foot_orientation = state.getFootOrientation(contact_frame);
        const Eigen::Vector3d& foot_linear_velocity = state.getFootLinearVelocity(contact_frame);
        const Eigen::Vector3d& foot_angular_velocity = state.getFootAngularVelocity(contact_frame);

        odom_msg.header.stamp = rclcpp::Time(timestamp);
        odom_msg.header.frame_id = "world";
        odom_msg.pose.pose.position.x = foot_position.x();
        odom_msg.pose.pose.position.y = foot_position.y();
        odom_msg.pose.pose.position.z = foot_position.z();
        odom_msg.pose.pose.orientation.x = foot_orientation.x();
        odom_msg.pose.pose.orientation.y = foot_orientation.y();
        odom_msg.pose.pose.orientation.z = foot_orientation.z();
        odom_msg.pose.pose.orientation.w = foot_orientation.w();

        odom_msg.twist.twist.linear.x = foot_linear_velocity.x();
        odom_msg.twist.twist.linear.y = foot_linear_velocity.y();
        odom_msg.twist.twist.linear.z = foot_linear_velocity.z();
        odom_msg.twist.twist.angular.x = foot_angular_velocity.x();
        odom_msg.twist.twist.angular.y = foot_angular_velocity.y();
        odom_msg.twist.twist.angular.z = foot_angular_velocity.z();
        foot_odom_publishers_[contact_frame]->publish(odom_msg);
    }
}

void SerowRos2::publishCentroidState(const serow::State& state) {
    // Get the timestamp from the state
    const double timestamp = state.getTimestamp("centroidal");
    // Get seconds and nanoseconds from the timestamp
    const auto seconds = static_cast<int32_t>(timestamp);
    const auto nanoseconds = static_cast<int32_t>((timestamp - seconds) * 1e9);

    const double mass = state.getMass();
    auto point_msg = geometry_msgs::msg::PointStamped();
    auto twist_msg = geometry_msgs::msg::TwistStamped();
    auto wrench_msg = geometry_msgs::msg::WrenchStamped();

    const Eigen::Vector3d& com_position = state.getCoMPosition();
    const Eigen::Vector3d& com_linear_velocity = state.getCoMLinearVelocity();
    const Eigen::Vector3d& com_linear_acceleration = state.getCoMLinearAcceleration();
    const Eigen::Vector3d& com_external_forces = state.getCoMExternalForces();
    const Eigen::Vector3d& com_angular_momentum = state.getCoMAngularMomentum();
    const Eigen::Vector3d& com_angular_momentum_rate = state.getCoMAngularMomentumRate();
    const Eigen::Vector3d& cop_position = state.getCOPPosition();

    point_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
    point_msg.header.frame_id = "world";
    point_msg.point.x = com_position.x();
    point_msg.point.y = com_position.y();
    point_msg.point.z = com_position.z();
    com_position_publisher_->publish(point_msg);

    point_msg.point.x = cop_position.x();
    point_msg.point.y = cop_position.y();
    point_msg.point.z = cop_position.z();
    cop_position_publisher_->publish(point_msg);

    twist_msg.header.stamp = rclcpp::Time(timestamp);
    twist_msg.header.frame_id = "world";
    twist_msg.twist.linear.x = mass * com_linear_velocity.x();
    twist_msg.twist.linear.y = mass * com_linear_velocity.y();
    twist_msg.twist.linear.z = mass * com_linear_velocity.z();
    twist_msg.twist.angular.x = com_angular_momentum.x();
    twist_msg.twist.angular.y = com_angular_momentum.y();
    twist_msg.twist.angular.z = com_angular_momentum.z();
    com_momentum_publisher_->publish(twist_msg);

    twist_msg.twist.linear.x = mass * com_linear_acceleration.x();
    twist_msg.twist.linear.y = mass * com_linear_acceleration.y();
    twist_msg.twist.linear.z = mass * com_linear_acceleration.z();
    twist_msg.twist.angular.x = com_angular_momentum_rate.x();
    twist_msg.twist.angular.y = com_angular_momentum_rate.y();
    twist_msg.twist.angular.z = com_angular_momentum_rate.z();
    com_momentum_rate_publisher_->publish(twist_msg);

    const Eigen::Vector3d com_external_torque = com_position.cross(com_external_forces);
    wrench_msg.header.stamp = rclcpp::Time(timestamp);
    wrench_msg.header.frame_id = "world";
    wrench_msg.wrench.force.x = com_external_forces.x();
    wrench_msg.wrench.force.y = com_external_forces.y();
    wrench_msg.wrench.force.z = com_external_forces.z();
    wrench_msg.wrench.torque.x = com_external_torque.x();
    wrench_msg.wrench.torque.y = com_external_torque.y();
    wrench_msg.wrench.torque.z = com_external_torque.z();
    com_external_wrench_publisher_->publish(wrench_msg);
}

void SerowRos2::publishContactState(const serow::State& state) {
    // Get the timestamp from the state
    const double timestamp = state.getTimestamp("contact");
    // Get seconds and nanoseconds from the timestamp
    const auto seconds = static_cast<int32_t>(timestamp);
    const auto nanoseconds = static_cast<int32_t>((timestamp - seconds) * 1e9);

    for (const auto& contact_frame : state.getContactsFrame()) {
        auto float_msg = std_msgs::msg::Float64();
        auto wrench_msg = geometry_msgs::msg::WrenchStamped();
        const auto& contact_probability = state.getContactProbability(contact_frame);
        const auto& contact_force = state.getContactForce(contact_frame);
        wrench_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
        wrench_msg.header.frame_id = "world";
        if (contact_force) {
            wrench_msg.wrench.force.x = contact_force->x();
            wrench_msg.wrench.force.y = contact_force->y();
            wrench_msg.wrench.force.z = contact_force->z();
            const auto& contact_torque = state.getContactTorque(contact_frame);
            if (contact_torque) {
                wrench_msg.wrench.torque.x = contact_torque->x();
                wrench_msg.wrench.torque.y = contact_torque->y();
                wrench_msg.wrench.torque.z = contact_torque->z();
            }
        }
        foot_wrench_publishers_[contact_frame]->publish(wrench_msg);
        if (contact_probability) {
            float_msg.data = contact_probability.value();
        }
        foot_contact_probability_publishers_[contact_frame]->publish(float_msg);
    }
}

void SerowRos2::jointStateCallback(const sensor_msgs::msg::JointState& msg) {
    this->joint_state_data_ = msg;
}

void SerowRos2::baseImuCallback(const sensor_msgs::msg::Imu& msg) {
    this->base_imu_data_ = msg;
}
