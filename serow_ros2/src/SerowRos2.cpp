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
#include <limits>
#include <mutex>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "yaml-cpp/yaml.h"

using std::placeholders::_1;
using std::placeholders::_2;
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
        throw std::runtime_error("Failed to load YAML file: " + std::string(e.what()));
    } catch (const YAML::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Error parsing YAML file: %s", e.what());
        throw std::runtime_error("Error parsing YAML file: " + std::string(e.what()));
    }

    // Extract configuration values
    const std::string& robot_name = config["robot_name"].as<std::string>();
    const std::string& joint_state_topic = config["topics"]["joint_states"].as<std::string>();
    const std::string& base_imu_topic = config["topics"]["imu"].as<std::string>();
    for (const auto& topic : config["topics"]["force_torque_states"]) {
        force_torque_state_topics_.push_back(topic.as<std::string>());
    }
    const std::string& ground_truth_topic = config["topics"]["ground_truth"].as<std::string>();
    if (!ground_truth_topic.empty()) {
        ground_truth_subscriber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            ground_truth_topic, 10, std::bind(&SerowRos2::groundTruthCallback, this, _1));
    }
    const std::string& serow_config = config["serow_config"].as<std::string>();
    const bool publish_path = config["publish_path"].as<bool>();
    add_gravity_to_imu_ = config["add_gravity_to_imu"].as<bool>();

    RCLCPP_INFO(this->get_logger(), "Robot name: %s", robot_name.c_str());
    RCLCPP_INFO(this->get_logger(), "Serow config file: %s", serow_config.c_str());

    // Configure F/T synchronization tolerance
    ft_max_time_diff_ = this->declare_parameter<double>("ft_max_time_diff", 0.1);
    RCLCPP_INFO(this->get_logger(), "F/T max time difference: %.3f seconds", ft_max_time_diff_);

    // Initialize SERoW
    try {
        serow_.initialize(serow_config);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize SERoW: %s", e.what());
        throw std::runtime_error("Failed to initialize SERoW: " + std::string(e.what()));
    }

    // Create subscribers
    RCLCPP_INFO(this->get_logger(), "Creating joint state and base imu synchronizer subscription on topics: %s and %s",
                joint_state_topic.c_str(), base_imu_topic.c_str());

    // Create message_filters subscribers for synchronization
    joint_state_subscriber_ = this->create_subscription<sensor_msgs::msg::JointState>(joint_state_topic, 10, std::bind(&SerowRos2::jointStateCallback, this, _1));
    base_imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(base_imu_topic, 2, std::bind(&SerowRos2::baseImuCallback, this, _1));

    // Dynamically create a wrench callback, one for each force torque state topic
    for (size_t i = 0; i < force_torque_state_topics_.size(); ++i) {
        const auto& ft_topic = force_torque_state_topics_[i];
        
        // Create a dedicated mutex for this subscription
        ft_subscription_mutexes_.push_back(std::make_unique<std::mutex>());
        std::mutex* ft_mutex = ft_subscription_mutexes_.back().get();
        
        auto force_torque_state_topic_callback =
            [this, ft_mutex, ft_topic](const geometry_msgs::msg::WrenchStamped& msg) {
                std::lock_guard<std::mutex> lock(*ft_mutex);
                serow::ForceTorqueMeasurement ft{};
                ft.timestamp = static_cast<double>(msg.header.stamp.sec) +
                    static_cast<double>(msg.header.stamp.nanosec) * 1e-9;
                ft.force = Eigen::Vector3d(msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z);
                ft.torque.emplace(Eigen::Vector3d(msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z));
                this->ft_buffers_[msg.header.frame_id].add(ft);
                // Save frame_id to ft_topic mapping
                ft_topic_to_frame_id_[ft_topic] = msg.header.frame_id;
            };
        force_torque_state_topic_callbacks_.push_back(std::move(force_torque_state_topic_callback));
        RCLCPP_INFO(this->get_logger(), "Creating force torque state subscription on topic: %s",
                    ft_topic.c_str());
        auto ft_subscription = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
            ft_topic, 2, force_torque_state_topic_callbacks_.back());
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
        if (publish_path) {
            foot_odom_path_publishers_[contact_frame] = this->create_publisher<nav_msgs::msg::Path>("/serow/" + contact_frame + "/odom/path", 10);
        }
    }
    odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
        "/serow/odom", 10);
    joint_state_publisher_ =
        this->create_publisher<sensor_msgs::msg::JointState>("/serow/joint_states", 10);
    if  (!ground_truth_topic.empty()) {
        ground_truth_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>(
            "/serow/ground_truth", 10);
        if (publish_path) {
            ground_truth_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/serow/ground_truth/path", 10);
        }
    }
    if (publish_path) {
        odom_path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/serow/odom/path", 10);
    }

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
    RCLCPP_INFO(this->get_logger(), "SERoW was shut down successfully");
}

void SerowRos2::run() {
    // Fetch the most recent joint state data 
    sensor_msgs::msg::JointState joint_state_data;
    {
        std::lock_guard<std::mutex> lock(joint_data_mutex_);
        if (!joint_state_data_.has_value()) {
            return;
        }
        joint_state_data = joint_state_data_.value();

        // Clear the used joint measurements
        joint_state_data_.reset();
    }

    double joint_state_timestamp = static_cast<double>(joint_state_data.header.stamp.sec) +
        static_cast<double>(joint_state_data.header.stamp.nanosec) * 1e-9;
    
    // Fetch the base imu measurement from the buffer
    std::optional<serow::ImuMeasurement> base_imu_measurement = std::nullopt;
    {
        std::lock_guard<std::mutex> lock(base_imu_data_mutex_);
        base_imu_measurement = base_imu_buffer_.get(joint_state_timestamp, ft_max_time_diff_);
        if (!base_imu_measurement.has_value()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to get base imu measurement at timestamp: %f", joint_state_timestamp);
            const auto& time_range = base_imu_buffer_.getTimeRange();
            if (time_range) {
                RCLCPP_ERROR(this->get_logger(), "Time range: %f - %f", time_range->first, time_range->second);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Time range: not available");
            }
            return;
        } 
    }

    // Create the joint measurements
    std::map<std::string, serow::JointMeasurement> joint_measurements;
    for (size_t i = 0; i < joint_state_data.name.size(); i++) {
        serow::JointMeasurement joint{};
        joint.timestamp = joint_state_timestamp;
        joint.position = joint_state_data.position[i];
        if (!joint_state_data.velocity.empty()) {
            joint.velocity = joint_state_data.velocity[i];
        }
        joint_measurements[joint_state_data.name[i]] = std::move(joint);
    }
    
    // Retrieve the synchronized F/T measurements for each foot frame
    std::map<std::string, serow::ForceTorqueMeasurement> synchronized_ft_measurements;
    for (size_t i = 0; i < force_torque_state_topics_.size(); ++i) {
        const auto& ft_topic = force_torque_state_topics_[i];
        std::lock_guard<std::mutex> lock(*ft_subscription_mutexes_[i]);
        const auto& frame_id = ft_topic_to_frame_id_[ft_topic];
        const auto& ft_buffer = ft_buffers_[frame_id];
        const auto& ft_measurement = ft_buffer.get(joint_state_timestamp, ft_max_time_diff_);
        if (ft_measurement) {
            synchronized_ft_measurements[frame_id] = std::move(ft_measurement.value());
        } else {
            const auto& time_range = ft_buffer.getTimeRange();
            RCLCPP_ERROR(this->get_logger(), "Failed to get F/T measurement for frame: %s at timestamp: %f", frame_id.c_str(), joint_state_timestamp);
            if (time_range) {
                RCLCPP_ERROR(this->get_logger(), "Time range: %f - %f", time_range->first, time_range->second);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Time range: not available");
            }
        }
    }

    // Fetch the most recent ground truth data (if available)
    std::optional<serow::BasePoseGroundTruth> ground_truth_pose = std::nullopt;
    if (first_ground_truth_pose_.has_value()) {
        std::lock_guard<std::mutex> lock(ground_truth_data_mutex_); 
        const auto& ground_truth_odometry = ground_truth_odometry_buffer_.get(joint_state_timestamp, ft_max_time_diff_);
        if (ground_truth_odometry.has_value()) {
            serow::BasePoseGroundTruth gt;
            gt.timestamp = ground_truth_odometry.value().timestamp;
            gt.position = ground_truth_odometry.value().base_position;
            gt.orientation = ground_truth_odometry.value().base_orientation;
            ground_truth_pose = std::move(gt);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to get ground truth odometry at timestamp: %f", joint_state_timestamp);
            const auto& time_range = ground_truth_odometry_buffer_.getTimeRange();
            if (time_range) {
                RCLCPP_ERROR(this->get_logger(), "Time range: %f - %f", time_range->first, time_range->second);
            } else {
                RCLCPP_ERROR(this->get_logger(), "Time range: not available");
            }
        }
    }

    serow_.filter(base_imu_measurement.value(), joint_measurements, 
                  synchronized_ft_measurements.size() == force_torque_state_topics_.size() ? std::make_optional(synchronized_ft_measurements) : std::nullopt,
                  std::nullopt,
                  std::nullopt,
                  ground_truth_pose);
    const auto& state = serow_.getState();
    if (state) {
        // Queue the state for publishing instead of publishing directly to not block the main thread
        {
            std::lock_guard<std::mutex> lock(publish_queue_mutex_);
            publish_queue_.push(std::make_pair(state.value(), ground_truth_pose));

            // Keep only the latest state to avoid queue buildup
            while (publish_queue_.size() > 1) {
                publish_queue_.pop();
            }
        }
        publish_condition_.notify_one();
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
            auto [state, gt] = publish_queue_.front();
            publish_queue_.pop();
            lock.unlock();

            // Publish all state data with exception handling
            try {
                publishJointState(state);
                publishBaseState(state);
                publishCentroidState(state);
                publishContactState(state);
                if (gt.has_value()) {
                    publishGroundTruth(gt.value(), state.getBaseFrame());
                }
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Error publishing state: %s", e.what());
            }

            lock.lock();
        }
    }
}

void SerowRos2::publishGroundTruth(const serow::BasePoseGroundTruth& gt, const std::string& frame_id) {
    auto ground_truth_msg = nav_msgs::msg::Odometry();
    ground_truth_msg.header.stamp = rclcpp::Time(gt.timestamp);
    ground_truth_msg.header.frame_id = "world";
    ground_truth_msg.child_frame_id = frame_id;
    ground_truth_msg.pose.pose.position.x = gt.position.x();
    ground_truth_msg.pose.pose.position.y = gt.position.y();
    ground_truth_msg.pose.pose.position.z = gt.position.z();
    ground_truth_msg.pose.pose.orientation.x = gt.orientation.x();
    ground_truth_msg.pose.pose.orientation.y = gt.orientation.y();
    ground_truth_msg.pose.pose.orientation.z = gt.orientation.z();
    ground_truth_msg.pose.pose.orientation.w = gt.orientation.w();
    ground_truth_publisher_->publish(ground_truth_msg);

    if (ground_truth_path_publisher_) {
        gt_path_msg_.header.stamp = ground_truth_msg.header.stamp;
        gt_path_msg_.header.frame_id = "world";
        // Handle a max size of 1000 poses by removing the oldest pose when the size is reached
        if (gt_path_msg_.poses.size() >= 1000) {
            gt_path_msg_.poses.erase(gt_path_msg_.poses.begin());
        }
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = ground_truth_msg.header;
        pose_stamped.pose = ground_truth_msg.pose.pose;
        gt_path_msg_.poses.push_back(pose_stamped);
        ground_truth_path_publisher_->publish(gt_path_msg_);
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

    if (odom_path_publisher_) {
        odom_path_msg_.header.stamp = odom_msg.header.stamp;
        odom_path_msg_.header.frame_id = "world";
        // Handle a max size of 1000 poses by removing the oldest pose when the size is reached
        if (odom_path_msg_.poses.size() >= 1000) {
            odom_path_msg_.poses.erase(odom_path_msg_.poses.begin());
        }
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header = odom_msg.header;
        pose_stamped.pose = odom_msg.pose.pose;
        odom_path_msg_.poses.push_back(pose_stamped);
        odom_path_publisher_->publish(odom_path_msg_);
    }

    const auto& contact_frames = state.getContactsFrame();
    for (const auto& contact_frame : contact_frames) {
        const Eigen::Vector3d& foot_position = state.getFootPosition(contact_frame);
        const Eigen::Quaterniond& foot_orientation = state.getFootOrientation(contact_frame);
        const Eigen::Vector3d& foot_linear_velocity = state.getFootLinearVelocity(contact_frame);
        const Eigen::Vector3d& foot_angular_velocity = state.getFootAngularVelocity(contact_frame);

        odom_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
        odom_msg.header.frame_id = "world";
        odom_msg.child_frame_id = contact_frame;
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
        if (foot_odom_path_publishers_[contact_frame]) {
            foot_odom_path_msgs_[contact_frame].header.stamp = odom_msg.header.stamp;
            foot_odom_path_msgs_[contact_frame].header.frame_id = "world";
            // Handle a max size of 1000 poses by removing the oldest pose when the size is reached
            if (foot_odom_path_msgs_[contact_frame].poses.size() >= 1000) {
                foot_odom_path_msgs_[contact_frame].poses.erase(foot_odom_path_msgs_[contact_frame].poses.begin());
            }
            geometry_msgs::msg::PoseStamped pose_stamped;
            pose_stamped.header = odom_msg.header;
            pose_stamped.pose = odom_msg.pose.pose;
            foot_odom_path_msgs_[contact_frame].poses.push_back(pose_stamped);
            foot_odom_path_publishers_[contact_frame]->publish(foot_odom_path_msgs_[contact_frame]);
        }
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

    twist_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
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
    wrench_msg.header.stamp = rclcpp::Time(seconds, nanoseconds);
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

void SerowRos2::jointStateCallback(const sensor_msgs::msg::JointState::ConstSharedPtr& joint_state_msg) {
    std::lock_guard<std::mutex> lock(joint_data_mutex_);
    this->joint_state_data_ = *joint_state_msg;
}

void SerowRos2::baseImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr& base_imu_msg) {
    std::lock_guard<std::mutex> lock(base_imu_data_mutex_);
    serow::ImuMeasurement imu_measurement{};
    imu_measurement.timestamp = static_cast<double>(base_imu_msg->header.stamp.sec) +
        static_cast<double>(base_imu_msg->header.stamp.nanosec) * 1e-9;
    imu_measurement.linear_acceleration =
        Eigen::Vector3d(base_imu_msg->linear_acceleration.x, base_imu_msg->linear_acceleration.y,
                        base_imu_msg->linear_acceleration.z);
    imu_measurement.angular_velocity = Eigen::Vector3d(
        base_imu_msg->angular_velocity.x, base_imu_msg->angular_velocity.y, base_imu_msg->angular_velocity.z);
    
    if (add_gravity_to_imu_) {
        const Eigen::Matrix3d R_imu_to_world = Eigen::Quaterniond(base_imu_msg->orientation.w, base_imu_msg->orientation.x, base_imu_msg->orientation.y, base_imu_msg->orientation.z).toRotationMatrix();
        imu_measurement.linear_acceleration = R_imu_to_world.transpose() * (R_imu_to_world * imu_measurement.linear_acceleration + Eigen::Vector3d(0.0, 0.0, 9.81));
    }
    base_imu_buffer_.add(std::move(imu_measurement));
}


void SerowRos2::groundTruthCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& ground_truth_msg) {
    std::lock_guard<std::mutex> lock(ground_truth_data_mutex_);
    const Eigen::Quaterniond q(ground_truth_msg->pose.pose.orientation.w, ground_truth_msg->pose.pose.orientation.x, ground_truth_msg->pose.pose.orientation.y, ground_truth_msg->pose.pose.orientation.z);
    const Eigen::Vector3d p(ground_truth_msg->pose.pose.position.x, ground_truth_msg->pose.pose.position.y, ground_truth_msg->pose.pose.position.z);
    Eigen::Isometry3d current_ground_truth_pose = Eigen::Isometry3d::Identity();
    current_ground_truth_pose.linear() = q.toRotationMatrix();
    current_ground_truth_pose.translation() = p;

    if (!first_ground_truth_pose_.has_value()) {
        first_ground_truth_pose_ = current_ground_truth_pose;
    }

    // Express current pose in the first pose's frame: origin at first position,
    const Eigen::Isometry3d ground_truth_pose = first_ground_truth_pose_.value().inverse() * current_ground_truth_pose;

    serow::OdometryMeasurement ground_truth_odometry;
    ground_truth_odometry.timestamp = ground_truth_msg->header.stamp.sec + ground_truth_msg->header.stamp.nanosec * 1e-9;
    ground_truth_odometry.base_position = ground_truth_pose.translation();
    ground_truth_odometry.base_orientation = ground_truth_pose.linear();
    ground_truth_odometry_buffer_.add(std::move(ground_truth_odometry));
}
