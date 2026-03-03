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
#pragma once

#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <serow/ForceTorqueMeasurementBuffer.hpp>
#include <serow/ImuMeasurementBuffer.hpp>
#include <serow/OdometryMeasurementBuffer.hpp>
#include <serow/Serow.hpp>
#include <thread>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

class SerowRos2 : public rclcpp::Node {
public:
    SerowRos2();

    ~SerowRos2();

    void run();

private:
    void publish();

    void publishJointState(const serow::State& state);

    void publishBaseState(const serow::State& state);

    void publishCentroidState(const serow::State& state);

    void publishContactState(const serow::State& state);

    void publishGroundTruth(const serow::BasePoseGroundTruth& gt, const std::string& frame_id);

    void jointStateCallback(const sensor_msgs::msg::JointState::ConstSharedPtr& joint_state_msg);

    void baseImuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr& base_imu_msg);

    void groundTruthCallback(const nav_msgs::msg::Odometry::ConstSharedPtr& ground_truth_msg);

    void externalOdometryCallback(
        const nav_msgs::msg::Odometry::ConstSharedPtr& external_odometry_msg);

    serow::Serow serow_;
    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr odom_path_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr com_position_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr cop_position_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr com_momentum_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr com_momentum_rate_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr com_external_wrench_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr ground_truth_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr ground_truth_path_publisher_;
    std::map<std::string, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr>
        foot_odom_publishers_;
    std::map<std::string, rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr>
        foot_odom_path_publishers_;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr>
        foot_wrench_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr>
        foot_contact_probability_publishers_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr base_imu_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr external_odometry_subscriber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr ground_truth_subscriber_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr>
        force_torque_state_subscriptions_;
    std::vector<std::function<void(const geometry_msgs::msg::WrenchStamped&)>>
        force_torque_state_topic_callbacks_;

    std::vector<std::unique_ptr<std::mutex>>
        ft_subscription_mutexes_;  // One mutex per F/T subscription
    std::queue<sensor_msgs::msg::JointState> joint_state_queue_;
    std::map<std::string, serow::ForceTorqueMeasurementBuffer> ft_buffers_;
    serow::OdometryMeasurementBuffer ground_truth_odometry_buffer_;
    serow::OdometryMeasurementBuffer external_odometry_buffer_;
    serow::ImuMeasurementBuffer base_imu_buffer_;
    std::vector<std::string> force_torque_state_topics_;
    std::map<std::string, std::string> ft_topic_to_frame_id_;

    rclcpp::TimerBase::SharedPtr timer_;

    // Threading components for asynchronous publishing
    std::thread publishing_thread_;
    std::queue<std::pair<serow::State, std::optional<serow::BasePoseGroundTruth>>> publish_queue_;
    std::mutex publish_queue_mutex_;
    std::condition_variable publish_condition_;
    bool shutdown_requested_ = false;
    std::mutex joint_data_mutex_;
    std::mutex base_imu_data_mutex_;
    std::mutex ground_truth_data_mutex_;
    std::mutex external_odometry_data_mutex_;
    std::optional<Eigen::Isometry3d> first_ground_truth_pose_;
    nav_msgs::msg::Path odom_path_msg_;
    nav_msgs::msg::Path gt_path_msg_;
    std::map<std::string, nav_msgs::msg::Path> foot_odom_path_msgs_;
    bool add_gravity_to_imu_ = false;
    std::optional<Eigen::Vector3d> external_odometry_position_covariance_;
    std::optional<Eigen::Vector3d> external_odometry_orientation_covariance_;
    double ft_max_time_diff_{0.01};  // Max time difference for F/T synchronization (default: 10ms)
    double imu_max_time_diff_{0.005};  // Max time difference for IMU synchronization (default: 5ms)
    double gt_max_time_diff_{
        0.05};  // Max time difference for ground truth synchronization (default: 50ms)
    double external_odometry_max_time_diff_{
        0.1};  // Max time difference for external odometry synchronization (default: 100ms)
};
