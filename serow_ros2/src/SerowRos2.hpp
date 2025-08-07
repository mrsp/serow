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
#include <mutex>
#include <queue>
#include <serow/Serow.hpp>
#include <thread>

#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/wrench_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64.hpp>

class SerowRos2 : public rclcpp::Node {
public:
    SerowRos2();
    ~SerowRos2();

private:
    void run();

    void publish();

    void publishJointState(const serow::State& state);

    void publishBaseState(const serow::State& state);

    void publishCentroidState(const serow::State& state);

    void publishContactState(const serow::State& state);

    void jointStateCallback(const sensor_msgs::msg::JointState& msg);

    void baseImuCallback(const sensor_msgs::msg::Imu& msg);

    serow::Serow serow_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr com_position_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr cop_position_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr com_momentum_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr com_momentum_rate_publisher_;
    rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr com_external_wrench_publisher_;
    std::map<std::string, rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr>
        foot_odom_publishers_;
    std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::WrenchStamped>::SharedPtr>
        foot_wrench_publishers_;
    std::map<std::string, rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr>
        foot_contact_probability_publishers_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_publisher_;

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr base_imu_subscription_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr>
        force_torque_state_subscriptions_;
    std::vector<std::function<void(const geometry_msgs::msg::WrenchStamped&)>>
        force_torque_state_topic_callbacks_;
    std::optional<sensor_msgs::msg::Imu> base_imu_data_;
    std::optional<sensor_msgs::msg::JointState> joint_state_data_;
    std::map<std::string, geometry_msgs::msg::WrenchStamped> ft_data_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Threading components for asynchronous publishing
    std::thread publishing_thread_;
    std::queue<serow::State> publish_queue_;
    std::mutex publish_queue_mutex_;
    std::condition_variable publish_condition_;
    bool shutdown_requested_ = false;
};
