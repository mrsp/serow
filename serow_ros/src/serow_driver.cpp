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
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>

#include <functional>
#include <map>
#include <serow/Serow.hpp>

#include "pinocchio/fwd.hpp"  // Always first include to avoid boost related compilation errors

class SerowDriver {
public:
    SerowDriver(const ros::NodeHandle& nh) : nh_(nh) {
        std::string joint_state_topic, base_imu_topic, config_file_path;
        std::vector<std::string> force_torque_state_topics;

        // Load parameters from the parameter server
        if (!nh_.getParam("joint_state_topic", joint_state_topic)) {
            ROS_ERROR("Failed to get param 'joint_state_topic'");
        }
        if (!nh_.getParam("base_imu_topic", base_imu_topic)) {
            ROS_ERROR("Failed to get param 'base_imu_topic'");
        }
        if (!nh_.getParam("force_torque_state_topics", force_torque_state_topics)) {
            ROS_ERROR("Failed to get param 'force_torque_state_topics'");
        }
        if (!nh_.getParam("config_file_path", config_file_path)) {
            ROS_ERROR("Failed to get param 'config_file_path'");
        }
        if (!nh_.getParam("joint_state_rate", loop_rate_)) {
            ROS_ERROR("Failed to get param 'joint_state_rate'");
        }
        num_feet_ = static_cast<int>(force_torque_state_topics.size());
        // Initialize SEROW
        if (!serow_.initialize(config_file_path)) {
            ROS_ERROR("SEROW cannot be initialized, exiting...");
            return;
        }

        odom_.header.frame_id = "world";
        odom_.child_frame_id = "base_link";
        com_.header.frame_id = "world";
        com_.child_frame_id = "com";
        cop_.header.frame_id = "world";
        cop_.child_frame_id = "cop";

        // Create subscribers
        joint_state_subscription_ =
            nh_.subscribe(joint_state_topic, 1, &SerowDriver::joint_state_topic_callback, this);
        base_imu_subscription_ =
            nh_.subscribe(base_imu_topic, 1, &SerowDriver::base_imu_topic_callback, this);
        odom_publisher_ = nh_.advertise<nav_msgs::Odometry>("/serow/base/odom", 1);
        com_publisher_ = nh_.advertise<nav_msgs::Odometry>("/serow/com/odom", 1);
        momentum_publisher_ = nh_.advertise<geometry_msgs::TwistStamped>("/serow/com/momentum", 1);
        momentum_rate_publisher_ =
            nh_.advertise<geometry_msgs::TwistStamped>("/serow/com/momentum_rate", 1);
        cop_publisher_ = nh_.advertise<nav_msgs::Odometry>("/serow/cop/odom", 1);

        // Dynamically create a wrench callback one for each limb
        for (const auto& ft_topic : force_torque_state_topics) {
            auto ft_callback =
                boost::bind(&SerowDriver::force_torque_state_topic_callback, this, _1);
            force_torque_state_subscriptions_.push_back(
                nh_.subscribe<geometry_msgs::WrenchStamped>(ft_topic, 1, ft_callback));
        }

        auto state = serow_.getState(true);
        for (const auto& frame : state->getContactsFrame()) {
            nav_msgs::Odometry odom_msg;
            geometry_msgs::WrenchStamped force_msg;

            odom_msg.header.frame_id = "world";
            odom_msg.child_frame_id = frame;
            feet_.push_back(std::move(odom_msg));

            force_msg.header.frame_id = "odom";
            contact_forces_.push_back(std::move(force_msg));

            std::string odom_topic = "/serow/" + frame + "/odom";
            std::string force_topic = "/serow/" + frame + "/contact_force";

            std::transform(odom_topic.begin(), odom_topic.end(), odom_topic.begin(), ::tolower);
            std::transform(force_topic.begin(), force_topic.end(), force_topic.begin(), ::tolower);

            ros::Publisher odom_pub = nh_.advertise<nav_msgs::Odometry>(odom_topic, 1);
            ros::Publisher force_pub = nh_.advertise<geometry_msgs::WrenchStamped>(force_topic, 1);

            feet_publisher_.push_back(std::move(odom_pub));
            contact_forces_publisher_.push_back(std::move(force_pub));
        }

        ROS_INFO("SEROW was initialized successfully");

        // Run SEROW
        run();
    }

    void joint_state_topic_callback(const sensor_msgs::JointState::ConstPtr& msg) {
        joint_state_data_ = *msg;
    }

    void base_imu_topic_callback(const sensor_msgs::Imu::ConstPtr& msg) {
        base_imu_data_ = *msg;
    }

    void force_torque_state_topic_callback(const geometry_msgs::WrenchStamped::ConstPtr& msg) {
        ft_data_[msg->header.frame_id] = *msg;
    }

private:
    void run() {
        ros::Rate loop_rate(loop_rate_);
        while (ros::ok()) {
            if (joint_state_data_.has_value() &&
                base_imu_data_.has_value()) {  // New messages arrived
                const sensor_msgs::JointState& joint_state_data = joint_state_data_.value();
                const sensor_msgs::Imu& base_imu_data = base_imu_data_.value();

                const auto& timestamp = base_imu_data.header.stamp;
                // Create the joint measurements
                std::map<std::string, serow::JointMeasurement> joint_measurements;
                for (size_t i = 0; i < joint_state_data.name.size(); i++) {
                    serow::JointMeasurement joint{};
                    joint.timestamp = static_cast<double>(joint_state_data.header.stamp.sec) +
                        static_cast<double>(joint_state_data.header.stamp.nsec) * 1e-9;
                    joint.position = joint_state_data.position[i];
                    if (joint_state_data.position.size() == joint_state_data.velocity.size()) {
                        joint.velocity = joint_state_data.velocity[i];
                    }
                    joint_measurements[joint_state_data.name[i]] = std::move(joint);
                }

                // Create the base imu measurement
                serow::ImuMeasurement imu_measurement{};
                imu_measurement.timestamp =
                    static_cast<double>(timestamp.sec) + static_cast<double>(timestamp.nsec) * 1e-9;
                imu_measurement.linear_acceleration = Eigen::Vector3d(
                    base_imu_data.linear_acceleration.x, base_imu_data.linear_acceleration.y,
                    base_imu_data.linear_acceleration.z);
                imu_measurement.angular_velocity = Eigen::Vector3d(
                    base_imu_data.angular_velocity.x, base_imu_data.angular_velocity.y,
                    base_imu_data.angular_velocity.z);

                // Create the leg F/T measurement
                std::map<std::string, serow::ForceTorqueMeasurement> ft_measurements;
                if (ft_data_.size() == num_feet_) {
                    for (auto& [key, value] : ft_data_) {
                        serow::ForceTorqueMeasurement ft{};
                        ft.timestamp = static_cast<double>(value.header.stamp.sec) +
                            static_cast<double>(value.header.stamp.nsec) * 1e-9;
                        ft.force = Eigen::Vector3d(value.wrench.force.x, value.wrench.force.y,
                                                   value.wrench.force.z);
                        ft.torque = Eigen::Vector3d(value.wrench.torque.x, value.wrench.torque.y,
                                                    value.wrench.torque.z);
                        ft_measurements[key] = std::move(ft);
                    }
                }
                serow_.filter(imu_measurement, joint_measurements,
                              ft_measurements.size() == num_feet_
                                  ? std::make_optional(ft_measurements)
                                  : std::nullopt);

                auto state = serow_.getState();
                if (state) {
                    odom_.header.seq += 1;
                    odom_.header.stamp = timestamp;
                    // Base 3D position
                    odom_.pose.pose.position.x = state->getBasePosition().x();
                    odom_.pose.pose.position.y = state->getBasePosition().y();
                    odom_.pose.pose.position.z = state->getBasePosition().z();
                    // Base 3D orientation
                    odom_.pose.pose.orientation.x = state->getBaseOrientation().coeffs().x();
                    odom_.pose.pose.orientation.y = state->getBaseOrientation().coeffs().y();
                    odom_.pose.pose.orientation.z = state->getBaseOrientation().coeffs().z();
                    odom_.pose.pose.orientation.w = state->getBaseOrientation().coeffs().w();
                    // Base 3D linear velocity
                    odom_.twist.twist.linear.x = state->getBaseLinearVelocity().x();
                    odom_.twist.twist.linear.y = state->getBaseLinearVelocity().y();
                    odom_.twist.twist.linear.z = state->getBaseLinearVelocity().z();
                    // Base 3D angular velocity
                    odom_.twist.twist.angular.x = state->getBaseAngularVelocity().x();
                    odom_.twist.twist.angular.y = state->getBaseAngularVelocity().y();
                    odom_.twist.twist.angular.z = state->getBaseAngularVelocity().z();
                    // Base 3D pose and velocity covariance
                    const Eigen::Matrix<double, 6, 6>& base_pose_cov = state->getBasePoseCov();
                    const Eigen::Matrix<double, 6, 6>& base_velocity_cov =
                        state->getBaseVelocityCov();
                    for (size_t i = 0; i < 6; i++) {
                        for (size_t j = 0; j < 6; j++) {
                            // Row-major
                            odom_.pose.covariance[i * 6 + j] = base_pose_cov(i, j);
                            odom_.twist.covariance[i * 6 + j] = base_velocity_cov(i, j);
                        }
                    }
                    odom_publisher_.publish(odom_);

                    com_.header.seq += 1;
                    com_.header.stamp = timestamp;
                    // CoM 3D position
                    com_.pose.pose.position.x = state->getCoMPosition().x();
                    com_.pose.pose.position.y = state->getCoMPosition().y();
                    com_.pose.pose.position.z = state->getCoMPosition().z();
                    // CoM 3D linear velocity
                    com_.twist.twist.linear.x = state->getCoMLinearVelocity().x();
                    com_.twist.twist.linear.y = state->getCoMLinearVelocity().y();
                    com_.twist.twist.linear.z = state->getCoMLinearVelocity().z();
                    // CoM 3D position and linear velocity covariance
                    const Eigen::Matrix<double, 3, 3>& com_position_cov =
                        state->getCoMPositionCov();
                    const Eigen::Matrix<double, 3, 3>& com_linear_velocity_cov =
                        state->getCoMLinearVelocityCov();
                    for (size_t i = 0; i < 3; i++) {
                        for (size_t j = 0; j < 3; j++) {
                            // Row-major
                            com_.pose.covariance[i * 6 + j] = com_position_cov(i, j);
                            com_.twist.covariance[i * 6 + j] = com_linear_velocity_cov(i, j);
                        }
                    }
                    com_publisher_.publish(com_);

                    // 3D CoM linear and angular momentum
                    momentum_.header.seq += 1;
                    momentum_.header.stamp = timestamp;
                    momentum_.twist.linear.x = state->getMass() * state->getCoMLinearVelocity().x();
                    momentum_.twist.linear.y = state->getMass() * state->getCoMLinearVelocity().y();
                    momentum_.twist.linear.z = state->getMass() * state->getCoMLinearVelocity().z();
                    momentum_.twist.angular.x = state->getCoMAngularMomentum().x();
                    momentum_.twist.angular.y = state->getCoMAngularMomentum().y();
                    momentum_.twist.angular.z = state->getCoMAngularMomentum().z();
                    momentum_publisher_.publish(momentum_);

                    // 3D CoM linear and angular momentum rate
                    momentum_rate_.header.seq += 1;
                    momentum_rate_.header.stamp = timestamp;
                    momentum_rate_.twist.linear.x =
                        state->getMass() * state->getCoMLinearAcceleration().x();
                    momentum_rate_.twist.linear.y =
                        state->getMass() * state->getCoMLinearAcceleration().y();
                    momentum_rate_.twist.linear.z =
                        state->getMass() * state->getCoMLinearAcceleration().z();
                    momentum_rate_.twist.angular.x = state->getCoMAngularMomentumRate().x();
                    momentum_rate_.twist.angular.y = state->getCoMAngularMomentumRate().x();
                    momentum_rate_.twist.angular.z = state->getCoMAngularMomentumRate().z();
                    momentum_rate_publisher_.publish(momentum_rate_);

                    size_t i = 0;
                    for (auto& foot : feet_) {
                        foot.header.seq += 1;
                        foot.header.stamp = timestamp;
                        // Foot 3D position
                        foot.pose.pose.position.x = state->getFootPosition(foot.child_frame_id).x();
                        foot.pose.pose.position.y = state->getFootPosition(foot.child_frame_id).y();
                        foot.pose.pose.position.z = state->getFootPosition(foot.child_frame_id).z();
                        // Foot 3D orientation
                        foot.pose.pose.orientation.x =
                            state->getFootOrientation(foot.child_frame_id).coeffs().x();
                        foot.pose.pose.orientation.y =
                            state->getFootOrientation(foot.child_frame_id).coeffs().y();
                        foot.pose.pose.orientation.z =
                            state->getFootOrientation(foot.child_frame_id).coeffs().z();
                        foot.pose.pose.orientation.w =
                            state->getFootOrientation(foot.child_frame_id).coeffs().w();
                        // Foot 3D linear velocity
                        foot.twist.twist.linear.x =
                            state->getFootLinearVelocity(foot.child_frame_id).x();
                        foot.twist.twist.linear.y =
                            state->getFootLinearVelocity(foot.child_frame_id).y();
                        foot.twist.twist.linear.z =
                            state->getFootLinearVelocity(foot.child_frame_id).z();
                        // Foot 3D angular velocity
                        foot.twist.twist.angular.x =
                            state->getFootAngularVelocity(foot.child_frame_id).x();
                        foot.twist.twist.angular.y =
                            state->getFootAngularVelocity(foot.child_frame_id).y();
                        foot.twist.twist.angular.z =
                            state->getFootAngularVelocity(foot.child_frame_id).z();
                        feet_publisher_[i].publish(foot);

                        auto force_vec = state.value().getContactForce(foot.child_frame_id);
                        contact_forces_[i].header.seq += 1;
                        contact_forces_[i].header.frame_id = foot.child_frame_id;
                        contact_forces_[i].header.stamp = timestamp;
                        contact_forces_[i].wrench.force.x = force_vec.value().x();
                        contact_forces_[i].wrench.force.y = force_vec.value().y();
                        contact_forces_[i].wrench.force.z = force_vec.value().z();

                        contact_forces_publisher_[i].publish(contact_forces_[i]);
                        i++;
                    }

                    cop_.header.seq += 1;
                    cop_.header.stamp = timestamp;
                    // COP 3D position
                    cop_.pose.pose.position.x = state->getCOPPosition().x();
                    cop_.pose.pose.position.y = state->getCOPPosition().y();
                    cop_.pose.pose.position.z = state->getCOPPosition().z();
                    cop_publisher_.publish(cop_);
                }
                joint_state_data_.reset();
                base_imu_data_.reset();
            }
            ros::spinOnce();
            loop_rate.sleep();
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber joint_state_subscription_;
    ros::Subscriber base_imu_subscription_;
    ros::Publisher odom_publisher_;
    ros::Publisher com_publisher_;
    ros::Publisher cop_publisher_;
    ros::Publisher momentum_publisher_;
    ros::Publisher momentum_rate_publisher_;
    std::vector<ros::Publisher> feet_publisher_;
    std::vector<ros::Publisher> contact_forces_publisher_;
    std::vector<ros::Subscriber> force_torque_state_subscriptions_;
    std::optional<sensor_msgs::JointState> joint_state_data_;
    std::optional<sensor_msgs::Imu> base_imu_data_;
    std::map<std::string, geometry_msgs::WrenchStamped> ft_data_;
    double loop_rate_{};
    size_t num_feet_{};
    nav_msgs::Odometry odom_;
    nav_msgs::Odometry com_;
    nav_msgs::Odometry cop_;
    std::vector<nav_msgs::Odometry> feet_;
    std::vector<geometry_msgs::WrenchStamped> contact_forces_;
    geometry_msgs::TwistStamped momentum_;
    geometry_msgs::TwistStamped momentum_rate_;

    serow::Serow serow_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "serow_ros");
    ros::NodeHandle nh;

    SerowDriver driver(nh);

    return 0;
}
