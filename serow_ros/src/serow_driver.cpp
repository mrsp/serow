#include "pinocchio/fwd.hpp" // Always first include to avoid boost related compilation errors
#include <functional>
#include <geometry_msgs/WrenchStamped.h>
#include <map>
#include <nav_msgs/Odometry.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <serow/Serow.hpp>


class SerowDriver {
public:
    SerowDriver(const ros::NodeHandle& nh)
        : nh_(nh) {
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

        // Initialize SERoW
        if (!serow_.initialize(config_file_path)) {
            ROS_ERROR("SEROW cannot be initialized, exiting...");
            return;
        }
        
        odom_.header.frame_id = "world";
        odom_.child_frame_id = "base_link";
    
        // Create subscribers
        joint_state_subscription_ =
            nh_.subscribe(joint_state_topic, 1000, &SerowDriver::joint_state_topic_callback, this);
        base_imu_subscription_ =
            nh_.subscribe(base_imu_topic, 1000, &SerowDriver::base_imu_topic_callback, this);
        odom_publisher_ =
            nh_.advertise<nav_msgs::Odometry>("/serow/base/odometry", 1000);

        // Dynamically create a wrench callback one for each limb
        for (const auto& ft_topic : force_torque_state_topics) {
            auto ft_callback =
                boost::bind(&SerowDriver::force_torque_state_topic_callback, this, _1);
            force_torque_state_subscriptions_.push_back(
                nh_.subscribe<geometry_msgs::WrenchStamped>(ft_topic, 1000, ft_callback));
        }
        ROS_INFO("SEROW was initialized successfully");

        // Run SERoW
        run();
    }

    void joint_state_topic_callback(const sensor_msgs::JointState::ConstPtr& msg) {
        joint_state_data_.push(*msg);
        if (joint_state_data_.size() > 100) {
            joint_state_data_.pop();
            ROS_WARN("SEROW is dropping joint state measurements, SEROW estimate is not real-time");
        }
    }

    void base_imu_topic_callback(const sensor_msgs::Imu::ConstPtr& msg) {
        base_imu_data_.push(*msg);
        if (base_imu_data_.size() > 100) {
            base_imu_data_.pop();
            ROS_WARN("SEROW is dropping base IMU measurements, SEROW estimate is not real-time");
        }
    }

    void force_torque_state_topic_callback(const geometry_msgs::WrenchStamped::ConstPtr& msg) {
        std::string frame_id = msg->header.frame_id;
        ft_data_[frame_id].push(*msg);
        
        if (ft_data_.at(frame_id).size() > 100) {
            ft_data_.at(frame_id).pop();
            ROS_WARN("SEROW is dropping leg F/T measurements, SEROW estimate is not real-time");
        }
    }

private:
    void run() {
        ros::Rate loop_rate(loop_rate_); // Define the loop rate
        while (ros::ok()) { // Check if ROS is still running
            if (joint_state_data_.size() > 0 && base_imu_data_.size() > 0) {
                // Create the joint measurements
                const auto& joint_state_data = joint_state_data_.front();
                std::map<std::string, serow::JointMeasurement> joint_measurements;
                for (size_t i = 0; i < joint_state_data.name.size(); i++) {
                    serow::JointMeasurement joint{};
                    joint.timestamp =
                        static_cast<double>(joint_state_data.header.stamp.sec) +
                        static_cast<double>(joint_state_data.header.stamp.nsec) * 1e-9;
                    joint.position = joint_state_data.position[i];
                    joint_measurements[joint_state_data.name[i]] = std::move(joint);
                }
                joint_state_data_.pop();

                // Create the base imu measurement
                const auto& imu_data = base_imu_data_.front();
                const auto base_timestamp = imu_data.header.stamp;
                serow::ImuMeasurement imu_measurement{};
                imu_measurement.timestamp =
                    static_cast<double>(imu_data.header.stamp.sec) +
                    static_cast<double>(imu_data.header.stamp.nsec) * 1e-9;
                imu_measurement.linear_acceleration =
                    Eigen::Vector3d(imu_data.linear_acceleration.x, imu_data.linear_acceleration.y,
                                    imu_data.linear_acceleration.z);
                imu_measurement.angular_velocity =
                    Eigen::Vector3d(imu_data.angular_velocity.x, imu_data.angular_velocity.y,
                                    imu_data.angular_velocity.z);
                base_imu_data_.pop();

                // Create the leg F/T measurement
                std::map<std::string, serow::ForceTorqueMeasurement> ft_measurements;
                for (auto& [key, value] : ft_data_) {
                    if (value.size()) {
                        serow::ForceTorqueMeasurement ft{};
                        const auto& ft_data = value.front();
                        ft.timestamp = static_cast<double>(ft_data.header.stamp.sec) +
                                       static_cast<double>(ft_data.header.stamp.nsec) * 1e-9;
                        ft.force = Eigen::Vector3d(ft_data.wrench.force.x, ft_data.wrench.force.y,
                                                   ft_data.wrench.force.z);
                        ft.torque = Eigen::Vector3d(ft_data.wrench.torque.x,
                                                    ft_data.wrench.torque.y,
                                                    ft_data.wrench.torque.z);
                        ft_measurements[key] = std::move(ft);
                        value.pop();
                    }
                }

                serow_.filter(imu_measurement, joint_measurements,
                            ft_measurements.size() > 0 ? std::make_optional(ft_measurements)
                                                       : std::nullopt);

                auto state = serow_.getState();
                odom_.header.seq  += 1;
                odom_.header.stamp = base_timestamp;
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
                const Eigen::Matrix<double, 6, 6>& base_velocity_cov = state->getBaseVelocityCov();
                for (size_t i = 0; i < 6; i++) {
                    for (size_t j = 0; j < 6; j++) {
                        // Row-major
                        odom_.pose.covariance[i * 6 + j] = base_pose_cov(i, j);
                        odom_.twist.covariance[i * 6 + j] = base_velocity_cov(i, j);
                    }
                }
                odom_publisher_.publish(odom_);
            }

            ros::spinOnce();
            loop_rate.sleep();
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber joint_state_subscription_;
    ros::Subscriber base_imu_subscription_;
    ros::Publisher odom_publisher_;

    std::vector<ros::Subscriber> force_torque_state_subscriptions_;
    std::queue<sensor_msgs::JointState> joint_state_data_;
    std::queue<sensor_msgs::Imu> base_imu_data_;
    std::map<std::string, std::queue<geometry_msgs::WrenchStamped>> ft_data_;
    double loop_rate_{};

    nav_msgs::Odometry odom_;
    
    serow::Serow serow_;
};


int main(int argc, char** argv) {
    ros::init(argc, argv, "serow_driver");
    ros::NodeHandle nh;

    SerowDriver driver(nh);

    return 0;
}