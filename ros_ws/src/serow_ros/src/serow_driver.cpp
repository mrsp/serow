#include <functional>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include <map>
#include "pinocchio/fwd.hpp"
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/JointState.h>
#include <serow/Serow.hpp>



class SerowDriver {
public:
    SerowDriver(const ros::NodeHandle& nh, const std::string& joint_state_topic, const std::string& base_imu_topic,
                const std::vector<std::string>& force_torque_state_topics, const std::string& config_file_path)
        : nh_(nh) {
        // Initialize SERoW
        serow_ = serow::Serow(config_file_path);
        pose_estimate_.header.frame_id = "base_link";
        // Create subscribers
        joint_state_subscription_ = nh_.subscribe(joint_state_topic, 1000, &SerowDriver::joint_state_topic_callback, this);
        base_imu_subscription_ = nh_.subscribe(base_imu_topic, 1000, &SerowDriver::base_imu_topic_callback, this);
        base_state_publisher_= nh_.advertise<geometry_msgs::PoseStamped>("/serow/base_estimate", 100);
        // Dynamically create a wrench callback one for each limb
        for (const auto& ft_topic : force_torque_state_topics) {
            auto ft_callback = boost::bind(&SerowDriver::force_torque_state_topic_callback, this, _1);
            force_torque_state_subscriptions_.push_back(nh_.subscribe<geometry_msgs::WrenchStamped>(ft_topic, 1000, ft_callback));
        }
        ROS_INFO("SERoW was initialized successfully");

        run();
    }

    void joint_state_topic_callback(const sensor_msgs::JointState::ConstPtr& msg) {
        joint_state_data_.push(*msg);
        if (joint_state_data_.size() > 100) {
            joint_state_data_.pop();
        }
    }

    void base_imu_topic_callback(const sensor_msgs::Imu::ConstPtr& msg) {
        base_imu_data_.push(*msg);
        if (base_imu_data_.size() > 100) {
            base_imu_data_.pop();
        }
    }

    void force_torque_state_topic_callback(const geometry_msgs::WrenchStamped::ConstPtr& msg) {
        std::string frame_id = msg->header.frame_id;
        ft_data_[frame_id].push(*msg);
        if (ft_data_.at(frame_id).size() > 100) {
            ft_data_.at(frame_id).pop();
        }
    }


private:

    void run() {
        ros::Rate loop_rate(100); // Define the loop rate
        while (ros::ok()) { // Check if ROS is still running
            if (joint_state_data_.size() > 0 && base_imu_data_.size() > 0) {
                // Create the joint measurements
                const auto& joint_state_data = joint_state_data_.front();
                std::map<std::string, serow::JointMeasurement> joint_measurements;
                for (unsigned int i = 0; i < joint_state_data.name.size(); i++) {
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

                auto state_ = serow_.getState();
                pose_estimate_.header.seq  += 1;
                pose_estimate_.pose.position.x = state_->getBasePosition().x();
                pose_estimate_.pose.position.y = state_->getBasePosition().y();
                pose_estimate_.pose.position.z = state_->getBasePosition().z();
                pose_estimate_.pose.orientation.x = state_->getBaseOrientation().coeffs().x();
                pose_estimate_.pose.orientation.y = state_->getBaseOrientation().coeffs().y();
                pose_estimate_.pose.orientation.z = state_->getBaseOrientation().coeffs().z();
                pose_estimate_.pose.orientation.w = state_->getBaseOrientation().coeffs().w();
                base_state_publisher_.publish(pose_estimate_);
            }
            ros::spinOnce();
            loop_rate.sleep(); 
        }
    }

    ros::NodeHandle nh_;
    ros::Subscriber joint_state_subscription_;
    ros::Subscriber base_imu_subscription_;
    ros::Publisher base_state_publisher_ ;
    
    std::vector<ros::Subscriber> force_torque_state_subscriptions_;
    std::queue<sensor_msgs::JointState> joint_state_data_;
    std::queue<sensor_msgs::Imu> base_imu_data_;
    std::map<std::string, std::queue<geometry_msgs::WrenchStamped>> ft_data_;
    geometry_msgs::PoseStamped pose_estimate_;
    serow::Serow serow_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "serow_ros_driver");
    ros::NodeHandle nh;

    std::vector<std::string> force_torque_state_topics;
    force_torque_state_topics.push_back("/ihmc_ros/valkyrie/output/foot_force/left");
    force_torque_state_topics.push_back("/ihmc_ros/valkyrie/output/foot_force/right");

    SerowDriver driver(nh, "/joint_states", "/ihmc_ros/valkyrie/output/imu/pelvis_pelvisMiddleImu", force_torque_state_topics, "valk.json");
    // ros::spin();

    return 0;
}
