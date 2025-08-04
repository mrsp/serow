#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <functional>
#include <map>
#include <serow/Serow.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using std::placeholders::_1;
namespace fs = std::filesystem;

class SerowDriver : public rclcpp::Node {
public:
    SerowDriver() : Node("serow_ros2_driver") {
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
            joint_state_topic, 10, std::bind(&SerowDriver::jointStateCallback, this, _1));

        RCLCPP_INFO(this->get_logger(), "Creating base imu subscription on topic: %s",
                    base_imu_topic.c_str());
        base_imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            base_imu_topic, 10, std::bind(&SerowDriver::baseImuCallback, this, _1));

        // Dynamically create a wrench callback one for each limb
        for (const auto& ft_topic : force_torque_state_topics) {
            auto force_torque_state_topic_callback =
                [this](const geometry_msgs::msg::WrenchStamped& msg) {
                    this->ft_data_[msg.header.frame_id] = msg;
                };
            force_torque_state_topic_callbacks_.push_back(
                std::move(force_torque_state_topic_callback));
            RCLCPP_INFO(this->get_logger(), "Creating force torque state subscription on topic: %s",
                        ft_topic.c_str());
            auto ft_subscription = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
                ft_topic, 1, force_torque_state_topic_callbacks_.back());
            force_torque_state_subscriptions_.push_back(std::move(ft_subscription));
        }

        // Create a timer for processing data instead of the blocking loop
        timer_ = this->create_wall_timer(std::chrono::milliseconds(10),  // 100Hz processing rate
                                         std::bind(&SerowDriver::run, this));

        odom_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/serow/odom", 10);
        RCLCPP_INFO(this->get_logger(), "SERoW was initialized successfully");
    }

private:
    void run() {
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
            imu_measurement.angular_velocity =
                Eigen::Vector3d(imu_data.angular_velocity.x, imu_data.angular_velocity.y,
                                imu_data.angular_velocity.z);

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
                    ft.torque.emplace(Eigen::Vector3d(
                        ft_data.wrench.torque.x, ft_data.wrench.torque.y, ft_data.wrench.torque.z));
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
                publishOdometry(state.value());
            }
        }
    }

    void publishOdometry(const serow::State& state) {
        auto odom_msg = nav_msgs::msg::Odometry();
        const double timestamp = state.getBaseTimestamp();
        const Eigen::Vector3d& base_position = state.getBasePosition();
        const Eigen::Quaterniond& base_orientation = state.getBaseOrientation();
        const Eigen::Vector3d& base_linear_velocity = state.getBaseLinearVelocity();
        const Eigen::Vector3d& base_angular_velocity = state.getBaseAngularVelocity();

        const Eigen::Matrix<double, 6, 6> base_pose_cov = state.getBasePoseCov();
        const Eigen::Matrix<double, 6, 6> base_velocity_cov = state.getBaseVelocityCov();

        odom_msg.header.stamp = rclcpp::Time(timestamp);
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "base_link";
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
    }

    void jointStateCallback(const sensor_msgs::msg::JointState& msg) {
        this->joint_state_data_ = msg;
    }

    void baseImuCallback(const sensor_msgs::msg::Imu& msg) {
        this->base_imu_data_ = msg;
    }

    serow::Serow serow_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_publisher_;
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
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    try {
        auto serow_driver = std::make_shared<SerowDriver>();
        rclcpp::spin(serow_driver);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("serow_ros2"), "Error: %s", e.what());
        return 1;
    }

    rclcpp::shutdown();
    return 0;
}
