#include <functional>
#include <map>
#include <serow/Serow.hpp>

#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using std::placeholders::_1;

class SerowDriver : public rclcpp::Node {
public:
    SerowDriver(const std::string& joint_state_topic, const std::string& base_imu_topic,
                const std::vector<std::string>& force_torque_state_topics,
                const std::string& config_file_path)
        : Node("serow_ros2_driver") {
        // Initialize SERoW
        serow_.initialize(config_file_path);

        // Create subscribers
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_state_topic, 1, std::bind(&SerowDriver::joint_state_topic_callback, this, _1));
        base_imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            base_imu_topic, 1, std::bind(&SerowDriver::base_imu_topic_callback, this, _1));

        // Dynamically create a wrench callback one for each limp
        for (const auto& ft_topic : force_torque_state_topics) {
            auto force_torque_state_topic_callback =
                [this](const geometry_msgs::msg::WrenchStamped& msg) {
                    this->ft_data_[msg.header.frame_id] = msg;
                };
            force_torque_state_topic_callbacks_.push_back(
                std::move(force_torque_state_topic_callback));
            auto ft_subscription = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
                ft_topic, 1, force_torque_state_topic_callbacks_.back());
            force_torque_state_subscriptions_.push_back(std::move(ft_subscription));
        }
        RCLCPP_INFO(this->get_logger(), "SERoW was initialized successfully");
        run();
    }

    void run() {
        while (rclcpp::ok()) {
            if (joint_state_data_.has_value() > 0 && base_imu_data_.has_value() > 0) {
                // Create the joint measurements
                const auto& joint_state_data = joint_state_data_.value();
                std::map<std::string, serow::JointMeasurement> joint_measurements;
                for (unsigned int i = 0; i < joint_state_data.name.size(); i++) {
                    serow::JointMeasurement joint{};
                    joint.timestamp =
                        static_cast<double>(joint_state_data.header.stamp.sec) +
                        static_cast<double>(joint_state_data.header.stamp.nanosec) * 1e-9;
                    joint.position = joint_state_data.position[i];
                    joint_measurements[joint_state_data.name[i]] = std::move(joint);
                }

                // Create the base imu measurement
                const auto& imu_data = base_imu_data_.value();
                serow::ImuMeasurement imu_measurement{};
                imu_measurement.timestamp =
                    static_cast<double>(imu_data.header.stamp.sec) +
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
                        ft.torque.emplace(Eigen::Vector3d(ft_data.wrench.torque.x,
                                                          ft_data.wrench.torque.y,
                                                          ft_data.wrench.torque.z));
                        ft_measurements[key] = std::move(ft);
                    }
                    ft_data_.clear();
                }
                RCLCPP_INFO(this->get_logger(), "Filtering");
                serow_.filter(imu_measurement, joint_measurements,
                              ft_measurements.size() == force_torque_state_subscriptions_.size()
                                  ? std::make_optional(ft_measurements)
                                  : std::nullopt);

                const auto& state = serow_.getState();
                if (state) {
                    std::cout << "State: " << state->getBasePosition().transpose() << std::endl;
                }

                joint_state_data_.reset();
                base_imu_data_.reset();
            }
        }
    }

private:
    void joint_state_topic_callback(const sensor_msgs::msg::JointState& msg) {
        this->joint_state_data_ = msg;
    }

    void base_imu_topic_callback(const sensor_msgs::msg::Imu& msg) { this->base_imu_data_ = msg; }

    serow::Serow serow_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr base_imu_subscription_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr>
        force_torque_state_subscriptions_;
    std::vector<std::function<void(const geometry_msgs::msg::WrenchStamped&)>>
        force_torque_state_topic_callbacks_;
    std::optional<sensor_msgs::msg::Imu> base_imu_data_;
    std::optional<sensor_msgs::msg::JointState> joint_state_data_;
    std::map<std::string, geometry_msgs::msg::WrenchStamped> ft_data_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    std::vector<std::string> force_torque_state_topics;
    force_torque_state_topics.push_back("/h1/left_ankle/force_torque_states");
    force_torque_state_topics.push_back("/h1/right_ankle/force_torque_states");
    rclcpp::spin(std::make_shared<SerowDriver>("/h1/joint_states", "/h1/imu",
                                               force_torque_state_topics, "h1.json"));
    rclcpp::shutdown();
    return 0;
}
