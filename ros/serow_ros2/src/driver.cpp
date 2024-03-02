#include "geometry_msgs/msg/wrench.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#include <functional>

using std::placeholders::_1;

class SerowDriver : public rclcpp::Node {
   public:
    SerowDriver(const std::string& joint_state_topic, const std::string& imu_topic,
                const std::vector<std::string>& force_torque_state_topics)
        : Node("serow_driver") {
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_state_topic, 1000, std::bind(&SerowDriver::joint_state_topic_callback, this, _1));
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, 1000, std::bind(&SerowDriver::imu_topic_callback, this, _1));

        // Dynamically create a wrench callback one for each limp
        for (const auto& ft_topic : force_torque_state_topics) {
            auto force_torque_state_topic_callback = [this](const geometry_msgs::msg::Wrench& msg) {
                RCLCPP_INFO(this->get_logger(), "wrench callback");
            };
            force_torque_state_topic_callbacks_.push_back(
                std::move(force_torque_state_topic_callback));
            auto ft_subscription = this->create_subscription<geometry_msgs::msg::Wrench>(
                ft_topic, 1000, force_torque_state_topic_callbacks_.back());
            force_torque_state_subscriptions_.push_back(std::move(ft_subscription));
        }
    }

   private:
    void joint_state_topic_callback(const sensor_msgs::msg::JointState& msg) const {
        RCLCPP_INFO(this->get_logger(), "joint state callback");
    }

    void imu_topic_callback(const sensor_msgs::msg::Imu& msg) const {
        RCLCPP_INFO(this->get_logger(), "imu callback");
    }

    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::Wrench>::SharedPtr>
        force_torque_state_subscriptions_;
    std::vector<std::function<void(const geometry_msgs::msg::Wrench&)>>
        force_torque_state_topic_callbacks_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    std::vector<std::string> force_torque_state_topics;
    force_torque_state_topics.push_back("/left_leg/force_torque_states");
    force_torque_state_topics.push_back("/right_leg/force_torque_states");
    rclcpp::spin(
        std::make_shared<SerowDriver>("/joint_states", "/imu0", force_torque_state_topics));
    rclcpp::shutdown();
    return 0;
}
