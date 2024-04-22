#include <functional>
#include <map>
#include <queue>
#include <serow/Serow.hpp>

#include "geometry_msgs/msg/wrench_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using std::placeholders::_1;

class SerowDriver : public rclcpp::Node {
   public:
    SerowDriver(const std::string& joint_state_topic, const std::string& imu_topic,
                const std::vector<std::string>& force_torque_state_topics,
                const std::string& config_file_path)
        : Node("serow_driver") {
        SEROW_ = serow::Serow(config_file_path);
        joint_state_subscription_ = this->create_subscription<sensor_msgs::msg::JointState>(
            joint_state_topic, 1000, std::bind(&SerowDriver::joint_state_topic_callback, this, _1));
        imu_subscription_ = this->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic, 1000, std::bind(&SerowDriver::imu_topic_callback, this, _1));

        // Dynamically create a wrench callback one for each limp
        for (const auto& ft_topic : force_torque_state_topics) {
            auto force_torque_state_topic_callback =
                [this](const geometry_msgs::msg::WrenchStamped& msg) {
                    this->ft_data_[msg.header.frame_id].push(msg);
                    RCLCPP_INFO(this->get_logger(), "wrench callback %ld",
                                this->ft_data_.at(msg.header.frame_id).size());
                    if (this->ft_data_.at(msg.header.frame_id).size() > 100) {
                        this->ft_data_.at(msg.header.frame_id).pop();
                    }
                };
            force_torque_state_topic_callbacks_.push_back(
                std::move(force_torque_state_topic_callback));
            auto ft_subscription = this->create_subscription<geometry_msgs::msg::WrenchStamped>(
                ft_topic, 1000, force_torque_state_topic_callbacks_.back());
            force_torque_state_subscriptions_.push_back(std::move(ft_subscription));
        }
    }

   private:
    void joint_state_topic_callback(const sensor_msgs::msg::JointState& msg) {
        this->joint_state_data_.push(msg);
        RCLCPP_INFO(this->get_logger(), "joint state callback %ld", this->joint_state_data_.size());
        if (this->joint_state_data_.size() > 100) {
            this->joint_state_data_.pop();
        }
    }

    void imu_topic_callback(const sensor_msgs::msg::Imu& msg) {
        this->imu_data_.push(msg);
        RCLCPP_INFO(this->get_logger(), "imu callback %ld", this->imu_data_.size());
        if (this->imu_data_.size() > 100) {
            this->imu_data_.pop();
        }
    }

    serow::Serow SEROW_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_subscription_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscription_;
    std::vector<rclcpp::Subscription<geometry_msgs::msg::WrenchStamped>::SharedPtr>
        force_torque_state_subscriptions_;
    std::vector<std::function<void(const geometry_msgs::msg::WrenchStamped&)>>
        force_torque_state_topic_callbacks_;
    std::queue<sensor_msgs::msg::Imu> imu_data_;
    std::queue<sensor_msgs::msg::JointState> joint_state_data_;
    std::map<std::string, std::queue<geometry_msgs::msg::WrenchStamped>> ft_data_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    std::vector<std::string> force_torque_state_topics;
    force_torque_state_topics.push_back("/left_leg/force_torque_states");
    force_torque_state_topics.push_back("/right_leg/force_torque_states");
    rclcpp::spin(std::make_shared<SerowDriver>("/joint_states", "/imu0", force_torque_state_topics,
                                               "../config/nao.json"));
    rclcpp::shutdown();
    return 0;
}
