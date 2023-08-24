#include <iostream>
#include <unordered_map>
#include <string>
#include "ContactEKF.hpp"
#include "State.hpp"

int main() {
    State state;
    state.point_feet_ =  true;
    state.num_leg_ee_ = 1; 
    Eigen::Isometry3d foot_pose = Eigen::Isometry3d::Identity();
    foot_pose.translation() = Eigen::Vector3d(0.1, 0.5, -0.5);
    state.foot_pose_.insert({"left_foot", foot_pose});
    state.foot_frames_.insert({"left_foot"});
    state.foot_contact_.insert({"left_foot", true});

    ContactEKF base_ekf;
    Eigen::Vector3d g = Eigen::Vector3d(0, 0, -9.81);
    base_ekf.init(state);

    ImuMeasurement imu;
    imu.timestamp = 0.1;
    imu.linear_acceleration = Eigen::Vector3d(0.1, -0.1, 0.05) - g;
    imu.angular_velocity = Eigen::Vector3d(-0.1, 0.1, 0.0);

    KinematicMeasurement kin;
    kin.timestamp = 0.1;
    kin.contacts_position.insert({"left_foot", Eigen::Vector3d(0.1, 0.5, -0.5)});
    kin.contacts_status.insert({"left_foot", true});

    State predicted_state = base_ekf.predict(state, imu, kin);

    std::cout << "Base position after predict " << predicted_state.base_position_.transpose()
              << std::endl;
    std::cout << "Base velocity after predict " << predicted_state.base_linear_velocity_.transpose()
              << std::endl;
    std::cout << "Base orientation after predict " << predicted_state.base_orientation_<<std::endl;
    std::cout << "Left contact position after predict "
              << predicted_state.foot_pose_.at("left_foot").translation().transpose() << std::endl;
    return 0;
}
