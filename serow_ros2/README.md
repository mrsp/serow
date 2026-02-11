# Getting Started
These instructions will get you a copy of the project up and running on your local machine for testing purposes with ROS2.

## Install
* Install [SEROW](https://github.com/mrsp/serow/tree/devel?tab=readme-ov-file#getting-started)
* Create a ROS workspace with `mkdir -p ros2_ws/src`
* `cd ros2_ws/src`
* `ln -s $SEROW_PATH/serow_ros2 ./serow_ros2`
* `cd .. && colcon build --packages-select serow_ros2
* `source install/setup.bash`

## Minimum Robot Requirements
### Using the Base Estimator to estimate: 
* 3D-Base position/orientation/linear velocity
* 3D-Contact foot position/orientation
* IMU biases

### Requirements
* Robot state publisher (e.g. topic: `/joint_states`)
* IMU (e.g. topic `/imu0`)
* Feet Pressure or Force/Torque sensors for detecting contact (e.g. topic: `/left_leg/force_torque_states`, `/right_leg/force_torque_states`)

### Using the full cascade framework (Base Estimator + CoM Estimator) to estimate:
* 3D-Base position/orientation/linear velocity
* 3D-Contact foot position/orientation
* IMU biases
* 3D-CoM position/linear velocity
* 3D-External forces on CoM

### Requirements:
* Robot State Publisher (e.g. topic: `/joint_states`)
* IMU (e.g. topic `/imu0`)
* Feet Pressure or Force/Torque sensors for Center of Pressure (COP) computation in the local foot frame (e.g. topics `/left_leg/force_torque_states`, `/right_leg/force_torque_states`)

## ROS2 Examples
* Download the go2 bag file from
* `ros2 bag play go2.bag`
* `ros2 launch serow_ros2 serow_go2.launch.py`

## License
[GNU GPLv3](LICENSE) 
