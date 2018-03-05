# README
SEROW (State Estimation RObot Walking) Framework for Humanoid Robot Walking Estimation.  The code is open-source (BSD License). Please note that this work is an on-going research and thus some parts are not fully developed yet. Furthermore, the code will be subject to changes in the future which could include greater re-factoring.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* Ubuntu 14.04 and later
* ROS indigo and later
* Eigen 3.2.0 and later
* [hrl_kinematics](http://wiki.ros.org/hrl_kinematics) 

## Installing
* git clone https://github.com/ahornung/hrl_kinematics.git
* git clone https://github.com/mrsp/serow.git
* catkin_make

## Minimum Robot Requirements
#### Using the Rigid Body Estimator to estimate 3D-Body Position/Orientation/Velocity, 3D-Support foot Position/Orientation and IMU biases.
* Robot State Publisher (e.g. topics: /joint_states, /tf)
* IMU (e.g. topic /imu0)
* Feet Force Sensors for detecting contact (e.g. topic: /left_leg/force_torque_states, /right_leg/force_torque_states)

#### Using the full cascade framework (Rigid Body Estimator + CoM Estimator) to estimate 3D-Body Position/Orientation/Velocity, 3D-Support foot Position/Orientation and IMU biases, 3D-CoM Position/Velocity and 3D-External Forces on CoM.
* Robot State Publisher (e.g. topics: /joint_states, /tf)
* IMU(e.g. topic /imu0)
* Feet Force Sensors  + Center of Pressure (COP) measurements in the local foot frame (e.g. topics /left_leg/force_torque_states, /right_leg/force_torque_states, /left_leg/COP, /right_leg/COP)

#### Using our humanoid_fsr package
If your robot is employed with feet force sensors and you have available a measurement for each sensor, then you can use our humanoid_fsr package to compute the COP and Force/Torque measurements in each leg.  This package automatically generates the required by SEROW /left_leg/force_torque_states, /right_leg/force_torque_states, /left_leg/COP, /right_leg/COP topics.


## ROS Examples
### Valkyrie SRCsim
* Download the valkyrie bag file from [valk_bagfile](http://users.ics.forth.gr/~spiperakis/valk.bag)
* roscore
* rosbag play -s 1.0 --pause valk.bag
* roslaunch humanoid_state_estimation humanoid_estimator_driver_valkyrie.launch
* hit space to unpause the rosbag play

![valk](img/valk.jpg)
### NAO Walking on rough terrain outdoors
* Download the nao bag file from [nao_bagfile](http://users.ics.forth.gr/~spiperakis/nao.bag)
* roscore
* rosbag play --pause nao.bag
* roslaunch humanoid_state_estimation humanoid_estimator_driver_nao.launch
* hit space to unpause the rosbag play

![nao](img/nao.jpg)
### Launch on your Robot in real time
* Specify topics on config/estimation_params.yaml
* roslaunch humanoid_state_estimation humanoid_estimator_driver.launch
* roslaunch rqt_reconfigure rqt_reconfigure (If you want to reconfiqure filter params online -> easy tuning).
## License
[BSD](LICENSE) 

