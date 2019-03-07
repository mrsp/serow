SEROW V2.0

# README
SEROW (State Estimation RObot Walking) Framework for Humanoid Robot Walking Estimation.  The code is open-source (BSD License). Please note that this work is an on-going research and thus some parts are not fully developed yet. Furthermore, the code will be subject to changes in the future which could include greater re-factoring.

Video: https://www.youtube.com/watch?v=nkzqNhf3_F4

Papers: 
* Nonlinear State Estimation for Humanoid Robot Walking, https://ieeexplore.ieee.org/document/8403285 (RA-L + IROS 2018)
* Robust Gaussian Error-State Kalman Filter

Upon requests a matlab version will be released shortly.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* Ubuntu 16.04 and later
* ROS kinetic and later
* Eigen 3.2.0 and later
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio) 

## Installing
* sudo apt-get install ros-kinetic-pinocchio
* git clone https://github.com/mrsp/serow.git
* catkin_make -DCMAKE_BUILD_TYPE=Release 
* If you are using catkin tools run: catkin build  --cmake-args -DCMAKE_BUILD_TYPE=Release 

## Minimum Robot Requirements
### Using the Rigid Body Estimator to estimate: 
* 3D-Body Position/Orientation/Velocity
* 3D-Support Foot Position/Orientation
* IMU biases
### Requirements
* Robot State Publisher (e.g. topic: /joint_states)
* IMU (e.g. topic /imu0)
* Feet Force Sensors for detecting contact (e.g. topic: /left_leg/force_torque_states, /right_leg/force_torque_states)

### Using the full cascade framework (Rigid Body Estimator + CoM Estimator) to estimate:
* 3D-Body Position/Orientation/Velocity
* 3D-Support Foot Position/Orientation
* IMU biases
* 3D-CoM Position/Velocity
* 3D-External Forces on CoM
### Requirements:
* Robot State Publisher (e.g. topic: /joint_states)
* IMU(e.g. topic /imu0)
* Feet Force Sensors  + Center of Pressure (COP) measurements in the local foot frame (e.g. topics /left_leg/force_torque_states, /right_leg/force_torque_states, /left_leg/COP, /right_leg/COP)

### Using our humanoid_fsr package
If your robot is employed with feet force sensors and you have available a measurement for each sensor, then you can use our [humanoid_fsr](https://github.com/mrsp/humanoid_fsr) package to compute the COP and 3D - Force/Torque measurements in each leg.  This package automatically generates the required by SEROW /left_leg/force_torque_states, /right_leg/force_torque_states, /left_leg/COP, /right_leg/COP topics.

### Using our humanoid_cop package
If your robot is employed with F/T sensors and you have available a 6D wrench measurement for each leg, then you can use our [humanoid_cop](https://github.com/mrsp/humanoid_cop) package to compute the COP in each leg.  This package automatically generates the required by SEROW  /left_leg/COP, /right_leg/COP topics.


### Using our serow_utils package
Use the [serow_utils](https://github.com/mrsp/serow_utils) to visualize the estimated trajectories and to contrast them with other trajectories (e.g. ground_truth).

## ROS Examples
### Valkyrie SRCsim
* Download the valkyrie bag file from [valk_bagfile](http://users.ics.forth.gr/~spiperakis/valk.bag)
* roscore
* rosbag play --pause valk.bag
* roslaunch serow serow_valkyrie.launch
* roslaunch serow_utils serow_utils.launch
* hit space to unpause the rosbag play

![valk](img/valk.png)
### NAO Walking on rough terrain outdoors
* Download the nao bag file from [nao_bagfile](http://users.ics.forth.gr/~spiperakis/nao.bag)
* roscore
* rosbag play --pause nao.bag
* roslaunch serow serow_nao.launch
* roslaunch serow_utils serow_utils.launch
* hit space to unpause the rosbag play

![nao](img/nao.jpg)
### Launch on your Robot in real time
* Specify topics on config/estimation_params.yaml
* roslaunch serow serow.launch
## License
[BSD](LICENSE) 

