

# README
SEROW (State Estimation RObot Walking) Framework for Humanoid/Quadruped Robot Walking Estimation.  The code is open-source (BSD License). Please note that this work is an on-going research and thus some parts are not fully developed yet. Furthermore, the code will be subject to changes in the future which could include greater re-factoring.


# SEROW in Real-time
------------------------------------------------------------------ 

| Cogimon and SEROW  | Centauro and SEROW |
| ------------- | ------------- |
| [![YouTube Link](img/cogimon.png)  ](https://www.youtube.com/watch?v=MLmfgADDjj0)  | [![YouTube Link](img/centauro.png)  ](https://www.youtube.com/watch?v=cVWS8oopr_M) |

------------------------------------------------------------------ 


Relevant Papers:
* Non-linear ZMP based State Estimation for Humanoid Robot Locomotion, https://ieeexplore.ieee.org/document/7803278 (Humanoids 2016 - nominated for the best interactive paper award)
* Nonlinear State Estimation for Humanoid Robot Walking, https://ieeexplore.ieee.org/document/8403285 (RA-L + IROS 2018)
* Outlier-Robust State Estimation for Humanoid Robots, https://ieeexplore.ieee.org/document/8968152 (IROS 2019)

More Videos: 
* https://www.youtube.com/watch?v=nkzqNhf3_F4
* https://www.youtube.com/watch?v=9OvIBg8tn54
* https://www.youtube.com/watch?v=ojogeY3xSsw


# Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
* Ubuntu 16.04 and later
* ROS kinetic and later
* Eigen 3.2.0 and later
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio) 2.2.1

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
* Feet Pressure or Force/Torque Sensors for detecting contact (e.g. topic: /left_leg/force_torque_states, /right_leg/force_torque_states)

### Using the full cascade framework (Rigid Body Estimator + CoM Estimator) to estimate:
* 3D-Body Position/Orientation/Velocity
* 3D-Support Foot Position/Orientation
* IMU biases
* 3D-CoM Position/Velocity
* 3D-External Forces on CoM
### Requirements:
* Robot State Publisher (e.g. topic: /joint_states)
* IMU(e.g. topic /imu0)
* Feet Pressure or Force/Torque Sensors  for Center of Pressure (COP) measurements in the local foot frame (e.g. topics /left_leg/force_torque_states, /right_leg/force_torque_states)

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

### Quadruped State Estimation
* Specify similar parameters as in config/estimation_params_centauro.yaml

## Citation
Upon usage in an academic work kindly cite: <br/>

@ARTICLE{PiperakisRAL18, <br/>
    author={S. {Piperakis} and M. {Koskinopoulou} and P. {Trahanias}}, <br/>
    journal={IEEE Robotics and Automation Letters}, <br/>
    title={{Nonlinear State Estimation for Humanoid Robot Walking}}, <br/>
    year={2018}, <br/>
    volume={3}, <br/>
    number={4}, <br/>
    pages={3347-3354}, <br/>
    doi={10.1109/LRA.2018.2852788}, <br/>
    month={Oct},<br/>
}<br/>

## License
[BSD](LICENSE) 

