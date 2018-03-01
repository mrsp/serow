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


## ROS Examples
* Download valkyrie bag file from http://users.ics.forth.gr/~spiperakis/valk.bag
* roscore
* rosbag play -s 1.0 --pause valk.bag
* roslaunch humanoid_state_estimation humanoid_estimator_driver_valkyrie.launch
* hit space to unpause the rosbag play
![valk](img/valk.jpg)
![nao](img/nao.jpg)


## License
[BSD](LICENSE) 

