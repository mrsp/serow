cmake_minimum_required(VERSION 2.8.3)
project(humanoid_state_estimation)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf
  hrl_kinematics
  dynamic_reconfigure
)
find_package(Eigen3 REQUIRED)

## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
   cfg/VarianceControl.cfg
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES humanoid_state_estimation
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs tf hrl_kinematics  dynamic_reconfigure
  DEPENDS Eigen3
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIRS}
)

add_executable(humanoid_state_estimation src/humanoid_state_estimator_driver.cpp src/humanoid_ekf.cpp  src/IMUEKF.cpp src/JointSSKF.cpp src/MovingAverageFilter.cpp src/CoMEKF.cpp src/differentiator.cpp src/butterworthLPF.cpp)
target_link_libraries(humanoid_state_estimation ${catkin_LIBRARIES} ${Eigen3_LIBRARIES})

add_dependencies(humanoid_state_estimation ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)

