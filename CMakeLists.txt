cmake_minimum_required(VERSION 2.8.3)
project(serow)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  dynamic_reconfigure
  pinocchio
)





## Generate dynamic reconfigure parameters in the 'cfg' folder
generate_dynamic_reconfigure_options(
   cfg/VarianceControl.cfg
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES serow
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs dynamic_reconfigure
  DEPENDS Eigen3 pinocchio 
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
  ${PINOCCHIO_INCLUDE_DIRS}
)

link_directories(  
${PINOCCHIO_LIBRARY_DIRS})

add_executable(serow src/serow_driver.cpp src/humanoid_ekf.cpp  src/IMUEKF.cpp src/JointSSKF.cpp src/MovingAverageFilter.cpp src/CoMEKF.cpp src/differentiator.cpp src/butterworthLPF.cpp src/JointDF.cpp src/butterworthHPF.cpp src/IMUinEKF.cpp)
target_link_libraries(serow ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} ${PINOCCHIO_LIBRARIES})

add_dependencies(serow ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)
