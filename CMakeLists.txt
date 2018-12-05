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
)

find_package(PkgConfig REQUIRED)

PKG_CHECK_MODULES(EIGEN3 eigen3>=3.2 REQUIRED)
PKG_CHECK_MODULES(PINOCCHIO pinocchio REQUIRED)
PKG_CHECK_MODULES(URDFDOM urdfdom>=0.2 REQUIRED)
# Special care for Eigen>3.2.10
IF(EIGEN3_FOUND)
  IF(${EIGEN3_VERSION} VERSION_GREATER "3.2.10")
    ADD_DEFINITIONS(-DEIGEN3_FUTURE)
    SET(EIGEN3_FUTURE TRUE)
    SET(_PKG_CONFIG_CFLAGS "${_PKG_CONFIG_CFLAGS} -DEIGEN3_FUTURE" CACHE INTERNAL "")
    IF(${EIGEN3_VERSION} VERSION_LESS "3.3.0")
      ADD_DEFINITIONS(-DEIGEN3_BETA_3_2_9x)
      SET(EIGEN3_FUTURE TRUE)
      SET(_PKG_CONFIG_CFLAGS "${_PKG_CONFIG_CFLAGS} -DEIGEN3_BETA_3_2_9x" CACHE INTERNAL "")
    ENDIF(${EIGEN3_VERSION} VERSION_LESS "3.3.0")
  ELSE(${EIGEN3_VERSION} VERSION_GREATER "3.2.10")
    SET(EIGEN3_FUTURE FALSE)
  ENDIF(${EIGEN3_VERSION} VERSION_GREATER "3.2.10")
ENDIF(EIGEN3_FOUND)

# Special care for urdfdom versions
IF(URDFDOM_FOUND)
  IF(${URDFDOM_VERSION} VERSION_LESS "0.3.0")
    ADD_DEFINITIONS(-DURDFDOM_COLLISION_WITH_GROUP_NAME)
    SET(_PKG_CONFIG_CFLAGS "${_PKG_CONFIG_CFLAGS} -DURDFDOM_COLLISION_WITH_GROUP_NAME" CACHE INTERNAL "")
  ENDIF(${URDFDOM_VERSION} VERSION_LESS "0.3.0")
  # Types from version 0.4.0
  IF(NOT ${URDFDOM_VERSION} VERSION_LESS "0.4.0")
    ADD_DEFINITIONS(-DURDFDOM_TYPEDEF_SHARED_PTR)
    SET(_PKG_CONFIG_CFLAGS "${_PKG_CONFIG_CFLAGS} -DURDFDOM_TYPEDEF_SHARED_PTR" CACHE INTERNAL "")
  ENDIF(NOT ${URDFDOM_VERSION} VERSION_LESS "0.4.0")
  # From version 1.0.0, std::shared_ptr appears
  IF(${URDFDOM_VERSION} VERSION_GREATER "0.4.2")
    ADD_DEFINITIONS(-DURDFDOM_USE_STD_SHARED_PTR)
    SET(_PKG_CONFIG_CFLAGS "${_PKG_CONFIG_CFLAGS} -DURDFDOM_USE_STD_SHARED_PTR" CACHE INTERNAL "")
  ENDIF(${URDFDOM_VERSION} VERSION_GREATER "0.4.2")
ENDIF(URDFDOM_FOUND)




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

add_executable(serow src/serow_driver.cpp src/humanoid_ekf.cpp  src/IMUEKF.cpp src/JointSSKF.cpp src/MovingAverageFilter.cpp src/CoMEKF.cpp src/differentiator.cpp src/butterworthLPF.cpp src/JointDF.cpp)
target_link_libraries(serow ${catkin_LIBRARIES} ${Eigen3_LIBRARIES} ${PINOCCHIO_LIBRARIES})

add_dependencies(serow ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME}_gencfg)
