cmake_minimum_required(VERSION 3.8)
project(serow_ros2)

# Default to C++17
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Default to Release build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(serow REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(message_filters REQUIRED)

# Add executable
add_executable(serow_ros2 src/serow_driver.cpp src/SerowRos2.cpp)
target_include_directories(serow_ros2 PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIR}
)

# Link dependencies
target_link_libraries(serow_ros2 
  serow 
  yaml-cpp 
  pinocchio::pinocchio 
  Eigen3::Eigen 
  nlohmann_json::nlohmann_json
)

ament_target_dependencies(serow_ros2
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  ament_index_cpp
  message_filters
)

# Install targets
install(TARGETS serow_ros2
  DESTINATION lib/${PROJECT_NAME}
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(
  rclcpp
  nav_msgs
  sensor_msgs
  geometry_msgs
  ament_index_cpp
  yaml-cpp
  serow
)

ament_package()
