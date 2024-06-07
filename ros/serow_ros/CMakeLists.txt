cmake_minimum_required(VERSION 3.0.2)
project(serow_ros)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  nav_msgs
)

find_package(Boost REQUIRED NO_MODULE)
find_package(serow REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
)

add_executable(serow_driver src/serow_driver.cpp)
target_link_libraries(serow_driver ${catkin_LIBRARIES} Eigen3::Eigen pinocchio::pinocchio serow)
add_dependencies(serow_driver ${catkin_EXPORTED_TARGETS})

set_property(TARGET serow_driver PROPERTY CXX_STANDARD 17)
