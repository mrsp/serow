cmake_minimum_required(VERSION 3.0.2)
project(serow_ros)

add_compile_options(-std=c++17)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
)

find_package(Boost REQUIRED NO_MODULE)
find_package(serow REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(nlohmann_json REQUIRED)

catkin_package()

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
)

add_executable(serow_ros src/serow_driver.cpp)
target_link_libraries(serow_ros ${catkin_LIBRARIES} Eigen3::Eigen pinocchio::pinocchio serow)
add_dependencies(serow_ros ${catkin_EXPORTED_TARGETS})

set_property(TARGET serow_ros PROPERTY CXX_STANDARD 17)

catkin_install_python(PROGRAMS src/force_remapper.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
