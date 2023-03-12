cmake_minimum_required(VERSION 3.0.0)
project(serow VERSION 0.1.0)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(serow test.cpp State.cpp)
target_link_libraries (serow Eigen3::Eigen)