cmake_minimum_required(VERSION 3.22)
project(serow_tests VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(anymal_csv_test ../anymal_csv_test.cpp)

target_link_libraries(anymal_csv_test PRIVATE 
    serow 
    pinocchio::pinocchio 
    nlohmann_json::nlohmann_json
)
