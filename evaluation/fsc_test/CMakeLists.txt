cmake_minimum_required(VERSION 3.22.1...4.0.0)
project(serow_tests VERSION 1.0.0 LANGUAGES CXX)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

find_library(BAYESOPT_LIB bayesopt PATHS /usr/local/lib /usr/lib REQUIRED)
find_library(NLOPT_LIB nlopt PATHS /usr/local/lib /usr/lib REQUIRED)

# Add include paths
include_directories(/usr/local/include)

# Find and link HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Add the executable for HDF5 and Serow processing
add_executable(fsc_test ../fsc_test.cpp)
target_link_libraries(fsc_test PRIVATE serow pinocchio::pinocchio ${HDF5_LIBRARIES} nlohmann_json::nlohmann_json)

# Add the executable for automated hyperparameter tuning
add_executable(hypertuner ../hypertuner.cpp)
target_link_libraries(hypertuner PRIVATE serow pinocchio::pinocchio ${HDF5_LIBRARIES} nlohmann_json::nlohmann_json ${BAYESOPT_LIB} ${NLOPT_LIB})
