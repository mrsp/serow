cmake_minimum_required(VERSION 3.22.1...4.0.0)
project(serow_hypertuner VERSION 1.0.0 LANGUAGES CXX)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# Dependencies
find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

find_library(BAYESOPT_LIB bayesopt PATHS /usr/local/lib /usr/lib REQUIRED)
find_library(NLOPT_LIB nlopt PATHS /usr/local/lib /usr/lib REQUIRED)

find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Include local headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# Source files
set(SOURCES
  src/HyperTuner.cpp
  src/BayesOptimizer.cpp
  src/DataManager.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Executable
add_executable(hypertuner ${SOURCES})

# Link dependencies
target_link_libraries(hypertuner PRIVATE
  serow
  pinocchio::pinocchio
  nlohmann_json::nlohmann_json
  pinocchio::pinocchio
  ${HDF5_LIBRARIES}
  ${BAYESOPT_LIB}
  ${NLOPT_LIB}
)
