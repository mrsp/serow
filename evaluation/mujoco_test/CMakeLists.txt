cmake_minimum_required(VERSION 3.22.1...4.0.0)
project(serow_tests VERSION 1.0.0 LANGUAGES CXX)

# Enable ONNX support by default
message(STATUS "If SEROW was built with ONNX support, set USE_ONNX to ON")
option(USE_ONNX "Enable ONNX Runtime support" OFF)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

# Find and link HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
include_directories(${HDF5_INCLUDE_DIRS})

# Find ONNX Runtime if enabled
if(USE_ONNX)
    find_library(ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS /usr/local/onnxruntime/lib
        NO_DEFAULT_PATH
    )
    if(NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ONNX Runtime not found. Please ensure it is installed.")
    endif()
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
endif()

# Add the executable for HDF5 and Serow processing
add_executable(go2_mujoco_test ../go2_mujoco_test.cpp)
target_link_libraries(go2_mujoco_test PRIVATE serow pinocchio::pinocchio ${HDF5_LIBRARIES} nlohmann_json::nlohmann_json)

if(USE_ONNX)
    target_link_libraries(go2_mujoco_test PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()
