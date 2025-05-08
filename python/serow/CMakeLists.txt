cmake_minimum_required(VERSION 3.22.1...4.0.0)
project(serow_python)

# Set Python version and paths
set(Python_ADDITIONAL_VERSIONS 3.10)
find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter Development)
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARY ${Python3_LIBRARIES})

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

# Find the core library
find_library(SEROW_CORE_LIB serow
    PATHS 
        ${CMAKE_SOURCE_DIR}/../../build/core/src
        ${CMAKE_SOURCE_DIR}/../../build
    NO_DEFAULT_PATH
)

if(NOT SEROW_CORE_LIB)
    message(FATAL_ERROR "Could not find serow library. Please build the core library first by running:\n"
        "cd ${CMAKE_SOURCE_DIR}/../..\n"
        "mkdir -p build\n"
        "cd build\n"
        "cmake ..\n"
        "make")
endif()

# Add Python module
pybind11_add_module(serow serow_bindings.cpp)

# Set include directories
target_include_directories(serow PRIVATE 
    ${CMAKE_SOURCE_DIR}/../../core/src
)

# Link against Eigen, Pinocchio and core library
target_link_libraries(serow PRIVATE 
    Eigen3::Eigen 
    ${SEROW_CORE_LIB}
    pinocchio::pinocchio
)

# Set C++ standard
set_target_properties(serow PROPERTIES CXX_STANDARD 17)
