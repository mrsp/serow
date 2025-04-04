cmake_minimum_required(VERSION 3.10)
project(serow_python)

# Find required packages
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

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
pybind11_add_module(contact_ekf contact_ekf_bindings.cpp)
pybind11_add_module(state state_bindings.cpp)

# Set include directories
target_include_directories(contact_ekf PRIVATE ${CMAKE_SOURCE_DIR}/../../core/src)
target_include_directories(state PRIVATE ${CMAKE_SOURCE_DIR}/../../core/src)

# Link against Eigen and core library
target_link_libraries(contact_ekf PRIVATE Eigen3::Eigen ${SEROW_CORE_LIB})
target_link_libraries(state PRIVATE Eigen3::Eigen ${SEROW_CORE_LIB})

# Set C++ standard
set_target_properties(contact_ekf PROPERTIES CXX_STANDARD 17)
set_target_properties(state PROPERTIES CXX_STANDARD 17) 