cmake_minimum_required(VERSION 3.22)
project(serow_tests VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

find_package(zstd REQUIRED)
find_package(mcap)

find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

find_path(MCAP_INCLUDE_DIR NAMES mcap/writer.hpp PATH_SUFFIXES include)
find_library(MCAP_LIBRARY NAMES mcap PATH_SUFFIXES lib)

if (NOT MCAP_LIBRARY OR NOT MCAP_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find MCAP. Make sure you ran 'sudo make install' in mcap_builder.")
endif()

# Find compression libraries (Required because libmcap.a depends on them)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
pkg_check_modules(ZSTD REQUIRED libzstd)

add_executable(go2_mujoco_test ../go2_mujoco_test.cpp)

target_link_libraries(go2_mujoco_test PRIVATE 
    serow 
    mcap
    pinocchio::pinocchio 
    nlohmann_json::nlohmann_json
    ${LZ4_LIBRARIES}
    ${ZSTD_LIBRARIES}
)

target_include_directories(go2_mujoco_test PRIVATE 
    ${MCAP_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIRS}
)
