cmake_minimum_required(VERSION 3.22)
project(serow_tests VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()
find_package(mcap)

# 1. Find Main Dependencies
find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

# 2. Find MCAP and Compression Libs (System Installed)
# We use find_library to locate the libmcap you just installed
find_path(MCAP_INCLUDE_DIR NAMES mcap/writer.hpp PATH_SUFFIXES include)
find_library(MCAP_LIBRARY NAMES mcap PATH_SUFFIXES lib)

if (NOT MCAP_LIBRARY OR NOT MCAP_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find MCAP. Make sure you ran 'sudo make install' in mcap_builder.")
endif()

# Find compression libraries (Required because libmcap.a depends on them)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
pkg_check_modules(ZSTD REQUIRED libzstd)

# 3. Create Executable
add_executable(go2_mujoco_test ../go2_mujoco_test.cpp)

# 4. Link Libraries
# We link mcap, plus lz4 and zstd explicitly to resolve the 'vtable' and 'undefined reference' errors.
target_link_libraries(go2_mujoco_test PRIVATE 
    serow 
    mcap
    pinocchio::pinocchio 
    nlohmann_json::nlohmann_json
    ${LZ4_LIBRARIES}
    ${ZSTD_LIBRARIES}
)

# 5. Include Directories
target_include_directories(go2_mujoco_test PRIVATE 
    ${MCAP_INCLUDE_DIR}
    ${LZ4_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIRS}
)
