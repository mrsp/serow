cmake_minimum_required(VERSION 3.22)
project(fsc_test_project VERSION 1.0.0 LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# --- Dependencies ---
find_package(PkgConfig REQUIRED)
find_package(zstd REQUIRED)

pkg_check_modules(LZ4 liblz4)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Find MCAP
find_package(mcap)
if (NOT mcap_FOUND)
    message(STATUS "Falling back to manual MCAP search...")
    find_path(MCAP_INCLUDE_DIR NAMES mcap/writer.hpp PATH_SUFFIXES include)
    find_library(MCAP_LIBRARY NAMES mcap PATH_SUFFIXES lib)
    if (NOT MCAP_LIBRARY OR NOT MCAP_INCLUDE_DIR)
        message(FATAL_ERROR "Could not find MCAP. Ensure headers are in include/ and libmcap.a is in lib/.")
    endif()
endif()

find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)
find_package(nlohmann_json REQUIRED)

# Assuming you build from within a build/ directory
add_executable(fsc_test ../fsc_test.cpp)

target_compile_definitions(fsc_test PRIVATE MCAP_COMPRESSION_NO_LZ4)

# Include directories
if (NOT mcap_FOUND)
    target_include_directories(fsc_test PRIVATE ${MCAP_INCLUDE_DIR})
endif()

target_include_directories(fsc_test PRIVATE 
    ${LZ4_INCLUDE_DIRS}
    ${ZSTD_INCLUDE_DIRS}
)

target_link_libraries(fsc_test PRIVATE 
    serow 
    pinocchio::pinocchio 
    nlohmann_json::nlohmann_json
    ${ZSTD_LIBRARIES}
    pthread
    dl
)

if (TARGET mcap::mcap)
    target_link_libraries(fsc_test PRIVATE mcap::mcap)
elseif (TARGET mcap)
    target_link_libraries(fsc_test PRIVATE mcap)
else()
    target_link_libraries(fsc_test PRIVATE ${MCAP_LIBRARY})
endif()
