cmake_minimum_required(VERSION 3.15)
project(serow LANGUAGES CXX)

# Dependencies
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(FlatBuffers REQUIRED)

# Find zstd before mcap_builder
# zstd::libzstd_shared / zstd::libzstd_static. Create alias if needed.
find_package(zstd QUIET)
if(zstd_FOUND AND NOT TARGET zstd::libzstd)
  if(TARGET zstd::libzstd_shared)
    add_library(zstd::libzstd ALIAS zstd::libzstd_shared)
  elseif(TARGET zstd::libzstd_static)
    add_library(zstd::libzstd ALIAS zstd::libzstd_static)
  endif()
endif()

include(FetchContent)
FetchContent_Declare(
  mcap_builder
  GIT_REPOSITORY https://github.com/olympus-robotics/mcap_builder.git
  GIT_TAG main
)

# Configure mcap_builder to build with PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(mcap_builder)

include_directories(${FLATBUFFERS_INCLUDE_DIRS})

# Option for Python code generation 
option(GENERATE_PYTHON_SCHEMAS "Generate Python code from FlatBuffers schemas" OFF)

# Create the generated directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)
message(STATUS "Source dir: ${CMAKE_SOURCE_DIR}")

# Find all .fbs files in the flatbuffer directory
file(GLOB FLATBUFFER_SCHEMAS ${CMAKE_SOURCE_DIR}/flatbuffer/*.fbs)

# Generate C++ code from Flatbuffers schema
set(FLATC_COMMANDS
    COMMAND flatc --cpp -o ${CMAKE_BINARY_DIR}/generated -I ${CMAKE_SOURCE_DIR}/flatbuffer ${FLATBUFFER_SCHEMAS}
    COMMAND flatc --schema -b -o ${CMAKE_BINARY_DIR}/generated -I ${CMAKE_SOURCE_DIR}/flatbuffer ${FLATBUFFER_SCHEMAS}
)

# Add Python code generation if enabled
if(GENERATE_PYTHON_SCHEMAS)
    list(APPEND FLATC_COMMANDS
        COMMAND flatc --python --gen-object-api --gen-all -o ${CMAKE_BINARY_DIR}/generated -I ${CMAKE_SOURCE_DIR}/flatbuffer ${FLATBUFFER_SCHEMAS}
    )
    message(STATUS "Python code generation enabled")
endif()

execute_process(
    ${FLATC_COMMANDS}
    RESULT_VARIABLE FLATC_RESULT
    OUTPUT_VARIABLE FLATC_OUTPUT
    ERROR_VARIABLE FLATC_ERROR
)

if(NOT FLATC_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate code from Flatbuffers schema: ${FLATC_ERROR}")
endif()

message(STATUS "Generated code from Flatbuffers schema: ${FLATC_OUTPUT}")

# Pass the source directory as a preprocessor definition
add_definitions(-DSCHEMAS_DIR="${CMAKE_BINARY_DIR}/generated/")

# Source files
set(serow_src
    Serow.cpp 
    State.cpp 
    ContactEKF.cpp 
    CoMEKF.cpp 
    LegOdometry.cpp 
    ButterworthLPF.cpp 
    Differentiator.cpp 
    DerivativeEstimator.cpp
    OutlierDetector.cpp
    LocalTerrainMapper.cpp
    NaiveLocalTerrainMapper.cpp
    ProprioceptionLogger.cpp
    ExteroceptionLogger.cpp
    ThreadPool.cpp
    MeasurementLogger.cpp
    Timer.cpp
    ForceTorqueMeasurementBuffer.cpp
)

# Header files
set(serow_headers
    common.hpp
    Serow.hpp 
    State.hpp 
    ContactEKF.hpp 
    CoMEKF.hpp 
    LegOdometry.hpp 
    ButterworthLPF.hpp 
    Differentiator.hpp 
    DerivativeEstimator.hpp
    ContactDetector.hpp
    lie.hpp
    Mahony.hpp
    Measurement.hpp
    MovingMedianFilter.hpp
    RobotKinematics.hpp
    OutlierDetector.hpp
    LocalTerrainMapper.hpp
    NaiveLocalTerrainMapper.hpp
    ProprioceptionLogger.hpp
    ExteroceptionLogger.hpp
    ThreadPool.hpp
    Schemas.hpp
    MeasurementLogger.hpp
    Timer.hpp
    ForceTorqueMeasurementBuffer.hpp
)

add_library(serow SHARED ${serow_headers} ${serow_src})
target_include_directories(serow PRIVATE ${CMAKE_BINARY_DIR}/generated)

target_link_libraries(serow
    PUBLIC mcap
    PRIVATE Eigen3::Eigen pinocchio::pinocchio nlohmann_json::nlohmann_json
)

set_target_properties(serow PROPERTIES LINKER_LANGUAGE CXX)

# Safe, portable release optimization
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(serow PRIVATE
        -O3
        -DNDEBUG
        -DEIGEN_VECTORIZE_SSE4_2
        -DEIGEN_UNALIGNED_VECTORIZE
        -DEIGEN_FAST_MATH
        -DEIGEN_NO_DEBUG
        -fno-strict-aliasing
        -fno-omit-frame-pointer
    )
endif()

# Install
install(FILES ${serow_headers} DESTINATION include/serow)
install(TARGETS serow DESTINATION lib/)
install(FILES serowConfig.cmake DESTINATION lib/cmake/serow)

# Profile-guided optimization option
option(ENABLE_PGO "Enable Profile Guided Optimization" OFF)
if(ENABLE_PGO)
    if(PGO_TRAINING)
        target_compile_options(serow PRIVATE -fprofile-generate)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate")
    else()
        target_compile_options(serow PRIVATE -fprofile-use)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-use")
    endif()
endif()
