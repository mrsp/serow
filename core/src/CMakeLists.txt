find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(nlohmann_json REQUIRED)

include(FetchContent)
FetchContent_Declare(
  mcap_builder
  GIT_REPOSITORY https://github.com/olympus-robotics/mcap_builder.git
  GIT_TAG main
)

# Configure mcap_builder to build with PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(mcap_builder)

# Find FlatBuffers
find_package(FlatBuffers REQUIRED)
include_directories(${FLATBUFFERS_INCLUDE_DIRS})

# Create the generated directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)
message(STATUS "Source dir: ${CMAKE_SOURCE_DIR}")

# Find all .fbs files in the flatbuffer directory
file(GLOB FLATBUFFER_SCHEMAS ${CMAKE_SOURCE_DIR}/flatbuffer/*.fbs)

# Generate C++ code from Flatbuffers schema
execute_process(
    COMMAND flatc --cpp -o ${CMAKE_BINARY_DIR}/generated -I ${CMAKE_SOURCE_DIR}/flatbuffer ${FLATBUFFER_SCHEMAS}
    COMMAND flatc --schema -b -o ${CMAKE_BINARY_DIR}/generated -I ${CMAKE_SOURCE_DIR}/flatbuffer ${FLATBUFFER_SCHEMAS}

    
    RESULT_VARIABLE FLATC_RESULT
    OUTPUT_VARIABLE FLATC_OUTPUT
    ERROR_VARIABLE FLATC_ERROR
)

if(NOT FLATC_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate C++ code from Flatbuffers schema: ${FLATC_ERROR}")
endif()

message(STATUS "Generated C++ code from Flatbuffers schema: ${FLATC_OUTPUT}")

# Pass the source directory as a preprocessor definition
add_definitions(-DSCHEMAS_DIR="${CMAKE_BINARY_DIR}/generated/")

set(serow_src ${serow_src} 
    Serow.cpp 
    State.cpp 
    BaseEKF.cpp
    ContactEKF.cpp 
    CoMEKF.cpp 
    LegOdometry.cpp 
    ButterworthLPF.cpp 
    Differentiator.cpp 
    DerivativeEstimator.cpp
    OutlierDetector.cpp
    TerrainElevation.cpp
    NaiveTerrainElevation.cpp
    ProprioceptionLogger.cpp
    ExteroceptionLogger.cpp
    ThreadPool.cpp)

set(serow_headers ${serow_headers} 
    Serow.hpp 
    State.hpp 
    BaseEKF.hpp
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
    TerrainElevation.hpp
    NaiveTerrainElevation.hpp
    ProprioceptionLogger.hpp
    ExteroceptionLogger.hpp
    ThreadPool.hpp
    Schemas.hpp)


add_library(serow SHARED ${serow_headers} ${serow_src})
target_include_directories(serow PRIVATE ${CMAKE_BINARY_DIR}/generated)
target_link_libraries(serow PUBLIC mcap 
                            PRIVATE Eigen3::Eigen pinocchio::pinocchio nlohmann_json::nlohmann_json)

install(FILES ${serow_headers} DESTINATION include/serow)
install(TARGETS serow DESTINATION lib/)
install(FILES serowConfig.cmake DESTINATION lib/cmake/serow)
