find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(nlohmann_json REQUIRED)

# Option for ONNX Runtime support
option(USE_ONNX "Enable ONNX Runtime support" OFF)
if(USE_ONNX)
    # Find ONNX Runtime
    find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS /usr/local/onnxruntime/include
    NO_DEFAULT_PATH
    )

    find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS /usr/local/onnxruntime/lib
    NO_DEFAULT_PATH
    )

    # Check if ONNX Runtime was found
    if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Please ensure it is installed.")
    endif()
    add_definitions(-DUSE_ONNX)

    # Add ONNX Runtime library path to RPATH
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/onnxruntime/lib")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
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

# Find FlatBuffers
find_package(FlatBuffers REQUIRED)
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
    LocalTerrainMapper.cpp
    NaiveLocalTerrainMapper.cpp
    ProprioceptionLogger.cpp
    ExteroceptionLogger.cpp
    ThreadPool.cpp
    MeasurementLogger.cpp)

if(USE_ONNX)
    set(serow_src ${serow_src} ONNXInference.cpp)
endif()

set(serow_headers ${serow_headers} 
    common.hpp
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
    LocalTerrainMapper.hpp
    NaiveLocalTerrainMapper.hpp
    ProprioceptionLogger.hpp
    ExteroceptionLogger.hpp
    ThreadPool.hpp
    Schemas.hpp
    MeasurementLogger.hpp)

if(USE_ONNX)
    set(serow_headers ${serow_headers} ONNXInference.hpp)
endif()

add_library(serow SHARED ${serow_headers} ${serow_src})
target_include_directories(serow PRIVATE ${CMAKE_BINARY_DIR}/generated)
if (USE_ONNX)
    target_include_directories(serow PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(serow PRIVATE ${ONNXRUNTIME_LIBRARY})
    target_link_directories(serow PRIVATE /usr/local/onnxruntime/lib)
endif()

target_link_libraries(serow PUBLIC mcap 
                            PRIVATE Eigen3::Eigen pinocchio::pinocchio nlohmann_json::nlohmann_json)

set_target_properties(serow PROPERTIES 
    LINKER_LANGUAGE CXX
    INSTALL_RPATH "/usr/local/onnxruntime/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(FILES ${serow_headers} DESTINATION include/serow)
install(TARGETS serow DESTINATION lib/)
install(FILES serowConfig.cmake DESTINATION lib/cmake/serow)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Eigen-specific optimizations
    add_compile_options(-DEIGEN_VECTORIZE_SSE4_2)
    add_compile_options(-DEIGEN_UNALIGNED_VECTORIZE)
    add_compile_options(-DEIGEN_FAST_MATH)
    add_compile_options(-DEIGEN_MAX_ALIGN_BYTES=64)
    add_compile_options(-DEIGEN_MAX_STATIC_ALIGN_BYTES=64)
    
    # Matrix operation optimizations
    add_compile_options(-ffast-math)
    add_compile_options(-mavx2)
    add_compile_options(-mfma)
    add_compile_options(-funroll-loops)
    add_compile_options(-funroll-all-loops)
    
    # Memory and cache optimizations
    add_compile_options(-fprefetch-loop-arrays)
    add_compile_options(-foptimize-sibling-calls)
    add_compile_options(-foptimize-strlen)
    
    # Function inlining
    add_compile_options(-finline-functions)
    add_compile_options(-finline-limit=1000)
    
    # Link-time optimization
    add_compile_options(-flto)
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -flto")
    
    # Thread pool optimizations
    add_compile_options(-pthread)
    add_compile_options(-fopenmp)
endif()

# Profile-guided optimization option
option(ENABLE_PGO "Enable Profile Guided Optimization" OFF)
if(ENABLE_PGO)
    if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        if(PGO_TRAINING)
            add_compile_options(-fprofile-generate)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-generate")
        else()
            add_compile_options(-fprofile-use)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-use")
        endif()
    endif()
endif()
