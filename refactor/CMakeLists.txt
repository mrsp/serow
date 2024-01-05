cmake_minimum_required(VERSION 3.0.0)
project(serow_tests VERSION 0.1.0 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(pinocchio REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(serow_tests test.cpp Serow.cpp State.cpp ContactEKF.cpp CoMEKF.cpp LegOdometry.cpp ButterworthLPF.cpp Differentiator.cpp JointEstimator.cpp)
target_link_libraries(serow_tests PRIVATE Eigen3::Eigen pinocchio::pinocchio nlohmann_json::nlohmann_json GTest::gtest_main)
