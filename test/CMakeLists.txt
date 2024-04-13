cmake_minimum_required(VERSION 3.0.0)
project(serow_tests VERSION 1.0.0 LANGUAGES CXX)

# Default to C++20
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)

FetchContent_MakeAvailable(googletest)
add_library(GTest::GTest INTERFACE IMPORTED)
target_link_libraries(GTest::GTest INTERFACE gtest_main)

find_package(pinocchio REQUIRED)
find_package(serow REQUIRED)

add_executable(nao_test nao_test.cpp)
target_link_libraries(nao_test PRIVATE serow pinocchio::pinocchio GTest::gtest_main)

add_executable(go2_test go2_test.cpp)
target_link_libraries(go2_test PRIVATE serow pinocchio::pinocchio GTest::gtest_main)

add_executable(go1_test go1_test.cpp)
target_link_libraries(go1_test PRIVATE serow pinocchio::pinocchio GTest::gtest_main)