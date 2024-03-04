cmake_minimum_required(VERSION 3.0.0)
project(Serow_tests VERSION 1.0.0 LANGUAGES CXX)
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
find_package(Serow REQUIRED)

add_executable(nao_test nao_test.cpp)
target_link_libraries(nao_test PRIVATE serow pinocchio::pinocchio GTest::gtest_main)

add_executable(go2_test go2_test.cpp)
target_link_libraries(go2_test PRIVATE serow pinocchio::pinocchio GTest::gtest_main)
