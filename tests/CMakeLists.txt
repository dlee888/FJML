cmake_minimum_required(VERSION 3.16)

project(Catch2Test)

set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -O3 -Wall -g --coverage -march=native")

find_package(Catch2 3 REQUIRED)
# These tests can use the Catch2-provided main
add_executable(tests tests.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain FJML)

