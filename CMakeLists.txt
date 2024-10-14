# Minimum version of CMake required
cmake_minimum_required(VERSION 3.20)

# Project name
project(sorry)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(agent)
add_subdirectory(common)
add_subdirectory(engine)
add_subdirectory(ui)