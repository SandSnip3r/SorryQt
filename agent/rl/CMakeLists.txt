# Project name
project(sorry::agent::rl)

# Compiler flags
add_compile_options(-O3 -Wall)

# Find Python interpreter and development headers
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories for Python
include_directories(${Python3_INCLUDE_DIRS})

# Source files
set(SRC_FILES
  main.cpp
  actionMap.cpp
  jaxModel.cpp
  trajectory.cpp
)

# Build library
add_executable(rl_main ${SRC_FILES})
add_executable(syntaxTest syntaxTest.cpp)

target_compile_definitions(rl_main PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(rl_main PRIVATE sorry::common sorry::engine pybind11::embed pybind11::module ${Python3_LIBRARIES})

target_link_libraries(syntaxTest PRIVATE sorry::common sorry::engine pybind11::embed pybind11::module ${Python3_LIBRARIES})