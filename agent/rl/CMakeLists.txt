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
  actionMap.cpp
  common.cpp
  main.cpp
  trainingUtil.cpp
  trajectory.cpp
)

# =============================================================================

# Source files
set(REINFORCE_AGENT_LIBRARY_SRC_FILES
  actionMap.cpp
  common.cpp
  reinforceAgent.cpp
)

# Header files
set(REINFORCE_AGENT_LIBRARY_INC_FILES
  actionMap.hpp
  common.hpp
  reinforceAgent.hpp
)

# Build library
set(LIBRARY_NAME sorry_agent_rl)
add_library(${LIBRARY_NAME} ${REINFORCE_AGENT_LIBRARY_SRC_FILES} ${REINFORCE_AGENT_LIBRARY_INC_FILES})

# Create a virtual include directory structure
set(BINARY_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
set(EXPORTED_INCLUDE_DIR ${BINARY_INCLUDE_DIR}/sorry/agent/rl/)
file(MAKE_DIRECTORY ${EXPORTED_INCLUDE_DIR})

# Copy headers to the virtual include directory
foreach(header ${REINFORCE_AGENT_LIBRARY_INC_FILES})
  configure_file(${header} ${EXPORTED_INCLUDE_DIR}/${header} COPYONLY)
endforeach()

# Specify include directories
add_library(${LIBRARY_NAME}_headers INTERFACE)
target_include_directories(${LIBRARY_NAME}_headers INTERFACE ${BINARY_INCLUDE_DIR})

# Link the INTERFACE target with the main target
target_link_libraries(${LIBRARY_NAME} PUBLIC
  # sorry::common
  pybind11::embed
  ${LIBRARY_NAME}_headers
)

add_library(${PROJECT_NAME} ALIAS ${LIBRARY_NAME})

target_compile_definitions(${LIBRARY_NAME} PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# =============================================================================

add_executable(rl_main ${SRC_FILES})
add_executable(syntaxTest syntaxTest.cpp)

target_compile_definitions(rl_main PRIVATE SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(rl_main PRIVATE
  sorry::common
  sorry::engine
  pybind11::embed
  pybind11::module
  ${Python3_LIBRARIES}
)

target_link_libraries(syntaxTest PRIVATE
  sorry::common
  sorry::engine
  pybind11::embed
  pybind11::module
  ${Python3_LIBRARIES}
)