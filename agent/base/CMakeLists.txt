# Project name
project(sorry::agent::base)

# Header files
set(AGENT_BASE_LIBRARY_INC_FILES
  baseAgent.hpp
)

# Build library
set(LIBRARY_NAME sorry_agent_base)

# Create a virtual include directory structure
set(BINARY_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
set(EXPORTED_INCLUDE_DIR ${BINARY_INCLUDE_DIR}/sorry/agent/base/)
file(MAKE_DIRECTORY ${EXPORTED_INCLUDE_DIR})

# Copy headers to the virtual include directory
foreach(header ${AGENT_BASE_LIBRARY_INC_FILES})
  configure_file(${header} ${EXPORTED_INCLUDE_DIR}/${header} COPYONLY)
endforeach()

# Specify include directories
add_library(${LIBRARY_NAME} INTERFACE)

add_library(${PROJECT_NAME} ALIAS ${LIBRARY_NAME})