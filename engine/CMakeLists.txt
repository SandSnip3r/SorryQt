# Project name
project(sorry::engine)

# Compiler flags
add_compile_options(-O3 -Wall)

# Source files
set(SRC_FILES
  action.cpp
  card.cpp
  deck.cpp
  playerColor.cpp
  sorry.cpp
)

# Header files
set(INC_FILES
  action.hpp
  card.hpp
  deck.hpp
  playerColor.hpp
  sorry.hpp
)

# Build library
set(LIBRARY_NAME sorry_engine)
add_library(${LIBRARY_NAME} ${SRC_FILES} ${INC_FILES})

# Create a virtual include directory structure
set(BINARY_INCLUDE_DIR ${CMAKE_BINARY_DIR}/include)
set(EXPORTED_INCLUDE_DIR ${BINARY_INCLUDE_DIR}/sorry/engine)
file(MAKE_DIRECTORY ${EXPORTED_INCLUDE_DIR})

# Copy headers to the virtual include directory
foreach(header ${INC_FILES})
  configure_file(${header} ${EXPORTED_INCLUDE_DIR}/${header} COPYONLY)
endforeach()

# Specify include directories
add_library(${LIBRARY_NAME}_headers INTERFACE)
target_include_directories(${LIBRARY_NAME}_headers INTERFACE ${BINARY_INCLUDE_DIR})

# Link the INTERFACE target with the main target
target_link_libraries(${LIBRARY_NAME} PUBLIC ${LIBRARY_NAME}_headers)

add_library(${PROJECT_NAME} ALIAS ${LIBRARY_NAME})