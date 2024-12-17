# Project name
project(sorry::arena)

# Compiler flags
add_compile_options(-O3 -Wall)

# Source files
set(SRC_FILES
  arena.cpp
)

# Header files
set(INC_FILES)

add_executable(arena ${SRC_FILES} ${INC_FILES})

target_link_libraries(arena
  PRIVATE
  sorry::agent::mcts
  sorry::agent::rl
  sorry::common
  sorry::engine
)