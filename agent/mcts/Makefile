# Compiler
CC := g++
# Compiler flags
CFLAGS := -std=c++17 -Wall -O3

# Source files
SRC_FILES := $(wildcard *.cpp)
# Header files
INC_FILES := $(wildcard *.hpp) $(wildcard *.h)
# Object files
OBJ_FILES := $(SRC_FILES:.cpp=.o)

# Executable name
EXEC := main

all: $(EXEC)

# Build rule for the executable
$(EXEC): $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^

# Build rule for object files
%.o: %.cpp $(INC_FILES)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean rule
clean:
	rm -rf *.o $(EXEC)