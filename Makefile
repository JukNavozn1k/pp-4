CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall
TARGET = strassen_mpi
SRC_DIR = src
BUILD_DIR = build

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(SRC_DIR)/main.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(BUILD_DIR)/$(TARGET)
	mpirun -np 8 $(BUILD_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all run clean
