# NeuroGen Modular Brain Architecture Makefile
# Compiles CUDA and C++ code for the modular neural language model

# ============================================================================
# Configuration
# ============================================================================

# Compilers
HOST_COMPILER := /usr/bin/clang++
NVCC := nvcc
CXX := $(HOST_COMPILER)

# Auto-detect CUDA installation
NVCC_PATH := $(shell which $(NVCC) 2>/dev/null)

# Determine CUDA_HOME (strip /bin/nvcc to get base directory)
ifndef CUDA_HOME
  ifneq ($(strip $(NVCC_PATH)),)
    # Get directory of nvcc, then parent directory (e.g., /opt/cuda/bin/nvcc -> /opt/cuda)
    CUDA_BIN_DIR := $(dir $(NVCC_PATH))
    CUDA_HOME := $(patsubst %/,%,$(dir $(patsubst %/,%,$(CUDA_BIN_DIR))))
  else
    # Check /opt/cuda (common on Arch Linux)
    ifneq ($(wildcard /opt/cuda),)
      CUDA_HOME := /opt/cuda
    else
      CUDA_HOME := /usr/local/cuda
    endif
  endif
endif

# Directories
SRC_DIR := src
INC_DIR := include
BUILD_DIR := build
BIN_DIR := bin

# Python configuration - Use conda environment Python 3.9
CONDA_ENV_PATH := $(HOME)/anaconda3/envs/bio_trading_network
PYTHON_VERSION := 3.9
PYTHON_INCLUDES := -I$(CONDA_ENV_PATH)/include/python$(PYTHON_VERSION)

# Include paths
INCLUDES := -I$(INC_DIR) -I$(SRC_DIR) -I$(CUDA_HOME)/include -Iinclude/pybind11_repo/include $(PYTHON_INCLUDES)

# Compiler flags
NVCC_FLAGS := -std=c++17 -O3 -arch=sm_75 -use_fast_math --compiler-options -fPIC -ccbin $(HOST_COMPILER)
CXX_FLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra

# Linker flags
LDFLAGS := -L$(CUDA_HOME)/lib64 -lcudart -lcublas -lcurand -lpthread

# Shared library flags
SHARED_FLAGS := -shared

# Target executable and shared library
TARGET := $(BIN_DIR)/neurogen_modular_brain
SHARED_LIB := $(BIN_DIR)/libneurogen.so
# PYTHON_INCLUDE removed as we include it in INCLUDES directly


# ============================================================================
# Source Files
# ============================================================================

# CUDA kernel source files
CUDA_SOURCES := $(wildcard $(SRC_DIR)/engine/*.cu) $(wildcard $(SRC_DIR)/modules/*.cu)

# C++ source files
CPP_ENGINE_SOURCES := $(wildcard $(SRC_DIR)/engine/*.cpp)
CPP_MODULE_SOURCES := $(wildcard $(SRC_DIR)/modules/*.cpp)
CPP_INTERFACE_SOURCES := $(wildcard $(SRC_DIR)/interfaces/*.cpp)
CPP_PERSISTENCE_SOURCES := $(wildcard $(SRC_DIR)/persistence/*.cpp)
CPP_MAIN_SOURCE := $(SRC_DIR)/main.cpp

# All C++ sources (without main for library)
CPP_SOURCES := $(CPP_ENGINE_SOURCES) $(CPP_MODULE_SOURCES) $(CPP_INTERFACE_SOURCES) $(CPP_PERSISTENCE_SOURCES) $(CPP_MAIN_SOURCE)
CPP_LIB_SOURCES := $(CPP_ENGINE_SOURCES) $(CPP_MODULE_SOURCES) $(CPP_INTERFACE_SOURCES) $(CPP_PERSISTENCE_SOURCES)

# Python binding source
PYTHON_BINDING_SOURCE := $(SRC_DIR)/python/python_binding.cpp

# Object files
CUDA_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES))
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SOURCES))
CPP_LIB_OBJECTS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CUDA_SOURCES)) $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_LIB_SOURCES))

# All objects
OBJECTS := $(CUDA_OBJECTS) $(CPP_OBJECTS)

# ============================================================================
# Build Rules
# ============================================================================

.PHONY: all clean directories info test lib python-lib

# Default target - build both executable and library
all: directories $(TARGET) $(SHARED_LIB)

# Build only the shared library
lib: directories $(SHARED_LIB)

# Alias for shared library
python-lib: lib

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)/engine
	@mkdir -p $(BUILD_DIR)/modules
	@mkdir -p $(BUILD_DIR)/interfaces
	@mkdir -p $(BUILD_DIR)/persistence
	@mkdir -p $(BUILD_DIR)/python
	@mkdir -p $(BIN_DIR)
	@mkdir -p checkpoints

# Link final executable
$(TARGET): $(OBJECTS)
	@echo "🔗 Linking executable: $(TARGET)"
	$(NVCC) $(NVCC_FLAGS) $(OBJECTS) -o $(TARGET) $(LDFLAGS)
	@echo "✓ Build complete: $(TARGET)"

# Build shared library for Python
$(SHARED_LIB): $(CPP_LIB_OBJECTS) $(BUILD_DIR)/python/python_binding.o
	@echo "🔗 Linking shared library: $(SHARED_LIB)"
	$(NVCC) $(NVCC_FLAGS) $(SHARED_FLAGS) $(CPP_LIB_OBJECTS) $(BUILD_DIR)/python/python_binding.o -o $(SHARED_LIB) $(LDFLAGS)
	@echo "✓ Shared library complete: $(SHARED_LIB)"

# Compile CUDA source files
$(BUILD_DIR)/engine/%.o: $(SRC_DIR)/engine/%.cu
	@echo "🔨 Compiling CUDA: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA module source files
$(BUILD_DIR)/modules/%.o: $(SRC_DIR)/modules/%.cu
	@echo "🔨 Compiling CUDA module: $<"
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ engine source files
$(BUILD_DIR)/engine/%.o: $(SRC_DIR)/engine/%.cpp
	@echo "🔨 Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ module source files
$(BUILD_DIR)/modules/%.o: $(SRC_DIR)/modules/%.cpp
	@echo "🔨 Compiling C++ module: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ interface source files
$(BUILD_DIR)/interfaces/%.o: $(SRC_DIR)/interfaces/%.cpp
	@echo "🔨 Compiling C++ interface: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile C++ persistence source files
$(BUILD_DIR)/persistence/%.o: $(SRC_DIR)/persistence/%.cpp
	@echo "🧩 Compiling persistence: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile main.cpp
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp
	@echo "🔨 Compiling main: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $< -o $@

# Compile Python binding
$(BUILD_DIR)/python/%.o: $(SRC_DIR)/python/%.cpp
	@echo "🔨 Compiling Python binding: $<"
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(PYTHON_INCLUDE) -c $< -o $@

# ============================================================================
# Utility Targets
# ============================================================================

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
	@echo "✓ Clean complete"

# Display build information
info:
	@echo "=========================================="
	@echo "NeuroGen Modular Brain Build Info"
	@echo "=========================================="
	@echo "NVCC:         $(NVCC)"
	@echo "NVCC_PATH:    $(NVCC_PATH)"
	@echo "CUDA_HOME:    $(CUDA_HOME)"
	@echo "CXX:          $(CXX)"
	@echo "Target:       $(TARGET)"
	@echo "CUDA Sources: $(words $(CUDA_SOURCES)) files"
	@echo "C++ Sources:  $(words $(CPP_SOURCES)) files"
	@echo "INCLUDES:     $(INCLUDES)"
	@echo "=========================================="

# Run in training mode
train: $(TARGET)
	@echo "🎓 Running training mode..."
	./$(TARGET) train

# Run in generation mode
generate: $(TARGET)
	@echo "💭 Running generation mode..."
	./$(TARGET) generate

# Run demo mode
demo: $(TARGET)
	@echo "🎬 Running demo mode..."
	./$(TARGET) demo

# Quick test build (compile only, no link)
test-compile: directories
	@echo "🧪 Test compiling modules..."
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $(SRC_DIR)/modules/CorticalModule.cu -o $(BUILD_DIR)/test_module.o
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(SRC_DIR)/modules/InterModuleConnection.cpp -o $(BUILD_DIR)/test_connection.o
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c $(SRC_DIR)/modules/BrainOrchestrator.cpp -o $(BUILD_DIR)/test_brain.o
	@echo "✓ Test compilation successful"

# ============================================================================
# Help
# ============================================================================

help:
	@echo "NeuroGen Modular Brain - Makefile Help"
	@echo ""
	@echo "Available targets:"
	@echo "  make               - Build executable and shared library"
	@echo "  make lib           - Build only the shared library (.so)"
	@echo "  make python-lib    - Alias for 'make lib'"
	@echo "  make clean         - Remove all build artifacts"
	@echo "  make info          - Display build configuration"
	@echo "  make train         - Build and run training mode"
	@echo "  make generate      - Build and run generation mode"
	@echo "  make demo          - Build and run demo mode"
	@echo "  make test-compile  - Test compile modules only"
	@echo "  make help          - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make clean && make        # Clean build (both targets)"
	@echo "  make -j8                  # Parallel build with 8 jobs"
	@echo "  make lib                  # Build only Python library"
	@echo "  make train                # Train the model"
	@echo ""

