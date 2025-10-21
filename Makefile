# DESCRIPTION
# - Makefile for CARTS
# - Dependencies: ARTS, LLVM, Polygeist
# - Required: CMake, Ninja, Clang, Clang++

# Build verbosity control (set VERBOSE=1 to enable verbose output)
VERBOSE ?= 0
NINJA_FLAGS := $(if $(filter 1,$(VERBOSE)),-v,)

# Source Directories
CARTS_DIR = ${shell pwd}
ARTS_DIR ?= ${CARTS_DIR}/external/arts
POLYGEIST_DIR ?= ${CARTS_DIR}/external/Polygeist
LLVM_DIR ?= ${POLYGEIST_DIR}/llvm-project

# Install Directories
INSTALL_DIR ?= ${CARTS_DIR}/.install
CARTS_INSTALL_DIR ?= ${INSTALL_DIR}/carts
ARTS_INSTALL_DIR ?=$(INSTALL_DIR)/arts
LLVM_INSTALL_DIR ?=$(INSTALL_DIR)/llvm
POLYGEIST_INSTALL_DIR ?=$(INSTALL_DIR)/polygeist

CARTS_LINKER_PATH ?= ${LLVM_INSTALL_DIR}/bin/ld.lld

# Build Directories
CARTS_BUILD_DIR=build
ARTS_BUILD_DIR =$(ARTS_DIR)/build
POLYGEIST_BUILD_DIR =$(POLYGEIST_DIR)/build
LLVM_BUILD_DIR =$(LLVM_DIR)/build

# Targets
all: install

# Polygeist
polygeist-download:
	@if [ ! -d "$(POLYGEIST_DIR)/.git" ]; then \
		echo "Initializing Polygeist submodule..."; \
		git submodule update --init --recursive external/Polygeist; \
	else \
		echo "Polygeist submodule already initialized."; \
	fi 
polygeist:
	echo "Building Polygeist..."; \
	mkdir -p $(POLYGEIST_BUILD_DIR); \
	mkdir -p $(POLYGEIST_INSTALL_DIR); \
	cmake -B $(POLYGEIST_BUILD_DIR) \
		-S $(POLYGEIST_DIR) -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(POLYGEIST_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang \
		-DLLVM_EXTERNAL_LIT="$(LLVM_BUILD_DIR)/bin/llvm-lit" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DPOLYGEIST_USE_LINKER="$(CARTS_LINKER_PATH)"
	ninja -C $(POLYGEIST_BUILD_DIR) install
polygeist-clean:
	rm -f -r $(POLYGEIST_BUILD_DIR)
	rm -f -r $(POLYGEIST_INSTALL_DIR)

# LLVM
llvm:
	echo "Building LLVM..."; \
	mkdir -p $(LLVM_BUILD_DIR); \
	mkdir -p $(LLVM_INSTALL_DIR); \
	cmake -B $(LLVM_BUILD_DIR) \
		-S $(LLVM_DIR)/llvm -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DLLVM_ENABLE_PROJECTS='mlir;clang;lld;openmp' \
		-DLLVM_ENABLE_RUNTIMES='libcxx;libcxxabi;libunwind' \
		-DLLVM_TARGETS_TO_BUILD='host' \
		-DLLVM_OPTIMIZED_TABLEGEN=ON \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DLLVM_BUILD_TOOLS=ON \
		-DLLVM_INCLUDE_TOOLS=ON \
		-DLLVM_INCLUDE_EXAMPLES=OFF \
		-DLLVM_INCLUDE_TESTS=OFF \
		-DLLVM_BUILD_TESTS=OFF \
		-DLLVM_INSTALL_UTILS=ON \
		-DLLVM_INCLUDE_BENCHMARKS=OFF \
		-DLLVM_INCLUDE_UTILS=ON \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON; \
	ninja -C $(LLVM_BUILD_DIR) install; 
llvm-clean:
	rm -rf $(LLVM_BUILD_DIR)
	rm -f -r $(LLVM_INSTALL_DIR)

# ARTS
ARTS_BUILD_TYPE ?= Release
# Introspection is disabled by default for performance
# Use ARTS_USE_COUNTERS=ON and ARTS_USE_METRICS=ON to enable introspection
ARTS_USE_COUNTERS ?= OFF
ARTS_USE_METRICS ?= OFF
# Logging levels (disabled by default for performance)
ARTS_INFO_ENABLED ?= OFF
ARTS_DEBUG_ENABLED ?= OFF

arts-download:
	@if [ ! -d "$(ARTS_DIR)/.git" ]; then \
		echo "Initializing ARTS submodule..."; \
		git submodule update --init --recursive external/arts; \
	else \
		echo "ARTS submodule already initialized."; \
	fi
arts:
	echo "Building ARTS (build_type=$(ARTS_BUILD_TYPE), counters=$(ARTS_USE_COUNTERS), metrics=$(ARTS_USE_METRICS), info=$(ARTS_INFO_ENABLED), debug=$(ARTS_DEBUG_ENABLED))..."; \
	mkdir -p $(ARTS_BUILD_DIR); \
	mkdir -p $(ARTS_INSTALL_DIR); \
	cmake -B $(ARTS_BUILD_DIR) -S $(ARTS_DIR) \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_BUILD_TYPE=$(ARTS_BUILD_TYPE) \
		-DUSE_GPU=OFF \
		-DUSE_SMART_DB=ON \
		-DUSE_COUNTERS=$(ARTS_USE_COUNTERS) \
		-DUSE_METRICS=$(ARTS_USE_METRICS) \
		-DARTS_INFO_ENABLED=$(ARTS_INFO_ENABLED) \
		-DARTS_DEBUG_ENABLED=$(ARTS_DEBUG_ENABLED) \
		-DCMAKE_INSTALL_PREFIX=$(ARTS_INSTALL_DIR) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DARTS_USE_LINKER="$(CARTS_LINKER_PATH)"; 
	make -C $(ARTS_BUILD_DIR) install -j;
arts-clean:
	rm -f -r $(ARTS_BUILD_DIR)
	rm -f -r $(ARTS_INSTALL_DIR)
	# rm -f -r $(ARTS_DIR)

# CARTS
build:
	mkdir -p $(CARTS_BUILD_DIR)
	mkdir -p $(CARTS_INSTALL_DIR)
	cmake -B $(CARTS_BUILD_DIR) \
		-S $(CARTS_DIR) -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(CARTS_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang \
		-DCMAKE_CXX_COMPILER=$(LLVM_INSTALL_DIR)/bin/clang++ \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang \
		-DPOLYGEIST_BUILD_DIR=$(POLYGEIST_BUILD_DIR) \
		-DPOLYGEIST_DIR=$(POLYGEIST_DIR) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DCARTS_USE_LINKER="$(CARTS_LINKER_PATH)" 
		ninja $(NINJA_FLAGS) -C $(CARTS_BUILD_DIR) install

install: arts-download polygeist-download llvm arts polygeist build

uninstall:
	cat $(BUILD_DIR)/install_manifest.txt | xargs rm -f -r
	rm -rf $(BUILD_DIR)

fulluninstall: uninstall arts-clean
	rm -f .arts

clean:
	make -C $(BUILD_DIR) clean -j
	rm -rf $(BUILD_DIR)

.PHONY: all build install postinstall uninstall fulluninstall clean test llvm-runtimes arts-download polygeist-download
