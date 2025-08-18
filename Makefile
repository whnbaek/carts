# DESCRIPTION
# - Makefile for CARTS
# - Dependencies: ARTS, LLVM, Polygeist
# - Required: CMake, Ninja, Clang, Clang++


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

# Build Directories
CARTS_BUILD_DIR=build
ARTS_BUILD_DIR =$(ARTS_DIR)/build
POLYGEIST_BUILD_DIR =$(POLYGEIST_DIR)/build
LLVM_BUILD_DIR =$(LLVM_DIR)/build

# Targets
# all: hooks postinstall

installdeps: arts
	@echo "Installing dependencies"
	@echo "Installing LLVM"
	@make llvm
	@echo "Installing Polygeist"
	@make polygeist

# Enable environment
enable:
	echo "export ARTS_PATH=$(ARTS_INSTALL_DIR)" > enable
	echo "export PATH=$(CARTS_INSTALL_DIR)/bin:$(POLYGEIST_INSTALL_DIR)/bin:$(LLVM_INSTALL_DIR)/bin:\$$PATH" >> enable
	echo "export LD_LIBRARY_PATH=$(CARTS_INSTALL_DIR)/lib:$(POLYGEIST_INSTALL_DIR)/lib:$(ARTS_INSTALL_DIR)/lib:$(LLVM_INSTALL_DIR)/lib:\$$LD_LIBRARY_PATH" >> enable

# Polygeist
polygeist-download:
	mkdir -p $(POLYGEIST_DIR)
	git clone --branch carts --recursive https://github.com/randreshg/Polygeist.git $(POLYGEIST_DIR) 
polygeist:
	mkdir -p $(POLYGEIST_BUILD_DIR)
	mkdir -p $(POLYGEIST_INSTALL_DIR)
	cmake -B $(POLYGEIST_BUILD_DIR) \
		-S $(POLYGEIST_DIR) -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(POLYGEIST_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang \
		-DLLVM_EXTERNAL_LIT="$(LLVM_BUILD_DIR)/bin/llvm-lit" \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
	ninja -C $(POLYGEIST_BUILD_DIR) install
polygeist-clean:
	rm -f -r $(POLYGEIST_BUILD_DIR)
	rm -f -r $(POLYGEIST_INSTALL_DIR)

# LLVM
llvm:
	mkdir -p $(LLVM_BUILD_DIR)
	mkdir -p $(LLVM_INSTALL_DIR)
	cmake -B $(LLVM_BUILD_DIR) \
		-S $(LLVM_DIR)/llvm -G Ninja \
		-DCMAKE_INSTALL_PREFIX=$(LLVM_INSTALL_DIR) \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DLLVM_ENABLE_PROJECTS='mlir;clang' \
		-DLLVM_ENABLE_RUNTIMES='openmp' \
		-DLLVM_OPTIMIZED_TABLEGEN=ON \
		-DLLVM_TARGETS_TO_BUILD="host" \
		-DLLVM_ENABLE_ASSERTIONS=ON \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
	ninja -C $(LLVM_BUILD_DIR) install
llvm-clean:
	rm -rf $(LLVM_BUILD_DIR)
	rm -f -r $(LLVM_INSTALL_DIR)

# ARTS
arts-download:
	mkdir -p $(ARTS_DIR)
	git clone --recursive https://github.com/randreshg/ARTS.git $(ARTS_DIR)
arts:
	mkdir -p $(ARTS_BUILD_DIR)
	mkdir -p $(ARTS_INSTALL_DIR)
	cmake -B $(ARTS_BUILD_DIR) -S $(ARTS_DIR) \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_BUILD_TYPE=Release \
		-DSMART_DB=ON \
		-DCMAKE_INSTALL_PREFIX=$(ARTS_INSTALL_DIR) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
	make -C $(ARTS_BUILD_DIR) install -j
arts-debug:
	mkdir -p $(ARTS_BUILD_DIR)
	mkdir -p $(ARTS_INSTALL_DIR)
	cmake -B $(ARTS_BUILD_DIR) -S $(ARTS_DIR) \
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DCMAKE_BUILD_TYPE=Debug \
		-DSMART_DB=ON \
		-DCMAKE_INSTALL_PREFIX=$(ARTS_INSTALL_DIR) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
	make -C $(ARTS_BUILD_DIR) install -j
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
		-DCMAKE_C_COMPILER=clang \
		-DCMAKE_CXX_COMPILER=clang++ \
		-DMLIR_DIR=$(LLVM_BUILD_DIR)/lib/cmake/mlir \
		-DClang_DIR=$(LLVM_BUILD_DIR)/lib/cmake/clang \
		-DPOLYGEIST_BUILD_DIR=$(POLYGEIST_BUILD_DIR) \
		-DPOLYGEIST_DIR=$(POLYGEIST_DIR) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON 
		ninja -C $(CARTS_BUILD_DIR) install

install: build
	make -C $(BUILD_DIR) install -j

uninstall:
	-cat $(BUILD_DIR)/install_manifest.txt | xargs rm -f -r
	rm -f enable
	rm -rf $(BUILD_DIR)

fulluninstall: uninstall arts-clean
	rm -f .arts

clean:
	make -C $(BUILD_DIR) clean -j
	rm -rf $(BUILD_DIR)

.PHONY: all build install postinstall uninstall fulluninstall  clean test
