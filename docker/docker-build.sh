#!/bin/bash

# Docker Build Script for CARTS Multi-Node Testing
# This script builds the base Docker image with all dependencies installed
# Run this ONCE to create the base image with CARTS pre-built

set -e

# Source common print functions
source "$(dirname "$0")/../tools/config/carts-print.sh"

carts_header "CARTS Docker Build"
carts_info "Building CARTS Docker base image with all dependencies"

# Change to docker directory
cd "$(dirname "$0")"

# Build the base Docker image with dependencies
carts_step "Building base Docker image (arts-node-base)"
docker build -t arts-node-base .

# Start builder container with access to all system threads
carts_step "Starting builder container with full CPU access"
docker rm -f arts-node-builder >/dev/null 2>&1 || true

# Detect CPU cores 
DOCKER_CPUS=$(docker run --rm ubuntu:22.04 nproc 2>/dev/null || echo "2")
docker run -d --name arts-node-builder --hostname arts-node-builder --network bridge \
    --cpus="$DOCKER_CPUS" arts-node-base

# Wait for builder to be ready
carts_info "Waiting for builder to be ready"
sleep 5

# Clone and build CARTS in builder
carts_step "Cloning and building CARTS in builder (using $DOCKER_CPUS threads)"
GIT_URL=${GIT_URL:-https://github.com/randreshg/carts.git}
GIT_BRANCH=${GIT_BRANCH:-mlir}

docker exec arts-node-builder bash -c "\
    set -e; \
    echo 'Setting up CARTS workspace...'; \
    rm -rf /opt/carts && mkdir -p /opt/carts; \
    git config --global init.defaultBranch main; \
    git config --global pull.rebase false; \
    git config --global core.preloadindex true; \
    git config --global core.fscache true; \
    git config --global gc.auto 256; \
    git clone --branch $GIT_BRANCH --depth 1 --single-branch --no-tags --progress $GIT_URL /opt/carts; \
    cd /opt/carts; \
    export MAKEFLAGS='-j$DOCKER_CPUS'; \
    export CMAKE_BUILD_PARALLEL_LEVEL=$DOCKER_CPUS; \
    python3 -u tools/setup/carts-setup.py; \
    echo 'export PATH=/opt/carts/tools:\$PATH' >> /root/.bashrc; \
    echo 'export PATH=/opt/carts/tools:\$PATH' >> /root/.zshrc 2>/dev/null || true \
"

# Commit builder as the final reusable image
carts_step "Committing builder to arts-node:built image"
docker commit arts-node-builder arts-node:built >/dev/null

# Clean up builder container
carts_step "Cleaning up builder container"
docker stop arts-node-builder >/dev/null
docker rm arts-node-builder >/dev/null

carts_header "Build Complete"
carts_success "Created images:"
carts_info "  - arts-node-base: Base image with dependencies only"
carts_info "  - arts-node:built: Complete image with CARTS pre-built"

carts_next_steps \
    "Run './docker-run.sh' to start multi-node containers" \
    "Run './docker-update.sh' to pull latest changes and rebuild"

carts_info "To verify the build:"
carts_info "  docker images | grep arts-node"
