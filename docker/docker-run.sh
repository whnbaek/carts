#!/bin/bash

# Docker Run Script for CARTS Multi-Node Testing
# This script loads the pre-built image and optionally runs ARTS examples across multiple containers

set -e

# Source common print functions
source "$(dirname "$0")/../tools/config/carts-print.sh"

# Parse arguments
EXAMPLE_FILE=""
BUILD_ARGS=""
RUNTIME_ARGS=""

if [[ $# -gt 0 ]]; then
    EXAMPLE_FILE="$1"
    shift

    # Parse remaining arguments, using -- as separator between build and runtime args
    while [[ $# -gt 0 ]]; do
        if [[ "$1" == "--" ]]; then
            shift
            # Everything after -- goes to runtime args
            RUNTIME_ARGS="$*"
            break
        else
            # Everything before -- goes to build args
            BUILD_ARGS="$BUILD_ARGS $1"
            shift
        fi
    done
fi

carts_header "CARTS Multi-Node Containers"
carts_info "Starting CARTS multi-node containers"

# Change to docker directory
cd "$(dirname "$0")"

# Check if the built image exists
if ! docker images | grep -q "arts-node.*built"; then
    carts_error "arts-node:built image not found!"
    carts_warning "Please run './docker-build.sh' first to build the base image."
    exit 1
fi

# Clean up any existing containers
carts_step "Cleaning up existing containers"
docker rm -f arts-node-1 arts-node-2 arts-node-3 arts-node-4 >/dev/null 2>&1 || true

# Start all containers from prebuilt image
carts_step "Starting 4 ARTS node containers"
docker run -d --name arts-node-1 --hostname arts-node-1 --network bridge -e ARTS_NODE_ID=0 arts-node:built
docker run -d --name arts-node-2 --hostname arts-node-2 --network bridge -e ARTS_NODE_ID=1 arts-node:built
docker run -d --name arts-node-3 --hostname arts-node-3 --network bridge -e ARTS_NODE_ID=2 arts-node:built
docker run -d --name arts-node-4 --hostname arts-node-4 --network bridge -e ARTS_NODE_ID=3 arts-node:built

# Wait for containers to be ready
carts_info "Waiting for containers to be ready"
sleep 3

# Setup SSH keys between containers  - required on each run due to dynamic IPs
carts_step "Setting up SSH connectivity between containers"

# Get all arts-node containers
ALL_NODES=$(docker ps --format '{{.Names}}' | grep '^arts-node-' | sort)

# Setup SSH keyscan for all node combinations
for node_i in $ALL_NODES; do
    for node_j in $ALL_NODES; do
        if [ "$node_i" != "$node_j" ]; then
            NODE_J_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$node_j")
            docker exec "$node_i" bash -c "ssh-keyscan -H $NODE_J_IP >> /root/.ssh/known_hosts" >/dev/null 2>&1
        fi
    done
done

# Test SSH connectivity between first two nodes (if they exist)
FIRST_NODE=$(echo "$ALL_NODES" | head -1)
SECOND_NODE=$(echo "$ALL_NODES" | head -2 | tail -1)
if [ -n "$FIRST_NODE" ] && [ -n "$SECOND_NODE" ] && [ "$FIRST_NODE" != "$SECOND_NODE" ]; then
    SECOND_NODE_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$SECOND_NODE")
    docker exec "$FIRST_NODE" bash -c "ssh -o StrictHostKeyChecking=no root@$SECOND_NODE_IP 'echo SSH test successful'" >/dev/null 2>&1
fi

carts_header "Multi-Node Setup Complete"
carts_success "CARTS multi-node containers are ready!"
carts_container_status

# If an example file was provided, execute it
if [[ -n "$EXAMPLE_FILE" ]]; then
    carts_header "Executing Example"
    carts_info "Executing example: $EXAMPLE_FILE"

    # Find the test file using find command
    CARTS_DIR="$(cd .. && pwd)"
    TEST_FILE=""

    # First, try the file as provided (handles absolute paths and relative paths with directories)
    if [[ -f "$EXAMPLE_FILE" ]]; then
        TEST_FILE="$(realpath "$EXAMPLE_FILE")"
    else
        # Use find to search in tests and examples directories
        TEST_FILE=$(find "${CARTS_DIR}/tests" "${CARTS_DIR}/examples" -name "$EXAMPLE_FILE" -type f 2>/dev/null | head -1)
        
        # If not found, try with basename only (in case full path was provided)
        if [[ -z "$TEST_FILE" ]]; then
            BASENAME_FILE=$(basename "$EXAMPLE_FILE")
            TEST_FILE=$(find "${CARTS_DIR}/tests" "${CARTS_DIR}/examples" -name "$BASENAME_FILE" -type f 2>/dev/null | head -1)
        fi
    fi

    if [[ -z "$TEST_FILE" ]]; then
        carts_error "Example file '$EXAMPLE_FILE' not found in tests or examples directories"
        exit 1
    fi

    carts_success "Found test file: $TEST_FILE"

    # Execute the pipeline and run the binary inside the master container
    MASTER_CONTAINER="arts-node-1"
    FILE_DIR=$(dirname "$TEST_FILE")
    FILE_NAME=$(basename "$TEST_FILE")
    BASE_NAME="${FILE_NAME%.*}"

    # Convert host path to container path
    RELATIVE_PATH="${FILE_DIR#*/carts/}"
    CONTAINER_FILE_DIR="/opt/carts/$RELATIVE_PATH"

    carts_info "Executing inside master container ($MASTER_CONTAINER):"
    carts_substep "Directory: $CONTAINER_FILE_DIR"
    carts_substep "File: $FILE_NAME"
    carts_substep "Binary: $BASE_NAME"

    # First: build the binary on the master node
    REMOTE_BUILD_CMD="cd '$CONTAINER_FILE_DIR' && carts execute '$FILE_NAME'"
    if [[ -n "$BUILD_ARGS" ]]; then
        REMOTE_BUILD_CMD+=" $BUILD_ARGS"
    fi
    carts_build "Build command: $REMOTE_BUILD_CMD"
    docker exec "$MASTER_CONTAINER" bash -c "$REMOTE_BUILD_CMD"

    # Then: copy the built binary to all other nodes so SSH launcher can start it remotely
    OTHER_NODES=$(docker ps --format '{{.Names}}' | grep '^arts-node-' | grep -v "$MASTER_CONTAINER" || true)
    for node in $OTHER_NODES; do
        NODE_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$node")
        carts_info "Copying binary to $node ($NODE_IP): $CONTAINER_FILE_DIR/$BASE_NAME"
        docker exec "$MASTER_CONTAINER" bash -c "scp '$CONTAINER_FILE_DIR/$BASE_NAME' root@$NODE_IP:'$CONTAINER_FILE_DIR/'"
    done

    # Finally: run the binary on the master node (ARTS will SSH to others)
    REMOTE_RUN_CMD="cd '$CONTAINER_FILE_DIR' && ./$BASE_NAME"
    if [[ -n "$RUNTIME_ARGS" ]]; then
        REMOTE_RUN_CMD+=" $RUNTIME_ARGS"
    fi
    carts_build "Run command: $REMOTE_RUN_CMD"
    docker exec "$MASTER_CONTAINER" bash -c "$REMOTE_RUN_CMD"
fi
