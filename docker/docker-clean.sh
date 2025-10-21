#!/bin/bash

# Docker Clean Script for CARTS Multi-Node Testing
# This script cleans up all Docker containers, images, and networks

set -e

# Source common print functions
source "$(dirname "$0")/../tools/config/carts-print.sh"

carts_header "CARTS Docker Clean"
carts_info "Cleaning up Docker multi-node setup"

# Change to docker directory
cd "$(dirname "$0")"

# Stop and remove containers
carts_step "Stopping and removing containers"
docker rm -f arts-node-1 arts-node-2 >/dev/null 2>&1 || true
docker rm -f arts-node-builder arts-node-update >/dev/null 2>&1 || true

# Remove images
carts_step "Removing Docker images"
docker rmi arts-node:built >/dev/null 2>&1 || true
docker rmi arts-node-base >/dev/null 2>&1 || true

# Remove any orphaned networks
carts_step "Cleaning up networks"
docker network prune -f >/dev/null 2>&1 || true

# Remove all unused Docker resources (uncomment if needed)
carts_step "Removing all unused Docker resources"
docker system prune -af >/dev/null 2>&1 || true

carts_header "Cleanup Complete"
carts_success "Docker cleanup completed successfully!"

carts_next_steps \
    "Run './docker-build.sh' to rebuild base image (one-time)" \
    "Run './docker-run.sh' to start containers"

carts_info "Docker images remaining:"
docker images | grep -E "(arts-node|ubuntu)" || carts_info "No ARTS-related images found"
