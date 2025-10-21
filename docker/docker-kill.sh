#!/bin/bash

# Docker Kill Script for CARTS Multi-Node Testing
# This script enters all running ARTS nodes and kills all running processes

set -e

# Source common print functions
source "$(dirname "$0")/../tools/config/carts-print.sh"

carts_header "CARTS Docker Kill"
carts_info "Killing all processes in ARTS containers"

# Change to docker directory
cd "$(dirname "$0")"

# Function to kill processes in a container
kill_processes_in_container() {
    local container_name="$1"
    local container_ip="$2"

    carts_info "Killing processes in $container_name ($container_ip)"

    # This will kill ARTS processes, but keep the container running
    docker exec "$container_name" bash -c "
        # Kill all processes except essential system processes
        ps aux | grep -v -E '(ps|grep|sshd|/usr/sbin/sshd|bash|root|PID)' | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true

        # Also kill any remaining ARTS-related processes more specifically
        pkill -9 -f 'arts' 2>/dev/null || true
        pkill -9 -f 'carts' 2>/dev/null || true

        # Clean up any zombie processes
        ps aux | grep -E '<defunct>' | awk '{print \$2}' | xargs -r kill -9 2>/dev/null || true" 2>/dev/null || true
}

# Get all running ARTS containers
RUNNING_CONTAINERS=$(docker ps --format '{{.Names}}' | grep '^arts-node-' | sort)

if [[ -z "$RUNNING_CONTAINERS" ]]; then
    carts_error "No running ARTS containers found!"
    carts_info "Start containers first with: ./docker-run.sh"
    exit 1
fi

carts_step "Found running containers: $RUNNING_CONTAINERS"

# Kill processes in each container
for container in $RUNNING_CONTAINERS; do
    # Get container IP for display
    container_ip=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "$container" 2>/dev/null || echo "unknown")

    if [[ "$container_ip" == "<no value>" ]]; then
        container_ip="unknown"
    fi

    kill_processes_in_container "$container" "$container_ip"
done

carts_header "Kill Complete"
carts_success "All processes killed in ARTS containers!"
carts_info "Containers are still running but processes have been terminated."

# Show container status
carts_info "Container status:"
docker ps --filter "name=arts-node-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
