# Docker Multi-Node ARTS Testing

This folder contains everything needed to run ARTS tests across multiple Docker containers, simulating a true multi-node environment.

## Prerequisites

### Install Docker Runtime (Colima)
We use Colima for terminal-only Docker with full CPU access:

```bash
brew install colima

# Start Colima with maximum resources for fast compilation
colima start --cpu 16 --memory 32
```

**Important**: Configure Colima with enough resources for fast compilation:
- **CPU**: Use all available cores (e.g., `--cpu 16` for 16-core Mac)
- **Memory**: Use at least 16GB (e.g., `--memory 32` for 32GB Mac)

### Verify Docker is Working
```bash
docker ps
```

## Files

- `docker-compose.yml` - Orchestrates 2 ARTS containers (optional, we use manual commands)
- `Dockerfile` - Optimized base image with ARTS dependencies and SSH setup
- `docker-build.sh` - **One-time build script** - Creates base image with all dependencies and CARTS pre-built
- `docker-run.sh` - **Run script** - Loads pre-built image and starts multi-node containers
- `docker-update.sh` - **Update script** - Pulls latest changes and rebuilds CARTS components
- `docker-clean.sh` - Cleanup script - Removes all Docker containers and images
- `run-test.sh` - Helper script to run ARTS tests
- `.dockerignore` - Reduces build context for faster builds

## Step-by-Step Setup

### 1. One-Time Build (Run Once)
```bash
cd docker
./docker-build.sh
```

**What this does**:
- Builds base Docker image with Ubuntu 22.04 and all dependencies
- Installs LLVM 18, binutils, and CMake required by CARTS
- Sets up SSH for passwordless access between containers
- Clones CARTS repository and runs `carts-setup.py` with all available CPU cores
- The setup script handles the correct build order automatically
- Creates `arts-node:built` image with CARTS pre-installed

**Time**: 3-5 minutes (one-time only, with 16 cores and optimizations)

### 2. Start Multi-Node Containers
```bash
./docker-run.sh                                         # Start containers only
./docker-run.sh parallel.c                              # Start containers and run example
./docker-run.sh matrixmul -- 100 10                     # Start containers and run with runtime arguments
./docker-run.sh fib --run-args --debug-only=db -- 15    # Start containers and run with build+runtime args
```

**What this does**:
- Loads the pre-built `arts-node:built` image
- Starts 2 containers (arts-node-1 and arts-node-2)
- Sets up SSH keys between containers
- Creates multi-node ARTS configuration
- Tests SSH connectivity
- **If example file provided**: Builds and executes the example distributedly

**Time**: 30 seconds (very fast)

### 3. Fast Rebuild (For Development)
```bash
./docker-fast-rebuild.sh
```

**What this does**:
- Skips base image rebuild (uses existing arts-node-base)
- Only rebuilds CARTS components
- Perfect for development when you change CARTS code
- Uses shallow git clone for faster downloads

**Time**: 1-2 minutes (with 16 cores)

### 4. Update CARTS Components (When Needed)
```bash
./docker-update.sh                    # Update only changed components
./docker-update.sh --arts -f          # Force rebuild ARTS component
./docker-update.sh --llvm -f          # Force rebuild LLVM and dependent components
./docker-update.sh -f                 # Force rebuild everything
```

**What this does**:
- Pulls latest changes from CARTS, Polygeist, ARTS runtime, and LLVM repositories
- By default, only rebuilds components that have detected changes
- With force options, rebuilds specified components regardless of changes
- Rebuilds components with all available CPU cores in correct order:
  - `carts build --llvm` (when LLVM changes or forced)
  - `carts build --polygeist` (when Polygeist or LLVM changes)
  - `carts build --arts` (when ARTS changes or forced)
  - `carts build` (when CARTS changes or forced)
- Updates the `arts-node:built` image

**Time**: 2-5 minutes (depending on changes, with 16 cores)

**Force Options**:
- `--force, -f`: Force rebuild mode (rebuilds everything)
- `--arts, -a`: Force rebuild ARTS runtime only
- `--polygeist, -p`: Force rebuild Polygeist compiler only
- `--llvm, -l`: Force rebuild LLVM compiler (also rebuilds Polygeist)
- `--carts, -c`: Force rebuild CARTS framework only

## Running Tests

### Method 1: Manual Commands
```bash
# Access a container
docker exec -it arts-node-1 bash

# Run a test using carts wrapper
cd /opt/carts
carts execute tests/test/parallel/parallel.c
./parallel
```

### Method 2: Using Helper Script
```bash
./run-test.sh tests/test/parallel
```

### Method 3: Using carts docker command
```bash
# From host machine
carts docker --run example.c                    # Run example with default settings
carts docker --run matrixmul -- 100 10          # Run with runtime arguments: matrix size 100, block size 10
carts docker --run fib --run-args --debug-only=db -- 15    # Build with debug flags, run fibonacci with input 15
```

### Method 4: Using docker-run.sh directly
```bash
# From docker directory
./docker-run.sh parallel.c                    # Run example with default settings
./docker-run.sh matrixmul -- 100 10          # Run with runtime arguments: matrix size 100, block size 10
./docker-run.sh fib --run-args --debug-only=db -- 15    # Build with debug flags, run fibonacci with input 15
```

## CARTS Docker Commands

The `carts docker` command provides a unified interface for Docker operations:

```bash
carts docker --run <file> [build_args] [-- runtime_args]    # Run example in Docker containers
carts docker --update [options]                              # Update and rebuild Docker containers
carts docker --clean                                         # Clean all Docker containers and images
carts docker --help                                          # Show help
```

### Update Options

The `--update` command supports force rebuild options:

```bash
carts docker --update                    # Update only changed components
carts docker --update --arts -f         # Force rebuild ARTS component
carts docker --update --llvm -f         # Force rebuild LLVM and dependent components
carts docker --update -f                # Force rebuild everything
```

**Available options**:
- `--arts, -a`: Force rebuild ARTS runtime
- `--polygeist, -p`: Force rebuild Polygeist compiler
- `--llvm, -l`: Force rebuild LLVM compiler
- `--carts, -c`: Force rebuild CARTS framework
- `--force, -f`: Force rebuild mode (rebuilds everything)

### Argument Separation

The `--run` command supports separating build-time arguments from runtime arguments:

- **Build arguments** (before `--`): Passed to the `carts execute` command (compilation flags, debug options, etc.)
- **Runtime arguments** (after `--`): Passed to the actual binary execution (program input parameters)

**Examples:**
```bash
# Runtime arguments only
carts docker --run matrixmul -- 100 10

# Build arguments only
carts docker --run fib --run-args --debug-only=db

# Both build and runtime arguments
carts docker --run fib --run-args --debug-only=db -- 15
```

### Script Functionality

Each Docker script has a specific purpose:

- **`docker-build.sh`**: One-time setup script that creates the base Docker image with all CARTS dependencies pre-installed. This script should only be run once initially.

- **`docker-run.sh`**: Starts the multi-node Docker containers and sets up the ARTS runtime environment. This script configures SSH between containers and generates the multi-node ARTS configuration file. If an example file is provided as an argument, it will also build and execute the example distributedly across the containers.

- **`docker-update.sh`**: Intelligent update script that detects changes in CARTS, Polygeist, ARTS runtime, and LLVM repositories. Only rebuilds components that have actually changed, saving time during development.

- **`docker-clean.sh`**: Complete cleanup script that removes all Docker containers, images, and networks related to CARTS. Use this to reset the Docker environment.

## Container Management

### Check Container Status
```bash
docker ps
```

### Access Container Shell
```bash
docker exec -it arts-node-1 bash
```

### View Container Logs
```bash
docker logs arts-node-1
```

### Stop Containers
```bash
docker stop arts-node-1 arts-node-2
```

### Remove Containers
```bash
docker rm arts-node-1 arts-node-2
```

### Clean Up Everything
```bash
./docker-clean.sh
```

## Container Details

Each container:
- **Hostname**: arts-node-1, arts-node-2
- **OS**: Ubuntu 22.04
- **User**: root (for SSH access)
- **Working Directory**: /opt/carts (CARTS pre-installed)
- **SSH**: Passwordless SSH between all containers
- **IP Addresses**: Dynamically assigned by Docker (typically 172.17.0.x)

## ARTS Configuration

The setup generates a multi-node ARTS configuration:
- **2 nodes** with Docker-assigned IPs
- **SSH launcher** for inter-node communication
- **Port 34739** for ARTS communication
- **4 threads per node**
- **Config file**: `/opt/carts/arts.cfg`

## Troubleshooting

### SSH Connectivity Issues
```bash
# Test SSH between specific nodes
docker exec arts-node-1 bash -c "ssh -o StrictHostKeyChecking=no root@172.17.0.3 'echo SSH works!'"
```

### Container Not Starting
```bash
# Check container logs
docker logs arts-node-1

# Restart a specific container
docker restart arts-node-1
```

### Build Issues (during `docker-build.sh`)
```bash
# If the build fails in the builder container, you can access it before it's removed:
docker exec -it arts-node-builder bash
# Then manually debug the build process in /opt/carts
```

## Advantages over Loopback IPs

- **No system configuration** - No sudo, no loopback aliases
- **True isolation** - Each container is a real separate node
- **Automatic networking** - Docker handles all networking
- **SSH pre-configured** - Passwordless SSH between containers
- **Easy cleanup** - Just run cleanup script
- **Reproducible** - Same setup every time
- **No conflicts** - Won't interfere with your system

## Notes

- **Building ARTS**: CARTS is pre-built in the Docker image with all dependencies
- **IP Addresses**: Container IPs are dynamically assigned and may change between runs
- **SSH Keys**: Each container generates its own SSH keys for inter-container communication
- **Self-contained**: All containers have CARTS pre-installed at `/opt/carts`