#!/bin/bash

# Docker Update Script for CARTS Multi-Node Testing
# This script pulls latest changes and rebuilds CARTS components

set -e

# Source common print functions
source "$(dirname "$0")/../tools/config/carts-print.sh"

# Parse command line arguments
FORCE_MODE=false
FORCE_ARTS=false
FORCE_POLYGEIST=false
FORCE_LLVM=false
FORCE_CARTS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        "--force"|"-f")
            FORCE_MODE=true
            shift
            ;;
        "--arts"|"-a")
            FORCE_ARTS=true
            shift
            ;;
        "--polygeist"|"-p")
            FORCE_POLYGEIST=true
            shift
            ;;
        "--llvm"|"-l")
            FORCE_LLVM=true
            shift
            ;;
        "--carts"|"-c")
            FORCE_CARTS=true
            shift
            ;;
        "--help"|"-h")
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --force, -f        Force rebuild mode (rebuilds even without changes)"
            echo "  --arts, -a         Force rebuild ARTS runtime"
            echo "  --polygeist, -p    Force rebuild Polygeist compiler"
            echo "  --llvm, -l         Force rebuild LLVM compiler"
            echo "  --carts, -c        Force rebuild CARTS framework"
            echo "  --help, -h         Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --arts -f       # Force rebuild ARTS"
            echo "  $0 --llvm -f       # Force rebuild LLVM"
            echo "  $0 -f              # Force rebuild everything"
            exit 0
            ;;
        *)
            carts_error "Unknown option: $1"
            carts_info "Use '$0 --help' for available options"
            exit 1
            ;;
    esac
done

carts_header "CARTS Docker Update"
if [[ "$FORCE_MODE" == "true" ]]; then
    carts_info "Updating CARTS components in Docker (FORCE MODE)"
else
    carts_info "Updating CARTS components in Docker"
fi

# Check for local changes before proceeding
carts_step "Checking for local changes in repositories"
carts_info "Verifying all repositories are clean before updating..."

# Function to check if repository has local changes
check_repo_changes() {
    local repo_path="$1"
    local repo_name="$2"

    if [[ ! -d "$repo_path/.git" ]]; then
        carts_warning "$repo_name ($repo_path): Not a git repository"
        return 0
    fi

    if ! git -C "$repo_path" diff-index --quiet HEAD -- 2>/dev/null; then
        carts_error "$repo_name ($repo_path): Local changes detected!"
        return 1
    fi

    if ! git -C "$repo_path" diff --cached --quiet 2>/dev/null; then
        carts_error "$repo_name ($repo_path): Staged changes detected!"
        return 1
    fi

    carts_success "$repo_name ($repo_path): Clean"
    return 0
}

# Check all repositories for local changes
LOCAL_CHANGES_FOUND=false

# Check CARTS main repository
# Get the current directory (should be the docker directory)
DOCKER_DIR="$(cd "$(dirname "$0")" && pwd)"
CARTS_DIR="$(dirname "$DOCKER_DIR")"
if ! check_repo_changes "$CARTS_DIR" "CARTS"; then
    LOCAL_CHANGES_FOUND=true
fi
# Check Polygeist repository
if ! check_repo_changes "$CARTS_DIR/external/Polygeist" "Polygeist"; then
    LOCAL_CHANGES_FOUND=true
fi
# Check ARTS repository
if ! check_repo_changes "$CARTS_DIR/external/arts" "ARTS"; then
    LOCAL_CHANGES_FOUND=true
fi
# Check LLVM repository (if it exists)
if [[ -d "$CARTS_DIR/external/Polygeist/llvm-project/.git" ]]; then
    if ! check_repo_changes "$CARTS_DIR/external/Polygeist/llvm-project" "LLVM"; then
        LOCAL_CHANGES_FOUND=true
    fi
fi

# If any local changes found, exit
if [[ "$LOCAL_CHANGES_FOUND" == "true" ]]; then
    carts_line
    carts_error "Local changes detected in one or more repositories!"
    carts_info "Please commit and push your changes before running docker update."
    carts_line
    carts_header "Update Cancelled"
    exit 1
fi
carts_line
carts_success "All repositories are clean. Proceeding with update..."
carts_line

# Change to docker directory
cd "$(dirname "$0")"

# Check if the built image exists
if ! docker images | grep -q "arts-node.*built"; then
    carts_error "arts-node:built image not found!"
    carts_warning "Please run './docker-build.sh' first to build the base image."
    exit 1
fi

# Detect CPU cores
DOCKER_CPUS=$(docker run --rm ubuntu:22.04 nproc 2>/dev/null || echo "2")
carts_step "Starting update container with full CPU access"
docker rm -f arts-node-update >/dev/null 2>&1 || true
docker run -d --name arts-node-update --hostname arts-node-update --network bridge \
    --cpus="$DOCKER_CPUS" arts-node:built

# Wait for container to be ready
carts_info "Waiting for update container to be ready"
sleep 5

# Update CARTS components
carts_step "Checking for changes in repositories"
GIT_BRANCH=${GIT_BRANCH:-mlir}

# Check for remote changes inside the container
carts_info "Checking for remote changes in repositories..."

# Execute remote change detection inside the container
REMOTE_CHANGES=$(docker exec arts-node-update bash -c "
    set -e
    GIT_BRANCH='$GIT_BRANCH'

    # Function to check for remote changes
    check_repo() {
        local repo_path=\"\$1\"
        local repo_branch=\"\$2\"

        [[ ! -d \"\$repo_path/.git\" ]] && echo '' && return

        # Fetch and check for changes
        if ! git -C \"\$repo_path\" fetch origin \"\$repo_branch\" 2>/dev/null ||
           ! git -C \"\$repo_path\" show-ref --verify --quiet refs/remotes/origin/\"\$repo_branch\" 2>/dev/null; then
            echo \"WARNING: \$repo_path - Failed to fetch or branch not found\"
            echo ''
            return
        fi

        # Get changes in both directions
        local remote_ahead=\$(git -C \"\$repo_path\" log HEAD..origin/\"\$repo_branch\" --oneline 2>/dev/null)
        local local_ahead=\$(git -C \"\$repo_path\" log origin/\"\$repo_branch\"..HEAD --oneline 2>/dev/null)

        # Output combined changes or empty if no changes
        [[ -n \"\$remote_ahead\$local_ahead\" ]] && echo \"\$remote_ahead\$local_ahead\" || echo ''
    }

    # Check repositories and output in parseable format
    printf 'CARTS:%s\n' \"\$(check_repo '/opt/carts' \"\$GIT_BRANCH\")\"
    printf 'POLYGEIST:%s\n' \"\$(check_repo '/opt/carts/external/Polygeist' 'carts')\"
    printf 'ARTS:%s\n' \"\$(check_repo '/opt/carts/external/arts' 'master')\"

    # Check LLVM if it exists
    if [[ -d '/opt/carts/external/Polygeist/llvm-project/.git' ]]; then
        local llvm_branch=\$(git -C '/opt/carts/external/Polygeist/llvm-project' rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null | sed 's|origin/||' || echo 'main')
        printf 'LLVM:%s\n' \"\$(check_repo '/opt/carts/external/Polygeist/llvm-project' \"\$llvm_branch\")\"
    else
        printf 'LLVM:\n'
    fi
" 2>/dev/null || echo "ERROR: Failed to check remote changes")

# Parse results and handle warnings more efficiently
CARTS_CHANGES=""
POLYGEIST_CHANGES=""
ARTS_CHANGES=""
LLVM_CHANGES=""

while IFS=':' read -r repo changes; do
    case "$repo" in
        CARTS) CARTS_CHANGES="$changes" ;;
        POLYGEIST) POLYGEIST_CHANGES="$changes" ;;
        ARTS) ARTS_CHANGES="$changes" ;;
        LLVM) LLVM_CHANGES="$changes" ;;
        WARNING*) carts_warning "${changes#WARNING: }" ;;
    esac
done <<< "$REMOTE_CHANGES"

BUILD_LLVM=false
BUILD_POLY=false
BUILD_ARTS=false
BUILD_CARTS=false

# Determine what needs to be built
[[ -n "$CARTS_CHANGES" ]] && BUILD_CARTS=true
[[ -n "$POLYGEIST_CHANGES" ]] && BUILD_POLY=true
[[ -n "$ARTS_CHANGES" ]] && BUILD_ARTS=true
[[ -n "$LLVM_CHANGES" ]] && { BUILD_LLVM=true; BUILD_POLY=true; }

# Apply force flags (overrides change detection)
if [[ "$FORCE_MODE" == "true" ]]; then
    BUILD_LLVM=true
    BUILD_POLY=true
    BUILD_ARTS=true
    BUILD_CARTS=true
elif [[ "$FORCE_ARTS" == "true" ]]; then
    BUILD_ARTS=true
elif [[ "$FORCE_POLYGEIST" == "true" ]]; then
    BUILD_POLY=true
elif [[ "$FORCE_LLVM" == "true" ]]; then
    BUILD_LLVM=true
    BUILD_POLY=true  # LLVM changes require rebuilding Polygeist
elif [[ "$FORCE_CARTS" == "true" ]]; then
    BUILD_CARTS=true
fi

carts_line
if [[ "$FORCE_MODE" == "true" ]]; then
    carts_info "Build Mode: FORCE (rebuilding everything)"
else
    carts_info "Change Detection Results:"
fi
[[ -n "$CARTS_CHANGES" ]] && carts_warning "CARTS: Changes detected" || carts_success "CARTS: Up to date"
[[ -n "$POLYGEIST_CHANGES" ]] && carts_warning "Polygeist: Changes detected" || carts_success "Polygeist: Up to date"
[[ -n "$ARTS_CHANGES" ]] && carts_warning "ARTS: Changes detected" || carts_success "ARTS: Up to date"
[[ -n "$LLVM_CHANGES" ]] && carts_warning "LLVM: Changes detected" || carts_success "LLVM: Up to date"

# Show force rebuild indicators
if [[ "$FORCE_ARTS" == "true" ]]; then
    carts_warning "ARTS: Force rebuild enabled"
fi
if [[ "$FORCE_POLYGEIST" == "true" ]]; then
    carts_warning "Polygeist: Force rebuild enabled"
fi
if [[ "$FORCE_LLVM" == "true" ]]; then
    carts_warning "LLVM: Force rebuild enabled"
fi
if [[ "$FORCE_CARTS" == "true" ]]; then
    carts_warning "CARTS: Force rebuild enabled"
fi
carts_line

# If nothing changed, skip
if [[ "$BUILD_LLVM" == "false" && "$BUILD_POLY" == "false" && "$BUILD_ARTS" == "false" && "$BUILD_CARTS" == "false" ]]; then
    carts_success "No changes detected. Skipping build process."
    carts_step "Cleaning up update container"
    docker stop arts-node-update >/dev/null
    docker rm arts-node-update >/dev/null
    carts_header "Update Complete"
    carts_success "No changes needed - arts-node:built image is already up to date!"
    exit 0
fi

carts_step "Changes detected. Proceeding with selective update and build..."
# Pull repos that changed and rebuild selectively
docker exec arts-node-update bash -c "\
    set -e; \
    export MAKEFLAGS='-j$DOCKER_CPUS'; \
    [[ \"$CARTS_CHANGES\" != \"\" ]] && { cd /opt/carts && git pull --no-tags --quiet origin $GIT_BRANCH; }; \
    [[ \"$POLYGEIST_CHANGES\" != \"\" || \"$LLVM_CHANGES\" != \"\" ]] && { cd /opt/carts/external/Polygeist && git pull --no-tags --quiet origin carts; }; \
    [[ \"$ARTS_CHANGES\" != \"\" ]] && { cd /opt/carts/external/arts && git pull --no-tags --quiet origin master; }; \
    [[ \"$LLVM_CHANGES\" != \"\" ]] && { cd /opt/carts && carts build --llvm; }; \
    [[ \"$BUILD_POLY\" == \"true\" ]] && { cd /opt/carts && carts build --polygeist; }; \
    [[ \"$BUILD_ARTS\" == \"true\" ]] && { cd /opt/carts && carts build --arts-info; }; \
    [[ \"$BUILD_LLVM\" == \"true\" || \"$BUILD_POLY\" == \"true\" || \"$BUILD_ARTS\" == \"true\" || \"$BUILD_CARTS\" == \"true\" ]] && { cd /opt/carts && carts build; } \
"

# Commit updated container as new image
carts_step "Committing updated container to arts-node:built image"
docker commit arts-node-update arts-node:built >/dev/null

# Clean up update container
carts_step "Cleaning up update container"
docker stop arts-node-update >/dev/null
docker rm arts-node-update >/dev/null

# Run docker-run.sh
./docker-run.sh

carts_header "Update Complete"
carts_success "CARTS Docker update completed successfully!"

