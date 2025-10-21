#!/bin/bash

# CARTS Common Print Functions
# This script provides consistent colored output functions for all CARTS scripts
# Source this file in other scripts: source "$(dirname "$0")/../config/carts-print.sh"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
DIM='\033[2m'
DARK_BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Function to print colored text with consistent formatting
carts_print() {
    echo -e "$1"
}

carts_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

carts_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

carts_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

carts_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

carts_debug() {
    echo -e "${DIM}[DEBUG]${NC} $1"
}

carts_header() {
    echo -e "\n${BOLD}${DARK_BLUE}========================================${NC}"
    echo -e "${BOLD}${DARK_BLUE} $1${NC}"
    echo -e "${BOLD}${DARK_BLUE}========================================${NC}"
}

carts_step() {
    echo -e "${CYAN}->${NC} ${BOLD}$1${NC}"
}

carts_substep() {
    echo -e "  ${CYAN}•${NC} $1"
}

carts_line() {
    echo -e "${WHITE}- - - - - - - - - - - - - - - - - - - - -${NC}"
}

carts_double_line() {
    echo -e "${WHITE}════════════════════════════════════════${NC}"
}

carts_progress() {
    echo -e "${YELLOW}[PROGRESS]${NC} $1"
}

carts_build() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

carts_container() {
    echo -e "${CYAN}[CONTAINER]${NC} $1"
}

carts_git() {
    echo -e "${PURPLE}[GIT]${NC} $1"
}

carts_git_status() {
    echo -e "\n${BOLD}${BLUE}Git Status${NC}"
    carts_line
    echo -e "  ${CYAN}•${NC} Branch: ${BOLD}$1${NC}"
    echo -e "  ${CYAN}•${NC} Commit: ${BOLD}$2${NC}"
    echo -e "  ${CYAN}•${NC} Status: ${BOLD}$3${NC}"
    carts_line
}

carts_git_progress() {
    echo -e "  ${YELLOW}[GIT]${NC} $1"
}

# Additional git styling functions
carts_git_success() {
    echo -e "  ${GREEN}[GIT SUCCESS]${NC} $1"
}

carts_git_warning() {
    echo -e "  ${YELLOW}[GIT WARNING]${NC} $1"
}

carts_git_error() {
    echo -e "  ${RED}[GIT ERROR]${NC} $1"
}

# Function to show git operation summary
carts_git_summary() {
    echo -e "\n${BOLD}${GREEN}Git Operation Complete${NC}"
    carts_line
    echo -e "  ${CYAN}•${NC} Operation: ${BOLD}$1${NC}"
    echo -e "  ${CYAN}•${NC} Files Changed: ${BOLD}$2${NC}"
    echo -e "  ${CYAN}•${NC} Insertions: ${BOLD}$3${NC}"
    echo -e "  ${CYAN}•${NC} Deletions: ${BOLD}$4${NC}"
    carts_line
}

carts_docker() {
    echo -e "${BLUE}[DOCKER]${NC} $1"
}

# Function to print completion message
carts_completion() {
    echo -e "\n${BOLD}${GREEN}[SUCCESS]${NC} ${BOLD}$1${NC}"
}

# Function to print failure message
carts_failure() {
    echo -e "\n${BOLD}${RED}[FAILED]${NC} ${BOLD}$1${NC}"
}

# Function to print a table header
carts_table_header() {
    echo -e "${BOLD}${WHITE}$1${NC}"
    carts_line
}

# Function to print aligned command descriptions (replaces print_cmd)
carts_cmd() {
    printf "  ${BOLD}${CYAN}%-50s${NC} ${DIM}##${NC} %s\n" "$1" "$2"
}

# Function to print a summary section
carts_summary() {
    echo -e "\n${BOLD}${YELLOW}Summary:${NC}"
    carts_line
    for item in "$@"; do
        echo -e "  ${CYAN}•${NC} $item"
    done
    carts_line
}

# Function to print next steps
carts_next_steps() {
    echo -e "\n${BOLD}${GREEN}Next Steps:${NC}"
    carts_line
    for step in "$@"; do
        echo -e "  ${CYAN}->${NC} $step"
    done
    carts_line
}

# Function to print environment info
carts_env_info() {
    echo -e "\n${BOLD}${BLUE}Environment Info:${NC}"
    carts_line
    echo -e "  ${CYAN}•${NC} CPU Cores: $1"
    echo -e "  ${CYAN}•${NC} Git Branch: $2"
    echo -e "  ${CYAN}•${NC} Docker Version: $(docker --version 2>/dev/null | cut -d' ' -f3 | cut -d',' -f1 || echo 'Unknown')"
    carts_line
}

# Function to print container status table
carts_container_status() {
    carts_print "CONTAINER STATUS"
    carts_line
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | sed 's/^/  /'
    carts_line
}

# Function to print build results
carts_build_results() {
    echo -e "\n${BOLD}${GREEN}Build Results:${NC}"
    carts_line
    echo -e "  ${CYAN}•${NC} CARTS: $1"
    echo -e "  ${CYAN}•${NC} Polygeist: $2"
    echo -e "  ${CYAN}•${NC} ARTS: $3"
    echo -e "  ${CYAN}•${NC} LLVM: $4"
    carts_line
}

# Function to print timing information
carts_timing() {
    local start_time=$1
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local minutes=$((duration / 60))
    local seconds=$((duration % 60))
    
    if [ $minutes -gt 0 ]; then
        echo -e "${BOLD}${YELLOW}  Duration: ${minutes}m ${seconds}s${NC}"
    else
        echo -e "${BOLD}${YELLOW}  Duration: ${seconds}s${NC}"
    fi
}

# Function to print a spinner (for long operations)
carts_spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Function to check if colors are supported
carts_check_color_support() {
    if [ -t 1 ] && [ "$TERM" != "dumb" ]; then
        return 0  # Colors supported
    else
        return 1  # Colors not supported
    fi
}

# Disable colors if not supported
if ! carts_check_color_support; then
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    PURPLE=''
    CYAN=''
    WHITE=''
    BOLD=''
    DIM=''
    NC=''
fi
