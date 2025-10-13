#!/bin/bash
# Common utility functions for TripoSR scripts
# Source this file in your scripts: source "$(dirname "$0")/utils/common.sh"

# Colors
export RED='\033[0;31m'
export GREEN='\033[0;32m'
export YELLOW='\033[1;33m'
export BLUE='\033[0;34m'
export CYAN='\033[0;36m'
export MAGENTA='\033[0;35m'
export NC='\033[0m' # No Color

# Print functions
print_header() {
    echo -e "${CYAN}=========================================="
    echo "$1"
    echo -e "==========================================${NC}"
}

print_section() {
    echo -e "\n${BLUE}>>> $1${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ $1${NC}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if running in WSL
is_wsl() {
    grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null
}

# Get WSL version
get_wsl_version() {
    if grep -qi wsl2 /proc/version || [ "$(uname -r | grep -c WSL2)" -gt 0 ]; then
        echo "2"
    elif grep -qi microsoft /proc/version; then
        echo "1"
    else
        echo "0"
    fi
}

# Check CUDA availability
check_cuda() {
    if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Get CUDA version
get_cuda_version() {
    if check_cuda; then
        nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo ""
    else
        echo ""
    fi
}

# Get GPU name
get_gpu_name() {
    if check_cuda; then
        nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    else
        echo "No GPU detected"
    fi
}

# Setup CUDA environment
setup_cuda_env() {
    local cuda_paths=(
        "/usr/local/cuda"
        "/usr/local/cuda-12.6"
        "/usr/local/cuda-12"
        "/usr/local/cuda-11"
    )

    for cuda_path in "${cuda_paths[@]}"; do
        if [ -d "$cuda_path/bin" ]; then
            export PATH="$cuda_path/bin:$PATH"
            export LD_LIBRARY_PATH="$cuda_path/lib64:$LD_LIBRARY_PATH"
            export CUDA_HOME="$cuda_path"
            return 0
        fi
    done
    return 1
}

# Check Python version
check_python_version() {
    local required_version=$1
    if command_exists "python$required_version"; then
        echo "python$required_version"
        return 0
    elif command_exists python3; then
        local version=$(python3 --version 2>&1 | grep -oP "Python \K[0-9]+\.[0-9]+")
        if [ "$version" == "$required_version" ]; then
            echo "python3"
            return 0
        fi
    fi
    return 1
}

# Get system memory in GB
get_memory_gb() {
    free -g | awk '/^Mem:/{print $2}'
}

# Get disk space in GB
get_disk_space_gb() {
    df -BG . | tail -n 1 | awk '{print $4}' | tr -d 'G'
}

# Confirm action
confirm() {
    local prompt=$1
    local default=${2:-n}

    if [ "$default" == "y" ]; then
        prompt="$prompt [Y/n]: "
    else
        prompt="$prompt [y/N]: "
    fi

    read -p "$prompt" -n 1 -r
    echo

    if [ "$default" == "y" ]; then
        [[ ! $REPLY =~ ^[Nn]$ ]]
    else
        [[ $REPLY =~ ^[Yy]$ ]]
    fi
}

# Check if running with sudo
is_root() {
    [ "$EUID" -eq 0 ]
}

# Download file with progress
download_file() {
    local url=$1
    local output=$2

    if command_exists wget; then
        wget -O "$output" "$url"
    elif command_exists curl; then
        curl -L -o "$output" "$url"
    else
        print_error "wget or curl is required"
        return 1
    fi
}

# Create backup of file
backup_file() {
    local file=$1
    if [ -f "$file" ]; then
        local backup="${file}.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$file" "$backup"
        print_info "Backed up to: $backup"
    fi
}

# Check disk space before operation
check_disk_space() {
    local required_gb=$1
    local available=$(get_disk_space_gb)

    if [ "$available" -lt "$required_gb" ]; then
        print_error "Insufficient disk space. Required: ${required_gb}GB, Available: ${available}GB"
        return 1
    fi
    return 0
}

# Print system summary
print_system_summary() {
    print_section "System Summary"
    echo "OS: $(lsb_release -d 2>/dev/null | cut -f2 || echo "Unknown")"
    echo "Kernel: $(uname -r)"
    echo "Memory: $(get_memory_gb)GB"
    echo "Disk Space: $(get_disk_space_gb)GB available"

    if check_cuda; then
        echo "GPU: $(get_gpu_name)"
        echo "CUDA: $(get_cuda_version)"
    else
        echo "GPU: Not available"
    fi
}

# Error handler
error_exit() {
    print_error "$1"
    exit "${2:-1}"
}

# Spinner for long operations
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while ps -p $pid > /dev/null 2>&1; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Progress bar
progress_bar() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))

    printf "\r["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' ' '
    printf "] %3d%%" $percentage

    if [ "$current" -eq "$total" ]; then
        echo
    fi
}

# Log functions
LOG_FILE="${LOG_FILE:-/tmp/triposr_install.log}"

log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1" >> "$LOG_FILE"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1" >> "$LOG_FILE"
}

log_warning() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1" >> "$LOG_FILE"
}

# Initialize log file
init_log() {
    mkdir -p "$(dirname "$LOG_FILE")"
    echo "=== TripoSR Installation Log ===" > "$LOG_FILE"
    echo "Started at: $(date)" >> "$LOG_FILE"
    echo "================================" >> "$LOG_FILE"
}

# Export all functions
export -f print_header print_section print_success print_error print_warning print_info
export -f command_exists is_wsl get_wsl_version
export -f check_cuda get_cuda_version get_gpu_name setup_cuda_env
export -f check_python_version get_memory_gb get_disk_space_gb
export -f confirm is_root download_file backup_file check_disk_space
export -f print_system_summary error_exit spinner progress_bar
export -f log_info log_error log_warning init_log
