#!/bin/bash
# TripoSR System Information Check Script
# Optimized version with common library

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils/common.sh"

# Output file option
OUTPUT_FILE=""
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-o|--output FILE] [-v|--verbose] [-h|--help]"
            echo "Options:"
            echo "  -o, --output FILE   Save output to file"
            echo "  -v, --verbose       Show detailed information"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Redirect output if specified
if [ -n "$OUTPUT_FILE" ]; then
    exec > >(tee "$OUTPUT_FILE")
fi

print_header "TripoSR System Information Check"

# 1. System Information
print_section "System Information"
echo "$(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep "^NAME=" | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
if is_wsl; then
    echo "WSL Version: $(get_wsl_version)"
fi
echo ""

# 2. Hardware Information
print_section "Hardware Information"
CPU_INFO=$(lscpu | grep "Model name" | cut -d':' -f2 | xargs)
echo "CPU: $CPU_INFO"
echo "Memory: $(get_memory_gb)GB"
echo "Disk Space: $(get_disk_space_gb)GB available"
echo ""

# 3. GPU Information
print_section "GPU Information"
if check_cuda; then
    print_success "NVIDIA GPU detected"
    echo "  GPU: $(get_gpu_name)"
    echo "  Driver Version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    echo "  VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
    echo "  CUDA Version: $(get_cuda_version)"

    if [ "$VERBOSE" == "true" ]; then
        echo ""
        nvidia-smi
    fi
else
    print_warning "No NVIDIA GPU detected or drivers not installed"
fi
echo ""

# 4. Python Environment
print_section "Python Environment"
for pyver in python3.11 python3.10 python3.9 python3.8 python3; do
    if command_exists "$pyver"; then
        echo "$pyver: $($pyver --version 2>&1)"
    fi
done
echo ""

# 5. CUDA Toolkit
print_section "CUDA Toolkit"
if setup_cuda_env; then
    if command_exists nvcc; then
        print_success "CUDA toolkit installed"
        nvcc --version | grep "release"
    else
        print_warning "CUDA toolkit not found"
    fi
else
    print_warning "CUDA installation directory not found"
fi
echo ""

# 6. Package Managers
print_section "Package Managers"
if command_exists uv; then
    print_success "uv: $(uv --version)"
else
    print_warning "uv not installed"
    echo "Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if command_exists pip3 || command_exists pip; then
    local pip_cmd=$(command -v pip3 || command -v pip)
    print_success "pip: $($pip_cmd --version | cut -d' ' -f1-2)"
fi
echo ""

# 7. Compatibility Check
print_section "TripoSR Compatibility Check"

COMPAT_SCORE=0
TOTAL_CHECKS=6

# Check 1: GPU
if check_cuda; then
    print_success "GPU: Available"
    ((COMPAT_SCORE++))
else
    print_warning "GPU: Not available (will run on CPU - slower)"
fi

# Check 2: CUDA Version
CUDA_VER=$(get_cuda_version)
if [ -n "$CUDA_VER" ]; then
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 11 ]; then
        print_success "CUDA Version: $CUDA_VER (compatible)"
        ((COMPAT_SCORE++))
    else
        print_warning "CUDA Version: $CUDA_VER (may not be compatible)"
    fi
else
    print_warning "CUDA not detected"
fi

# Check 3: Python
if check_python_version "3.11" >/dev/null || check_python_version "3.10" >/dev/null || check_python_version "3.9" >/dev/null || check_python_version "3.8" >/dev/null; then
    print_success "Python: Compatible version available"
    ((COMPAT_SCORE++))
else
    print_error "Python: No compatible version found (need 3.8+)"
fi

# Check 4: Memory
MEM_GB=$(get_memory_gb)
if [ "$MEM_GB" -ge 8 ]; then
    print_success "Memory: ${MEM_GB}GB (sufficient)"
    ((COMPAT_SCORE++))
else
    print_warning "Memory: ${MEM_GB}GB (recommended: 8GB+)"
fi

# Check 5: Disk Space
DISK_GB=$(get_disk_space_gb)
if [ "$DISK_GB" -ge 10 ]; then
    print_success "Disk Space: ${DISK_GB}GB (sufficient)"
    ((COMPAT_SCORE++))
else
    print_warning "Disk Space: ${DISK_GB}GB (recommended: 10GB+)"
fi

# Check 6: uv
if command_exists uv; then
    print_success "uv package manager: Installed"
    ((COMPAT_SCORE++))
else
    print_warning "uv package manager: Not installed"
fi

echo ""
echo "Compatibility Score: $COMPAT_SCORE/$TOTAL_CHECKS"

if [ "$COMPAT_SCORE" -ge 5 ]; then
    print_success "System is ready for TripoSR installation!"
elif [ "$COMPAT_SCORE" -ge 3 ]; then
    print_warning "System meets minimum requirements but some components may need attention"
else
    print_error "System does not meet minimum requirements"
fi

echo ""
print_section "Recommendations"

if ! check_cuda; then
    echo "• Install NVIDIA drivers for GPU acceleration"
fi

if ! command_exists nvcc; then
    echo "• Install CUDA toolkit: sudo apt-get install -y cuda-toolkit-12-6"
fi

if ! command_exists uv; then
    echo "• Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

if [ "$MEM_GB" -lt 8 ]; then
    echo "• Consider upgrading RAM to at least 8GB"
fi

if [ "$DISK_GB" -lt 10 ]; then
    echo "• Free up disk space (at least 10GB recommended)"
fi

echo ""
print_info "Run './setup.sh' to install TripoSR"

if [ -n "$OUTPUT_FILE" ]; then
    echo ""
    print_success "Output saved to: $OUTPUT_FILE"
fi
