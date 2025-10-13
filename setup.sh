#!/bin/bash
# TripoSR One-Click Installation Script
# This script automates the entire installation process

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if scripts directory exists
if [ -f "$SCRIPT_DIR/scripts/utils/common.sh" ]; then
    source "$SCRIPT_DIR/scripts/utils/common.sh"
else
    # Fallback minimal functions if common.sh not found
    print_header() { echo "=== $1 ==="; }
    print_section() { echo ">>> $1"; }
    print_success() { echo "✓ $1"; }
    print_error() { echo "✗ $1"; }
    print_warning() { echo "⚠ $1"; }
    command_exists() { command -v "$1" >/dev/null 2>&1; }
fi

cd "$SCRIPT_DIR"

print_header "TripoSR One-Click Installation"

# Configuration
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_NAME="${VENV_NAME:-.venv}"
SKIP_SYSTEM_CHECK="${SKIP_SYSTEM_CHECK:-false}"
AUTO_YES="${AUTO_YES:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --venv)
            VENV_NAME="$2"
            shift 2
            ;;
        --skip-check)
            SKIP_SYSTEM_CHECK=true
            shift
            ;;
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --python VERSION    Python version to use (default: 3.11)"
            echo "  --venv NAME         Virtual environment name (default: .venv)"
            echo "  --skip-check        Skip system compatibility check"
            echo "  -y, --yes           Automatic yes to prompts"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Step 1: System Check
if [ "$SKIP_SYSTEM_CHECK" != "true" ]; then
    print_section "Step 1/4: System Compatibility Check"

    if [ -f "scripts/check_system.sh" ]; then
        bash scripts/check_system.sh
        echo ""

        if [ "$AUTO_YES" != "true" ]; then
            read -p "Continue with installation? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo "Installation cancelled."
                exit 0
            fi
        fi
    else
        print_warning "System check script not found, skipping..."
    fi
fi

# Step 2: Setup CUDA Environment
print_section "Step 2/4: Setting up CUDA Environment"

if [ -d "/usr/local/cuda/bin" ]; then
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
    print_success "CUDA environment configured"
else
    print_warning "CUDA not found. GPU acceleration may not be available."
    print_warning "To install CUDA toolkit, see docs/INSTALLATION.md"
fi

# Step 3: Run Installation Script
print_section "Step 3/4: Installing TripoSR"

if [ -f "scripts/install.sh" ]; then
    bash scripts/install.sh
elif [ -f "install_triposr.sh" ]; then
    # Fallback to old script if new one doesn't exist
    bash install_triposr.sh
else
    print_error "Installation script not found!"
    exit 1
fi

# Step 4: Verification
print_section "Step 4/4: Verifying Installation"

if [ -f "$VENV_NAME/bin/activate" ]; then
    source "$VENV_NAME/bin/activate"

    python -c "
import sys
try:
    import torch
    import torchmcubes
    import transformers
    print('✓ All core dependencies verified')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" && installation_success=true || installation_success=false

    if [ "$installation_success" = true ]; then
        print_header "Installation Complete!"
        echo ""
        print_success "TripoSR has been successfully installed!"
        echo ""
        echo "Quick Start:"
        echo "  1. Activate environment:  source $VENV_NAME/bin/activate"
        echo "  2. Run on an image:       python run.py examples/chair.png --output-dir output/"
        echo "  3. Start Gradio UI:       python gradio_app.py"
        echo ""
        echo "Documentation: See docs/ directory or README.md"
    else
        print_error "Installation verification failed"
        echo "Check the logs above for errors"
        exit 1
    fi
else
    print_error "Virtual environment not found"
    exit 1
fi
