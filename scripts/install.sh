#!/bin/bash

# TripoSR Installation Script for WSL
# Uses uv and Python 3.11

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.11"
VENV_NAME=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}=========================================="
echo "TripoSR Installation Script"
echo "==========================================${NC}"
echo ""

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}>>> $1${NC}\n"
}

# Function to check command availability
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Change to script directory
cd "$SCRIPT_DIR"

# 1. Check for uv
print_section "Checking for uv package manager"
if ! command_exists uv; then
    echo -e "${YELLOW}uv not found. Installing uv...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command_exists uv; then
        echo -e "${RED}Failed to install uv. Please install manually:${NC}"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
    echo -e "${GREEN}uv installed successfully!${NC}"
else
    echo -e "${GREEN}uv is already installed: $(uv --version)${NC}"
fi

# 2. Detect CUDA version
print_section "Detecting CUDA version"
CUDA_AVAILABLE=false
CUDA_VERSION=""
PYTORCH_INDEX=""

if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    CUDA_AVAILABLE=true
    # Extract CUDA version
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || echo "")

    if [ -n "$CUDA_VERSION" ]; then
        echo -e "${GREEN}NVIDIA GPU detected with CUDA $CUDA_VERSION${NC}"

        # Determine PyTorch CUDA version
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        if [ "$CUDA_MAJOR" -ge 12 ]; then
            PYTORCH_CUDA="cu121"
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu121"
            echo -e "${CYAN}Will install PyTorch with CUDA 12.1 support${NC}"
        elif [ "$CUDA_MAJOR" -eq 11 ]; then
            PYTORCH_CUDA="cu118"
            PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"
            echo -e "${CYAN}Will install PyTorch with CUDA 11.8 support${NC}"
        else
            echo -e "${YELLOW}CUDA version $CUDA_VERSION detected, will use CPU version${NC}"
            CUDA_AVAILABLE=false
        fi
    else
        echo -e "${YELLOW}Could not determine CUDA version${NC}"
        CUDA_AVAILABLE=false
    fi
else
    echo -e "${YELLOW}NVIDIA GPU not detected. Will install CPU-only PyTorch${NC}"
    echo -e "${YELLOW}Warning: TripoSR will be significantly slower on CPU${NC}"
fi

# 3. Create Python 3.11 virtual environment with uv
print_section "Creating Python $PYTHON_VERSION virtual environment"

if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_NAME${NC}"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_NAME"
    else
        echo -e "${CYAN}Using existing virtual environment${NC}"
    fi
fi

if [ ! -d "$VENV_NAME" ]; then
    echo -e "${CYAN}Creating virtual environment with Python $PYTHON_VERSION...${NC}"
    uv venv --python "$PYTHON_VERSION" "$VENV_NAME"
    echo -e "${GREEN}Virtual environment created successfully!${NC}"
fi

# Activate virtual environment
source "$VENV_NAME/bin/activate"

# Verify Python version
PYTHON_ACTUAL=$(python --version 2>&1 | grep -oP "Python \K[0-9]+\.[0-9]+")
echo -e "${GREEN}Using Python $PYTHON_ACTUAL${NC}"

# 4. Upgrade pip and setuptools
print_section "Upgrading pip and setuptools"
uv pip install --upgrade pip setuptools
# Ensure setuptools >= 49.6.0 for torchmcubes
uv pip install "setuptools>=49.6.0"

# 5. Install PyTorch
print_section "Installing PyTorch"

if [ "$CUDA_AVAILABLE" = true ] && [ -n "$PYTORCH_INDEX" ]; then
    echo -e "${CYAN}Installing PyTorch with CUDA support...${NC}"
    uv pip install torch torchvision --index-url "$PYTORCH_INDEX"
else
    echo -e "${CYAN}Installing CPU-only PyTorch...${NC}"
    uv pip install torch torchvision
fi

# Verify PyTorch installation
echo -e "\n${CYAN}Verifying PyTorch installation...${NC}"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# 6. Install TripoSR dependencies
print_section "Installing TripoSR dependencies"

if [ -f "requirements.txt" ]; then
    echo -e "${CYAN}Installing dependencies (excluding torchmcubes)...${NC}"
    # Install all dependencies except torchmcubes first
    grep -v "torchmcubes" requirements.txt > /tmp/requirements_no_mcubes.txt
    uv pip install -r /tmp/requirements_no_mcubes.txt

    echo -e "${GREEN}Core dependencies installed!${NC}"

    # Install torchmcubes separately with proper environment
    echo -e "${CYAN}Installing torchmcubes with CUDA support...${NC}"

    # Set environment variables for torchmcubes compilation
    export CMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null || echo "")
    export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"  # Support for modern GPUs

    # Try to install torchmcubes
    if uv pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git; then
        echo -e "${GREEN}torchmcubes installed successfully!${NC}"
    else
        echo -e "${YELLOW}torchmcubes CUDA build failed, trying CPU-only version...${NC}"
        # If CUDA build fails, install CPU version
        FORCE_CUDA=0 uv pip install --no-build-isolation git+https://github.com/tatsy/torchmcubes.git || true
    fi

    echo -e "${GREEN}Dependencies installed successfully!${NC}"
else
    echo -e "${RED}requirements.txt not found!${NC}"
    exit 1
fi

# 7. Verify torchmcubes installation
print_section "Verifying torchmcubes with CUDA support"

TORCHMCUBES_OK=$(python -c "
import torch
try:
    import torchmcubes
    if torch.cuda.is_available():
        try:
            # Test CUDA functionality
            x = torch.randn(1, device='cuda')
            print('OK')
        except:
            print('FAIL')
    else:
        print('CPU')
except Exception as e:
    print(f'ERROR: {e}')
" 2>&1)

if [ "$TORCHMCUBES_OK" = "OK" ]; then
    echo -e "${GREEN}torchmcubes is properly installed with CUDA support${NC}"
elif [ "$TORCHMCUBES_OK" = "CPU" ]; then
    echo -e "${YELLOW}torchmcubes installed (CPU mode)${NC}"
else
    echo -e "${YELLOW}torchmcubes may need reinstallation${NC}"
    echo -e "${CYAN}Attempting to reinstall torchmcubes...${NC}"

    # Reinstall torchmcubes
    uv pip uninstall -y torchmcubes || true
    uv pip install git+https://github.com/tatsy/torchmcubes.git

    echo -e "${GREEN}torchmcubes reinstalled${NC}"
fi

# 8. Test basic imports
print_section "Testing TripoSR imports"

python -c "
import sys
try:
    import torch
    import transformers
    import trimesh
    import rembg
    import omegaconf
    import einops
    print('✓ All core dependencies imported successfully')
    sys.exit(0)
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}All dependencies are working correctly!${NC}"
else
    echo -e "${RED}Some dependencies failed to import${NC}"
    exit 1
fi

# 9. Download model (optional)
print_section "Model setup"

echo -e "${CYAN}TripoSR will automatically download the model on first run.${NC}"
echo -e "${CYAN}Model will be cached in ~/.cache/huggingface/hub/${NC}"
echo ""
echo -e "${YELLOW}Note: The model is approximately 1.5GB${NC}"

# 10. Create a test run script
print_section "Creating helper scripts"

cat > run_triposr.sh << 'EOFSCRIPT'
#!/bin/bash
# Helper script to run TripoSR with virtual environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run install_triposr.sh first."
    exit 1
fi

# Run TripoSR
python run.py "$@"
EOFSCRIPT

chmod +x run_triposr.sh

echo -e "${GREEN}Created run_triposr.sh helper script${NC}"

# 11. Final summary
print_section "Installation Complete!"

echo -e "${GREEN}=========================================="
echo "TripoSR has been successfully installed!"
echo "==========================================${NC}"
echo ""
echo -e "${CYAN}To use TripoSR:${NC}"
echo ""
echo -e "1. Activate the virtual environment:"
echo -e "   ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "2. Run TripoSR on an image:"
echo -e "   ${YELLOW}python run.py examples/chair.png --output-dir output/${NC}"
echo ""
echo -e "   Or use the helper script:"
echo -e "   ${YELLOW}./run_triposr.sh examples/chair.png --output-dir output/${NC}"
echo ""
echo -e "3. Start the Gradio app:"
echo -e "   ${YELLOW}python gradio_app.py${NC}"
echo ""
echo -e "4. For texture baking:"
echo -e "   ${YELLOW}python run.py image.png --bake-texture --texture-resolution 1024${NC}"
echo ""
echo -e "${CYAN}System Information:${NC}"
echo -e "  Python: $(python --version)"
echo -e "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
if [ "$CUDA_AVAILABLE" = true ]; then
    echo -e "  CUDA: ${GREEN}Available${NC} ($(python -c 'import torch; print(torch.version.cuda)'))"
    echo -e "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo -e "  CUDA: ${YELLOW}Not available (CPU mode)${NC}"
fi
echo ""
echo -e "${CYAN}For help and options:${NC}"
echo -e "  ${YELLOW}python run.py --help${NC}"
echo ""
echo -e "${GREEN}Happy 3D reconstructing!${NC}"
