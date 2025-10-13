#!/bin/bash
# CUDA 完全重装脚本
# 用途：当CUDA环境损坏或需要完全重装时使用
# 警告：这将删除所有CUDA相关组件，请谨慎使用！

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${CYAN}==== $1 ====${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║   CUDA 完全重装脚本                                    ║"
echo "║   ⚠️  这将删除所有 CUDA 组件！                         ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

print_warning "此脚本将执行以下操作："
echo "  1. 完全卸载所有 CUDA 相关包"
echo "  2. 下载并安装 CUDA 12.8 keyring"
echo "  3. 安装 CUDA Toolkit 12.8"
echo "  4. 配置环境变量"
echo "  5. 验证安装"
echo ""

print_warning "建议使用场景："
echo "  - CUDA 环境损坏无法修复"
echo "  - nvcc 命令找不到或版本错误"
echo "  - 多个 CUDA 版本冲突"
echo "  - torchmcubes 编译时出现 CUDA 相关错误"
echo ""

read -p "确认继续？这将删除所有CUDA组件！(yes/no) " -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
    echo "操作已取消"
    exit 0
fi

# 步骤1：完全卸载CUDA
print_step "步骤 1/5: 卸载所有 CUDA 组件"
print_warning "正在卸载..."

sudo apt-get --purge remove -y \
    "*cuda*" \
    "*cublas*" \
    "*cufft*" \
    "*cufile*" \
    "*curand*" \
    "*cusolver*" \
    "*cusparse*" \
    "*gds-tools*" \
    "*npp*" \
    "*nvjpeg*" \
    "nsight*" \
    "*nvvm*" 2>&1 | grep -v "Unable to locate package" || true

sudo apt-get autoremove -y
sudo apt-get autoclean

print_success "CUDA 组件卸载完成"
echo ""

# 步骤2：下载 CUDA keyring
print_step "步骤 2/5: 下载 CUDA 12.8 keyring"

if [ -f "cuda-keyring_1.1-1_all.deb" ]; then
    print_warning "keyring 文件已存在，跳过下载"
else
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
fi

sudo dpkg -i cuda-keyring_1.1-1_all.deb
print_success "keyring 安装完成"
echo ""

# 步骤3：更新并安装 CUDA Toolkit
print_step "步骤 3/5: 安装 CUDA Toolkit 12.8"
print_warning "这可能需要几分钟..."

sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8

print_success "CUDA Toolkit 12.8 安装完成"
echo ""

# 步骤4：配置环境变量
print_step "步骤 4/5: 配置环境变量"

# 检查 .bashrc 是否已包含 CUDA 路径
if grep -q "/usr/local/cuda-12.8/bin" ~/.bashrc; then
    print_warning "环境变量已存在于 .bashrc"
else
    echo "" >> ~/.bashrc
    echo "# CUDA 12.8 环境变量" >> ~/.bashrc
    echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    print_success "环境变量已添加到 .bashrc"
fi

# 立即应用环境变量
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

print_success "环境变量配置完成"
echo ""

# 步骤5：验证安装
print_step "步骤 5/5: 验证 CUDA 安装"

if command -v nvcc &> /dev/null; then
    echo "nvcc 版本："
    nvcc --version
    print_success "CUDA 编译器可用"
else
    print_error "nvcc 未找到！"
    print_warning "请执行以下命令："
    echo "  source ~/.bashrc"
    exit 1
fi

echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU 状态："
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
    print_success "GPU 驱动正常"
else
    print_warning "nvidia-smi 未找到（WSL2 环境可能需要 Windows 端驱动）"
fi

echo ""
print_success "═══════════════════════════════════════"
print_success "  CUDA 12.8 重装完成！               "
print_success "═══════════════════════════════════════"
echo ""

print_step "下一步操作"
echo "1. 重新打开终端或执行："
echo "   source ~/.bashrc"
echo ""
echo "2. 重新安装 PyTorch（如需要）："
echo "   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128"
echo ""
echo "3. 验证 PyTorch CUDA 支持："
echo '   python -c "import torch; print(f'"'"'PyTorch: {torch.__version__}'"'"'); print(f'"'"'CUDA: {torch.cuda.is_available()}'"'"')"'
echo ""
echo "4. 安装 torchmcubes："
echo "   ./scripts/safe_install_torchmcubes.sh"
echo ""

print_warning "重要提醒"
echo "- 如果当前终端 nvcc 仍不可用，请关闭终端重新打开"
echo "- 对于 RTX 50 系列 GPU，确保使用 PyTorch cu128 版本"
echo "- 参考文档：docs/BLACKWELL_GPU_GUIDE.md"
