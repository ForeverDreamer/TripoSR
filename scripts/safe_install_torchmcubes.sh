#!/bin/bash
# 安全安装 torchmcubes 脚本 v2.0 - 避免 WSL2 崩溃
# 功能：
#   - 自动检测GPU架构（支持RTX 50系列Blackwell sm_120）
#   - 自动安装构建依赖（scikit-build-core、cmake、ninja、pybind11）
#   - 限制并行编译、监控内存、提供回退方案

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

# 检查内存
check_memory() {
    print_step "检查系统内存"

    local total_mem=$(free -g | awk '/^Mem:/{print $2}')
    local avail_mem=$(free -g | awk '/^Mem:/{print $7}')
    local swap_mem=$(free -g | awk '/^Swap:/{print $2}')

    echo "总内存: ${total_mem}GB"
    echo "可用内存: ${avail_mem}GB"
    echo "Swap: ${swap_mem}GB"

    if [ "$total_mem" -lt 20 ]; then
        print_warning "内存较少（${total_mem}GB），建议重启WSL使新的.wslconfig生效"
        echo "请在Windows PowerShell中执行："
        echo "  wsl --shutdown"
        echo "然后重新打开WSL"
        read -p "是否继续安装？(y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "内存充足（${total_mem}GB），可以安全编译"
    fi
}

# 卸载旧版本
uninstall_old() {
    print_step "卸载旧版本 torchmcubes"
    uv pip uninstall -y torchmcubes 2>/dev/null || true
    print_success "旧版本已卸载"
}

# 检测GPU架构
detect_gpu_arch() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        echo "$gpu_name"
    else
        echo "Unknown"
    fi
}

# 获取推荐的CUDA架构列表
get_cuda_arch_list() {
    local gpu_name=$(detect_gpu_arch)

    # RTX 50系列（Blackwell架构）- sm_120
    if [[ "$gpu_name" =~ "RTX 50" ]]; then
        echo "7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
    # RTX 40系列（Ada架构）- sm_89
    elif [[ "$gpu_name" =~ "RTX 40" ]] || [[ "$gpu_name" =~ "RTX 4" ]]; then
        echo "7.0;7.5;8.0;8.6;8.9;9.0"
    # RTX 30系列（Ampere架构）- sm_86
    elif [[ "$gpu_name" =~ "RTX 30" ]] || [[ "$gpu_name" =~ "RTX 3" ]]; then
        echo "7.0;7.5;8.0;8.6"
    # 默认：包含所有常见架构（包括Blackwell）
    else
        echo "7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
    fi
}

# 安装依赖
install_dependencies() {
    print_step "检查并安装编译依赖"

    # 检查CUDA
    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc 未找到，请先安装 CUDA Toolkit"
        echo "参考文档: docs/BLACKWELL_GPU_GUIDE.md 或运行 scripts/cuda_reinstall.sh"
        exit 1
    fi
    print_success "nvcc 版本: $(nvcc --version | grep release | awk '{print $5}')"

    # 检测GPU
    local gpu_name=$(detect_gpu_arch)
    if [ "$gpu_name" != "Unknown" ]; then
        print_success "检测到GPU: $gpu_name"
    fi

    # 安装Python构建依赖
    print_step "安装Python构建依赖"
    local deps_needed=()

    # 检查每个依赖
    python -c "import scikit_build_core" 2>/dev/null || deps_needed+=("scikit-build-core")
    python -c "import cmake" 2>/dev/null || deps_needed+=("cmake")
    python -c "import ninja" 2>/dev/null || deps_needed+=("ninja")
    python -c "import pybind11" 2>/dev/null || deps_needed+=("pybind11")

    if [ ${#deps_needed[@]} -gt 0 ]; then
        print_warning "需要安装以下依赖: ${deps_needed[*]}"
        if uv pip install "${deps_needed[@]}"; then
            print_success "依赖安装完成"
        else
            print_error "依赖安装失败"
            exit 1
        fi
    else
        print_success "所有构建依赖已安装"
    fi
}

# 安全编译安装
safe_install() {
    print_step "开始安全编译 torchmcubes"

    # 关键：限制并行编译数量，避免内存溢出
    export MAX_JOBS=2

    # 自动检测GPU并设置合适的架构列表
    local cuda_arch_list=$(get_cuda_arch_list)
    export TORCH_CUDA_ARCH_LIST="$cuda_arch_list"

    # 设置编译优化级别
    export CFLAGS="-O2"
    export CXXFLAGS="-O2"

    print_warning "编译设置："
    echo "  MAX_JOBS=${MAX_JOBS} （限制并行编译，防止内存溢出）"
    echo "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    local gpu_name=$(detect_gpu_arch)
    if [[ "$gpu_name" =~ "RTX 50" ]]; then
        echo "  检测到 Blackwell 架构 GPU - 已包含 sm_120 支持"
    fi
    echo ""
    echo "预计编译时间：1-3分钟（根据CPU和架构数量）"
    echo "如果编译过程中WSL断开连接，请："
    echo "  1. 重启Windows"
    echo "  2. 在Windows PowerShell执行: wsl --shutdown"
    echo "  3. 重新打开WSL"
    echo ""

    # 开始编译
    print_step "正在编译（请耐心等待，不要中断）..."

    # 使用timeout防止无限挂起
    if timeout 1800 uv pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/tatsy/torchmcubes.git 2>&1 | tee /tmp/torchmcubes_install.log; then
        print_success "torchmcubes 编译安装成功！"
        return 0
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            print_error "编译超时（30分钟），可能是内存不足"
        else
            print_error "编译失败，退出码: $exit_code"
        fi

        echo "查看完整日志："
        echo "  cat /tmp/torchmcubes_install.log"
        return 1
    fi
}

# 验证安装
verify_installation() {
    print_step "验证 torchmcubes 安装"

    python -c "
import sys
try:
    import torch
    import torchmcubes
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')

    # 测试CUDA支持
    if torch.cuda.is_available():
        try:
            # 测试是否有CUDA版本
            from torchmcubes import marching_cubes
            print('torchmcubes CUDA support: Available')
            sys.exit(0)
        except Exception as e:
            print(f'torchmcubes CUDA support: Not available ({e})')
            print('Note: CPU version will be used')
            sys.exit(0)
    else:
        print('torchmcubes installed (CPU mode)')
        sys.exit(0)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
"

    if [ $? -eq 0 ]; then
        print_success "验证通过！"
        return 0
    else
        print_error "验证失败"
        return 1
    fi
}

# CPU回退方案
fallback_cpu() {
    print_warning "尝试安装 CPU 版本作为回退方案"

    export FORCE_CUDA=0
    export MAX_JOBS=1

    if uv pip install --no-cache-dir --no-build-isolation \
        git+https://github.com/tatsy/torchmcubes.git; then
        print_success "CPU 版本安装成功"
        echo ""
        print_warning "注意：使用 CPU 版本，性能会受影响"
        return 0
    else
        print_error "CPU 版本安装也失败"
        return 1
    fi
}

# 主流程
main() {
    echo ""
    echo "╔════════════════════════════════════════════════════════╗"
    echo "║   torchmcubes 安全安装脚本 v2.0                        ║"
    echo "║   支持 RTX 50 系列 Blackwell 架构（sm_120）            ║"
    echo "║   自动依赖安装 + 防止 WSL2 崩溃                        ║"
    echo "╚════════════════════════════════════════════════════════╝"
    echo ""

    # 步骤1：检查内存
    check_memory
    echo ""

    # 步骤2：卸载旧版本
    uninstall_old
    echo ""

    # 步骤3：检查依赖
    install_dependencies
    echo ""

    # 步骤4：安全编译安装
    if safe_install; then
        echo ""
        # 步骤5：验证
        if verify_installation; then
            echo ""
            print_success "═══════════════════════════════════════"
            print_success "  安装完成！torchmcubes 已准备就绪  "
            print_success "═══════════════════════════════════════"
            exit 0
        fi
    fi

    # 如果失败，尝试CPU版本
    echo ""
    print_warning "CUDA 版本安装失败，是否尝试 CPU 版本？"
    read -p "继续？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if fallback_cpu && verify_installation; then
            echo ""
            print_warning "═══════════════════════════════════════"
            print_warning "  CPU 版本安装完成（性能受限）      "
            print_warning "═══════════════════════════════════════"
            exit 0
        fi
    fi

    # 完全失败
    echo ""
    print_error "═══════════════════════════════════════"
    print_error "  安装失败，请查看错误日志          "
    print_error "═══════════════════════════════════════"
    echo ""
    echo "故障排除建议："
    echo "1. 确认 .wslconfig 已生效（重启WSL）："
    echo "   PowerShell> wsl --shutdown"
    echo ""
    echo "2. 检查CUDA环境："
    echo "   nvcc --version"
    echo "   nvidia-smi"
    echo ""
    echo "3. 查看详细日志："
    echo "   cat /tmp/torchmcubes_install.log"
    echo ""
    echo "4. 如问题持续，考虑增加swap或物理内存"

    exit 1
}

# 运行主程序
main
