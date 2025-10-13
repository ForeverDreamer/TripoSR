# RTX 50 系列 GPU（Blackwell 架构）完整指南

## 📌 快速概览

**适用范围**：RTX 5090, RTX 5080, RTX 5070 Ti, RTX 5070 等所有 Blackwell 架构 GPU

**核心问题**：RTX 50 系列使用全新的 Blackwell 架构（计算能力 sm_120），大多数现有的 CUDA 扩展默认不支持此架构，需要特殊配置。

**关键要求**：
- CUDA 12.8+
- PyTorch 2.7+ with cu128
- 所有 CUDA 扩展必须针对 sm_120 编译

---

## 🎯 核心问题说明

### 什么是 Blackwell 架构？

Blackwell 是 NVIDIA 2025 年发布的最新 GPU 架构，相比前代：
- **计算能力**：sm_120（12.0）
- **前代架构**：Ada (sm_89), Ampere (sm_86)
- **特点**：全新的流处理器设计，不兼容旧的 CUDA 二进制

### 为什么会出现错误？

**典型错误信息**：
```
CUDA kernel failed : no kernel image is available for execution on the device
```

或：
```
NVIDIA GeForce RTX 5070 Ti with CUDA capability sm_120 is not compatible
with the current PyTorch installation.
```

**原因**：
1. CUDA 扩展编译时未包含 sm_120 架构
2. PyTorch 版本过旧，不支持 Blackwell
3. CUDA Toolkit 版本低于 12.8

---

## ✅ 完整解决方案

### 步骤 1：验证系统要求

#### 1.1 检查 GPU
```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

**预期输出**：
```
name, compute_cap
NVIDIA GeForce RTX 5070 Ti, 12.0
```

如果 compute_cap 是 12.0，说明是 Blackwell 架构。

#### 1.2 检查 CUDA 版本
```bash
nvcc --version
```

**要求**：CUDA 12.8 或更高

**如果版本过低**：
```bash
# 使用重装脚本
./scripts/cuda_reinstall.sh

# 或手动安装
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8
```

#### 1.3 配置环境变量
```bash
# 添加到 .bashrc
echo 'export PATH=/usr/local/cuda-12.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 2：安装正确的 PyTorch

#### 2.1 卸载旧版本（如有）
```bash
uv pip uninstall torch torchvision torchaudio
```

#### 2.2 安装 PyTorch cu128 版本
```bash
# 稳定版（推荐）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 或 Nightly 版本（最新特性）
uv pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

#### 2.3 验证 PyTorch
```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Supported architectures: {torch.cuda.get_arch_list()}')
"
```

**预期输出**：
```
PyTorch: 2.8.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti
Supported architectures: ['sm_70', 'sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

**关键**：`sm_120` 必须在支持列表中！

### 步骤 3：安装 torchmcubes（针对 Blackwell）

#### 3.1 使用自动化脚本（推荐）
```bash
cd /home/doer/repos/TripoSR
./scripts/safe_install_torchmcubes.sh
```

该脚本会：
- 自动检测 RTX 5070 Ti
- 包含 sm_120 架构编译
- 自动安装所有依赖

#### 3.2 手动安装（如果脚本失败）
```bash
# 安装构建依赖
uv pip install scikit-build-core cmake ninja pybind11

# 设置环境变量
export MAX_JOBS=2
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
export CFLAGS="-O2"
export CXXFLAGS="-O2"

# 卸载旧版本
uv pip uninstall torchmcubes

# 编译安装
uv pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/tatsy/torchmcubes.git
```

**关键点**：`TORCH_CUDA_ARCH_LIST` 必须包含 `12.0`（对应 sm_120）

#### 3.3 验证安装
```bash
python -c "
import torch
import torchmcubes
from torchmcubes import marching_cubes

print('torchmcubes 安装成功！')
print(f'CUDA support: {torch.cuda.is_available()}')

# 快速测试
if torch.cuda.is_available():
    print('正在测试 CUDA 功能...')
    # 创建测试数据
    voxels = torch.randn(32, 32, 32).cuda()
    verts, faces = marching_cubes(voxels, 0.0)
    print(f'测试成功！生成 {len(verts)} 个顶点')
"
```

### 步骤 4：测试 TripoSR
```bash
cd /home/doer/repos/TripoSR
python run.py examples/chair.png --output-dir output/
```

**成功标志**：
```
2025-10-13 22:12:43,423 - INFO - Extracting mesh ...
2025-10-13 22:12:43,506 - INFO - Exporting mesh finished in 83.20ms.
```

输出文件：`output/0/mesh.obj`

---

## 🔧 常见问题

### 问题 1：编译时出现 "no kernel image available"

**原因**：torchmcubes 未针对 sm_120 编译

**解决**：
```bash
# 确保包含 sm_120
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
uv pip uninstall torchmcubes
uv pip install --no-cache-dir --no-build-isolation git+https://github.com/tatsy/torchmcubes.git
```

### 问题 2：PyTorch 不支持 sm_120

**检查**：
```bash
python -c "import torch; print(torch.cuda.get_arch_list())"
```

**如果没有 sm_120**：
```bash
# 重装 PyTorch cu128
uv pip uninstall torch torchvision torchaudio
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 问题 3：nvcc 版本低于 12.8

**解决**：
```bash
# 使用完全重装脚本
./scripts/cuda_reinstall.sh
```

### 问题 4：编译时缺少 CMake、pybind11 等依赖

**症状**：
```
ModuleNotFoundError: No module named 'scikit_build_core'
CMakeNotFoundError: Could not find CMake with version >=3.15
Could not find a package configuration file provided by "pybind11"
```

**解决**：
```bash
uv pip install scikit-build-core cmake ninja pybind11
```

`safe_install_torchmcubes.sh` v2.0 会自动安装这些依赖。

### 问题 5：WSL2 在编译时崩溃

**原因**：内存不足

**解决**：
1. 配置 WSL2 内存（参考 `docs/WSL2_ML_OPTIMIZATION.md`）
2. 编辑 `C:\Users\<用户名>\.wslconfig`：
```ini
[wsl2]
memory=50GB
processors=24
swap=20GB
```
3. 重启 WSL2：
```powershell
# Windows PowerShell
wsl --shutdown
```

---

## 📊 架构对照表

| GPU 系列 | 架构名称 | 计算能力 | CUDA 要求 | PyTorch 要求 |
|---------|---------|---------|----------|--------------|
| RTX 50xx | Blackwell | sm_120 (12.0) | 12.8+ | 2.7+ cu128 |
| RTX 40xx | Ada | sm_89 (8.9) | 11.8+ | 2.0+ cu118 |
| RTX 30xx | Ampere | sm_86 (8.6) | 11.0+ | 1.7+ cu110 |
| RTX 20xx | Turing | sm_75 (7.5) | 10.0+ | 1.0+ cu100 |

**注意**：每个架构使用不同的 CUDA 二进制格式，必须针对目标架构编译！

---

## 🚀 最佳实践

### 1. 编译 CUDA 扩展的通用模板

对于任何需要 CUDA 的 Python 包：

```bash
# 设置通用环境变量
export MAX_JOBS=2  # 限制并行编译，避免内存溢出
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
export CFLAGS="-O2"
export CXXFLAGS="-O2"

# 清理缓存
uv cache clean

# 编译安装
uv pip install --no-cache-dir --no-build-isolation <package>
```

### 2. 验证 CUDA 扩展的架构支持

```bash
# 查看已安装包的架构支持
python -c "
import torch
print('PyTorch 支持的架构:', torch.cuda.get_arch_list())

# 检查当前 GPU 架构
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    sm = f'sm_{cap[0]}{cap[1]}'
    print(f'当前 GPU 架构: {sm}')

    if sm in torch.cuda.get_arch_list():
        print('✓ PyTorch 支持当前 GPU')
    else:
        print('✗ PyTorch 不支持当前 GPU - 需要重装！')
"
```

### 3. 项目环境检查清单

在开始任何 CUDA 项目前：

```bash
# 创建检查脚本
cat > check_cuda_env.sh << 'EOF'
#!/bin/bash
echo "=== CUDA 环境检查 ==="
echo ""

echo "1. GPU 信息："
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv
echo ""

echo "2. CUDA 版本："
nvcc --version | grep release
echo ""

echo "3. PyTorch 信息："
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}'); print(f'支持的架构: {torch.cuda.get_arch_list()}')"
echo ""

echo "4. GPU 可用性："
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"N/A\"}')"
EOF

chmod +x check_cuda_env.sh
./check_cuda_env.sh
```

---

## 🔗 相关资源

### 官方文档
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)

### 本项目文档
- [`SESSION_SUMMARY.md`](../SESSION_SUMMARY.md) - 完整问题解决过程
- [`WSL2_ML_OPTIMIZATION.md`](WSL2_ML_OPTIMIZATION.md) - WSL2 大模型训练优化
- [`WSL2_CRASH_SOLUTION.md`](WSL2_CRASH_SOLUTION.md) - WSL2 崩溃问题解决

### 脚本工具
- [`scripts/safe_install_torchmcubes.sh`](../scripts/safe_install_torchmcubes.sh) - torchmcubes 自动化安装
- [`scripts/cuda_reinstall.sh`](../scripts/cuda_reinstall.sh) - CUDA 完全重装
- [`scripts/wsl_performance_monitor.sh`](../scripts/wsl_performance_monitor.sh) - 性能监控

---

## 💡 总结

### RTX 50 系列核心要点

1. **必须使用 CUDA 12.8+**
2. **必须使用 PyTorch cu128 版本**
3. **所有 CUDA 扩展必须包含 sm_120 编译**
4. **编译时设置 `TORCH_CUDA_ARCH_LIST` 包含 `12.0`**

### 快速命令参考

```bash
# 完整安装流程（从零开始）
./scripts/cuda_reinstall.sh                    # 1. 重装 CUDA 12.8
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # 2. 安装 PyTorch
./scripts/safe_install_torchmcubes.sh          # 3. 安装 torchmcubes
python run.py examples/chair.png --output-dir output/  # 4. 测试

# 验证命令
nvidia-smi                                      # GPU 状态
nvcc --version                                  # CUDA 版本
python -c "import torch; print(torch.cuda.get_arch_list())"  # PyTorch 架构支持
```

### 故障排除优先级

1. **确认 GPU 架构**：`nvidia-smi --query-gpu=compute_cap --format=csv`
2. **确认 CUDA 版本**：`nvcc --version` ≥ 12.8
3. **确认 PyTorch 版本**：`python -c "import torch; print(torch.version.cuda)"` = 12.8
4. **确认架构支持**：`python -c "import torch; print('sm_120' in str(torch.cuda.get_arch_list()))"`
5. **重新编译扩展**：包含 `TORCH_CUDA_ARCH_LIST="12.0"`

---

**最后更新**：2025-10-13
**文档版本**：1.0
**测试环境**：RTX 5070 Ti, CUDA 12.8, PyTorch 2.8.0+cu128, WSL2 Ubuntu 22.04
