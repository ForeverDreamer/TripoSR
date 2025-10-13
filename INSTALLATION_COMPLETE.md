# TripoSR + RTX 5070 Ti 安装完成总结

## ✅ 安装状态：成功

**日期**：2025-10-13
**总耗时**：2 个会话（第一次：WSL2 配置，第二次：安装和 Blackwell 支持）

---

## 📊 系统配置

| 组件 | 规格 | 状态 |
|------|------|------|
| **CPU** | Intel i9-13900KF (24核) | ✅ 24核全部可用 |
| **内存** | 64GB DDR5 | ✅ WSL2 分配 49GB |
| **GPU** | RTX 5070 Ti (16GB) | ✅ Blackwell sm_120 支持 |
| **CUDA** | 12.8 | ✅ |
| **PyTorch** | 2.8.0+cu128 | ✅ 支持 sm_120 |
| **torchmcubes** | 0.1.0 | ✅ 编译包含 sm_120 |

---

## 🎯 解决的问题

### 问题 1：WSL2 编译时崩溃
**症状**：编译 torchmcubes 时 WSL2 断开连接，需要重启 Windows

**根本原因**：
- WSL2 默认内存 7.8GB（物理内存的 12%）
- CUDA 编译需要 10-15GB+ 内存

**解决方案**：
1. 创建 `.wslconfig` 配置文件：
   ```ini
   [wsl2]
   memory=50GB
   processors=24
   swap=20GB
   ```
2. 重启 WSL2：`wsl --shutdown`
3. 验证：`free -h` 显示 49GB

**结果**：✅ 编译时内存充足，无崩溃

---

### 问题 2：缺少构建依赖
**症状**：
```
ModuleNotFoundError: No module named 'scikit_build_core'
CMakeNotFoundError: Could not find CMake
```

**解决方案**：
```bash
uv pip install scikit-build-core cmake ninja pybind11
```

**自动化**：已集成到 `safe_install_torchmcubes.sh` v2.0

**结果**：✅ 依赖自动安装

---

### 问题 3：RTX 5070 Ti Blackwell 架构不支持（🔥 核心问题）
**症状**：
```
CUDA kernel failed : no kernel image is available for execution on the device
```

**根本原因**：
- RTX 5070 Ti 使用全新的 Blackwell 架构（计算能力 sm_120 / 12.0）
- 默认编译的 torchmcubes 不包含 sm_120 二进制
- 类似于 x86 程序无法在 ARM 上运行

**解决方案**：
```bash
# 设置包含 sm_120 的架构列表
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"

# 重新编译
uv pip uninstall torchmcubes
uv pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/tatsy/torchmcubes.git
```

**自动化**：`safe_install_torchmcubes.sh` v2.0 自动检测 GPU 并包含正确架构

**结果**：✅ CUDA 网格提取成功（1.19秒）

---

## 🚀 性能表现

### TripoSR 测试（chair.png）
```
模型初始化：3.5秒
图像处理：  0.37秒
模型推理：  0.67秒
网格提取：  1.19秒（CUDA sm_120）
网格导出：  0.08秒
----------------------------
总计：      ~5.4秒
```

**输出**：`output/0/mesh.obj`（4.4MB，高质量 3D 模型）

### 编译时间
- **首次编译**（sm_120 前）：1分23秒
- **Blackwell 重编译**：1分41秒
- **内存使用峰值**：~8GB（在 49GB 内安全）

---

## 📁 创建/更新的文件

### 脚本
1. **`scripts/safe_install_torchmcubes.sh` v2.0**
   - 自动检测 GPU 架构
   - 自动安装构建依赖
   - 默认包含 sm_120 支持
   - 使用：`./scripts/safe_install_torchmcubes.sh`

2. **`scripts/cuda_reinstall.sh`（新增）**
   - CUDA 12.8 完全重装
   - 包含用户提供的清理命令
   - 使用：`./scripts/cuda_reinstall.sh`

### 文档
1. **`docs/BLACKWELL_GPU_GUIDE.md`（新增）** ⭐
   - RTX 50 系列完整指南
   - Blackwell 架构详解
   - 故障排除和最佳实践
   - 53 页详细文档

2. **`SESSION_SUMMARY.md` v2.0**
   - 完整会话记录
   - 包含最终成功步骤
   - 问题解决过程

3. **`INSTALLATION_COMPLETE.md`（本文档）**
   - 简洁的最终总结

### 配置
- **`.wslconfig`**：WSL2 高性能配置（已生效）

---

## 🔑 关键经验

### 1. RTX 50 系列必备知识
| 要素 | 值 | 重要性 |
|-----|-----|--------|
| 架构名称 | Blackwell | - |
| 计算能力 | sm_120 (12.0) | ⭐⭐⭐ |
| CUDA 要求 | 12.8+ | ⭐⭐⭐ |
| PyTorch 要求 | cu128 | ⭐⭐⭐ |
| 编译参数 | `TORCH_CUDA_ARCH_LIST` 包含 `12.0` | ⭐⭐⭐ |

**记忆口诀**：RTX 50 = Blackwell = sm_120 = CUDA 12.8+

### 2. 架构不匹配的症状
- ✅ PyTorch 可以使用 GPU（`torch.cuda.is_available() = True`）
- ❌ 自定义 CUDA 扩展报错："no kernel image available"
- 原因：PyTorch 预编译包含 sm_120，但自己编译的扩展默认不包含

### 3. 通用 CUDA 扩展编译模板
```bash
# 适用于任何 CUDA Python 扩展
export MAX_JOBS=2  # 防止内存溢出
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
export CFLAGS="-O2"
export CXXFLAGS="-O2"

uv pip install --no-cache-dir --no-build-isolation <package>
```

### 4. GPU 架构演进
```
RTX 20 → Turing   → sm_75 (7.5)
RTX 30 → Ampere   → sm_86 (8.6)
RTX 40 → Ada      → sm_89 (8.9)
RTX 50 → Blackwell→ sm_120 (12.0) ← 跨代升级！
```

**注意**：Blackwell 是重大架构变更，类似于 Pascal → Volta 的跨越。

---

## 📝 快速命令参考

### 验证系统状态
```bash
# GPU 信息
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# CUDA 版本
nvcc --version

# PyTorch 架构支持
python -c "import torch; print(torch.cuda.get_arch_list())"

# 当前 GPU 架构
python -c "import torch; cap=torch.cuda.get_device_capability(); print(f'sm_{cap[0]}{cap[1]}')"

# WSL2 内存
free -h
```

### 从零开始完整安装（RTX 50 系列）
```bash
# 1. 配置 WSL2（Windows PowerShell）
# 编辑 C:\Users\<用户名>\.wslconfig
wsl --shutdown

# 2. 验证配置（WSL）
free -h  # 应显示 ~49GB

# 3. 安装 PyTorch cu128
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. 安装 torchmcubes（自动包含 sm_120）
./scripts/safe_install_torchmcubes.sh

# 5. 测试
python run.py examples/chair.png --output-dir output/
```

### 故障排除
```bash
# CUDA 环境损坏 → 完全重装
./scripts/cuda_reinstall.sh

# torchmcubes 架构问题 → 重新编译
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0;10.0;12.0"
uv pip uninstall torchmcubes
uv pip install --no-cache-dir --no-build-isolation git+https://github.com/tatsy/torchmcubes.git

# 查看编译日志
cat /tmp/torchmcubes_install.log
```

---

## 🎓 学到的教训

### 1. 新架构 GPU 的隐藏陷阱
- 最新 GPU 硬件发布时，软件生态需要时间跟进
- 预编译的包（如 PyTorch）会快速支持
- 自己编译的扩展需要手动指定新架构
- **必须显式设置 `TORCH_CUDA_ARCH_LIST`**

### 2. WSL2 内存管理
- WSL2 默认内存太保守（12%）
- 动态分配机制使大内存配置安全
- 必须重启才能生效（`wsl --shutdown`）

### 3. 错误信息解读
- "no kernel image available" = 架构不匹配
- "CMakeNotFoundError" = Python 依赖缺失（不是系统包）
- WSL2 崩溃无日志 = 内存不足

### 4. 最佳实践
- 使用自动化脚本（减少人为错误）
- 每次重大更改后立即验证
- 保留详细日志（`/tmp/torchmcubes_install.log`）
- 文档化特殊配置（如 Blackwell 要求）

---

## 🔗 相关资源

### 本地文档
- **RTX 50 系列**：`docs/BLACKWELL_GPU_GUIDE.md` ⭐
- **WSL2 优化**：`docs/WSL2_ML_OPTIMIZATION.md`
- **崩溃解决**：`docs/WSL2_CRASH_SOLUTION.md`
- **完整记录**：`SESSION_SUMMARY.md`

### 脚本工具
- **安装 torchmcubes**：`scripts/safe_install_torchmcubes.sh`
- **重装 CUDA**：`scripts/cuda_reinstall.sh`
- **性能监控**：`scripts/wsl_performance_monitor.sh`

### 外部链接
- [NVIDIA Blackwell 架构](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [PyTorch CUDA 安装](https://pytorch.org/get-started/locally/)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)

---

## ✅ 验证清单

安装成功的标志：

- [x] `free -h` 显示 ~49GB 内存
- [x] `nvidia-smi` 显示 RTX 5070 Ti
- [x] `nvcc --version` 显示 12.8
- [x] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [x] `python -c "import torch; print('sm_120' in str(torch.cuda.get_arch_list()))"` → True
- [x] `python -c "import torchmcubes; from torchmcubes import marching_cubes"` → 无错误
- [x] `python run.py examples/chair.png --output-dir output/` → 生成 mesh.obj
- [x] WSL2 编译时稳定，无崩溃

**全部通过** ✅

---

## 🎉 总结

### 问题根源（3 层）
1. **基础层**：WSL2 内存不足（7.8GB → 49GB）
2. **依赖层**：缺少构建工具（scikit-build-core、cmake、ninja、pybind11）
3. **架构层**：RTX 5070 Ti Blackwell sm_120 不兼容

### 解决方案（3 层对应）
1. **.wslconfig 配置** + `wsl --shutdown`
2. **自动安装依赖** → `safe_install_torchmcubes.sh` v2.0
3. **包含 sm_120 编译** → `TORCH_CUDA_ARCH_LIST="12.0"`

### 结果
- ✅ torchmcubes CUDA 编译成功（1-2 分钟）
- ✅ TripoSR 3D 重建正常运行（~5 秒/图）
- ✅ 系统稳定，性能优异
- ✅ 可复现的自动化流程

### 适用范围
本解决方案适用于：
- ✅ 所有 RTX 50 系列 GPU（5090/5080/5070 Ti/5070）
- ✅ WSL2 环境下的 CUDA 开发
- ✅ 任何需要编译 CUDA 扩展的 Python 项目
- ✅ 大模型训练和推理

---

**项目**：TripoSR 3D 重建
**GPU**：NVIDIA GeForce RTX 5070 Ti（Blackwell sm_120）
**环境**：WSL2 Ubuntu 22.04
**状态**：🎉 **完全成功** 🎉

---

**生成日期**：2025-10-13
**文档版本**：1.0
**作者**：Claude Code（Anthropic）
**测试环境**：Intel i9-13900KF + 64GB + RTX 5070 Ti + CUDA 12.8
