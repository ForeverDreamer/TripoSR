# WSL2 大模型训练与推理优化指南

## 📋 目录

- [概述](#概述)
- [硬件配置](#硬件配置)
- [WSL2 优化配置](#wsl2-优化配置)
- [性能对比](#性能对比)
- [最佳实践](#最佳实践)
- [性能监控](#性能监控)
- [常见问题](#常见问题)
- [高级优化](#高级优化)

---

## 概述

本文档提供了一套完整的 WSL2 优化方案，专门针对大模型训练和推理场景。通过合理配置资源分配和系统参数，可以在保证 Windows 稳定性的前提下，最大化 WSL2 的 AI/ML 工作负载性能。

### 为什么选择 WSL2？

**优势**：
- ✅ 原生 Linux 内核，完整的 CUDA 支持
- ✅ 无需双系统或虚拟机
- ✅ 与 Windows 无缝集成
- ✅ 支持 Docker 和容器化部署
- ✅ 便捷的文件系统访问
- ✅ **动态资源分配**，未使用资源可被 Windows 调用

**性能考虑**：
- ⚠️ GPU 密集任务相比原生 Linux 慢约 5-10%（虚拟化开销）
- ⚠️ 跨文件系统（/mnt/c）访问慢 30-50%
- ⚠️ 默认资源限制需要优化

---

## ⚡ WSL2 动态资源管理机制（重要！）

### 关键概念：动态分配 vs 静态预分配

**❓ 常见误解**：
> "配置 `memory=50GB` 后，Windows 是不是只剩 14GB 可用？"

**✅ 正确理解**：
- `memory=50GB` 是 WSL2 的**内存上限**，而非固定分配
- WSL2 采用**动态内存分配**，按实际需求使用
- **Windows 可以使用 WSL2 未调用的资源**

### 实际运行示例

#### 场景 1：WSL2 空闲时

```
系统总内存：    64GB
WSL2 配置上限：  50GB
WSL2 实际占用：  1.5GB  (仅基础系统)
Windows 可用：   62.5GB (64GB - 1.5GB)

结论：✅ Windows 几乎可以使用所有内存
```

#### 场景 2：运行大模型训练

```
系统总内存：    64GB
WSL2 配置上限：  50GB
WSL2 实际占用：  38GB  (训练中)
Windows 可用：   26GB  (64GB - 38GB)

结论：✅ Windows 仍有充足内存运行程序
```

#### 场景 3：训练完成后（启用 autoMemoryReclaim）

```
训练结束前：
  WSL2 占用：38GB
  Windows 可用：26GB

5分钟后（自动回收）：
  WSL2 占用：5GB ⬇️ (释放 33GB)
  Windows 可用：59GB ⬆️

结论：✅ 内存自动归还给 Windows
```

### 与传统虚拟机的对比

| 对比项 | 传统 VM (VirtualBox/VMware) | WSL2 |
|--------|----------------------------|------|
| 启动时内存占用 | 立即占用全部分配内存 | 仅占用几百 MB |
| 运行时内存管理 | 固定占用，不归还 | 动态增长，按需使用 |
| 空闲时内存 | 无法被主机使用 | 可被 Windows 使用 |
| 任务完成后 | 继续占用 | 自动回收（如启用） |
| CPU 调度 | 独占分配的核心 | 与 Windows 共享时间片 |

### autoMemoryReclaim 的作用

我们的配置使用了 `autoMemoryReclaim=gradual`，它的工作原理：

```ini
autoMemoryReclaim=gradual    # 推荐配置
```

**工作机制**：
1. **监控空闲**：检测 WSL2 CPU 空闲状态
2. **逐步释放**：空闲 60 秒后开始释放缓存内存
3. **归还 Windows**：未使用的内存返回给主机
4. **不影响训练**：训练时 CPU 不空闲，不会触发回收

**三种模式对比**：

| 模式 | 释放速度 | 适用场景 | 性能影响 |
|------|---------|---------|---------|
| `gradual` | 逐步释放 | **大模型训练**（推荐） | 最小 |
| `dropcache` | 立即释放 | 模型推理服务 | 中等 |
| `disabled` | 不自动释放 | 极限性能测试 | 无，但占用内存 |

### 实际验证方法

#### 在 Windows 中查看

```powershell
# 打开任务管理器 -> 性能 -> 内存
# 查看 "Vmmem" 进程的内存占用
# 这就是 WSL2 的实际内存使用量

# 或使用命令行
Get-Process vmmem | Select-Object WorkingSet64
```

#### 在 WSL2 中查看

```bash
# 查看 WSL2 内部视角的内存
free -h

# 示例输出（空闲时）：
#               total        used        free
# Mem:           50Gi        1.5Gi       48.5Gi  # 只用了 1.5GB

# 示例输出（训练时）：
#               total        used        free
# Mem:           50Gi         38Gi       12Gi    # 实际用了 38GB
```

### 关键要点总结

✅ **配置 memory=50GB 是安全的**
   - 这是上限保护，防止 WSL2 占用过多内存
   - 不会导致 Windows "只剩 14GB"

✅ **Windows 可以自由使用未被 WSL2 占用的内存**
   - WSL2 空闲时，Windows 几乎可使用全部 64GB
   - WSL2 工作时，双方动态平衡资源

✅ **autoMemoryReclaim 确保内存高效利用**
   - 训练完成后自动归还内存
   - 不影响训练性能
   - 提升整体系统响应速度

✅ **CPU 资源共享更灵活**
   - processors=24 不是独占
   - Hyper-V 智能调度 CPU 时间片
   - Windows 和 WSL2 同时运行无压力

---

## 硬件配置

### 当前系统配置

```
CPU:     Intel i9-13900KF
         - 24 核心 (8P + 16E)
         - 32 线程
         - 基频 3.0GHz, 睿频 5.8GHz

内存:    64GB DDR5
         - 双通道
         - 高频率支持

GPU:     NVIDIA GeForce RTX 5070 Ti
         - 16GB GDDR7 显存
         - CUDA 计算能力 8.9
         - 支持 CUDA 12.8+

存储:    NVMe SSD
         - 高速读写性能
```

### 性能分析

| 组件 | 性能等级 | 适用场景 |
|------|---------|---------|
| CPU | 旗舰级 | ✅ 大规模数据预处理、多进程训练 |
| 内存 | 充足 | ✅ 可加载 70B+ 参数模型（量化） |
| GPU | 高端 | ✅ 训练中等规模模型、快速推理 |
| 存储 | 优秀 | ✅ 快速数据加载、模型检查点保存 |

---

## WSL2 优化配置

### 资源分配策略

#### 设计原则

1. **最大化 WSL2 性能**：分配 78% 系统资源给 AI 训练
2. **保证 Windows 稳定**：保留 14GB 内存和 8 个线程
3. **防止 OOM**：配置充足的 Swap 空间
4. **自动内存管理**：启用 WSL 2.2.4+ 新特性

#### 推荐配置

文件位置：`C:\Users\YourUsername\.wslconfig`

```ini
[wsl2]

# ═══════════════════════════════════════════════════════════════
# 核心资源分配 (78% 资源给 WSL2)
# ═══════════════════════════════════════════════════════════════

memory=50GB              # 物理内存的 78% (保留 14GB 给 Windows)
processors=24            # 75% CPU 核心 (保留 8 个给 Windows)
swap=20GB                # 内存的 40%，防止大模型 OOM

# ═══════════════════════════════════════════════════════════════
# 内存管理优化 (WSL 2.2.4+ 新特性)
# ═══════════════════════════════════════════════════════════════

autoMemoryReclaim=gradual    # 逐步回收未使用内存
vmIdleTimeout=60000          # 空闲 60 秒后开始释放内存

# ═══════════════════════════════════════════════════════════════
# 存储优化
# ═══════════════════════════════════════════════════════════════

sparseVhd=true              # 自动压缩虚拟磁盘
pageReporting=true          # 提高内存分配效率
guessVhdSize=true           # 优化虚拟磁盘扩展

# ═══════════════════════════════════════════════════════════════
# 虚拟化和网络
# ═══════════════════════════════════════════════════════════════

nestedVirtualization=true   # 支持 Docker
localhostForwarding=true    # 端口转发（TensorBoard等）

# ═══════════════════════════════════════════════════════════════
# Linux 内核参数优化
# ═══════════════════════════════════════════════════════════════

kernelCommandLine = sysctl.vm.swappiness=10 sysctl.vm.max_map_count=262144 sysctl.vm.overcommit_memory=1 sysctl.fs.inotify.max_user_watches=524288 sysctl.kernel.pid_max=4194304
```

#### 应用配置

```powershell
# 在 Windows PowerShell 中执行（管理员权限）
wsl --shutdown

# 等待 8-10 秒后重新打开 WSL
```

#### 验证配置

```bash
# 在 WSL 中检查
free -h          # 应显示约 50GB 内存
nproc            # 应显示 24
swapon --show    # 应显示约 20GB swap
```

---

## 性能对比

### 配置前后对比

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 可用内存 | 7.8GB | 50GB | **541%** ⬆️ |
| CPU 核心 | 4 | 24 | **500%** ⬆️ |
| Swap 空间 | 2GB | 20GB | **900%** ⬆️ |
| 编译稳定性 | ❌ 崩溃 | ✅ 稳定 | - |
| 训练 batch size | 受限 | 灵活 | - |

### 实际性能表现

**大模型加载**：
- 7B 模型：✅ 完全加载到内存
- 13B 模型：✅ 完全加载到内存
- 33B 模型：✅ 可加载（可能使用部分 swap）
- 70B 模型：⚠️ 需要量化（INT8/INT4）

**训练性能**（相比原生 Linux）：
- GPU 计算：~95% 性能（5% 虚拟化损失）
- 数据加载：~90% 性能（I/O 开销）
- 整体吞吐量：~92% 性能

---

## 最佳实践

### 1. 文件系统优化

#### 规则：数据放在 WSL2 文件系统

```bash
# ✅ 推荐：数据集放在 WSL2 文件系统
~/datasets/
/home/user/models/

# ❌ 避免：数据集放在 Windows 分区
/mnt/c/datasets/           # 慢 30-50%
/mnt/d/models/             # 慢 30-50%
```

#### 数据迁移示例

```bash
# 从 Windows 迁移数据到 WSL2
cp -r /mnt/c/Users/YourName/datasets ~/datasets
# 或使用 rsync 显示进度
rsync -av --progress /mnt/c/Users/YourName/datasets ~/
```

### 2. Python 环境配置

#### 使用虚拟环境

```bash
# 推荐使用 uv (更快)
uv venv .venv
source .venv/bin/activate

# 或使用 conda
conda create -n ml python=3.11
conda activate ml
```

#### PyTorch 安装

```bash
# CUDA 12.8
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. 训练参数优化

#### Batch Size 调整

```python
# 根据显存动态调整
import torch

# RTX 5070 Ti 16GB 显存建议
batch_sizes = {
    'small_model': 64,      # <1B 参数
    'medium_model': 32,     # 1-7B 参数
    'large_model': 8,       # 7-13B 参数
    'xlarge_model': 2,      # 13B+ 参数
}

# 自动检测可用显存
def get_optimal_batch_size(model_size):
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    if gpu_mem < 8:
        return batch_sizes['small_model'] // 4
    elif gpu_mem < 16:
        return batch_sizes['medium_model'] // 2
    else:
        return batch_sizes['large_model']
```

#### 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

# 使用 FP16 减少显存占用
scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()

    with autocast():
        output = model(data)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### 梯度累积

```python
# 在显存不足时使用梯度累积模拟大 batch size
accumulation_steps = 4

for i, (data, target) in enumerate(dataloader):
    output = model(data)
    loss = criterion(output, target) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. 数据加载优化

#### DataLoader 配置

```python
from torch.utils.data import DataLoader

# 优化后的 DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,          # 使用多进程加载
    pin_memory=True,        # 加速 GPU 传输
    persistent_workers=True, # 保持 worker 进程
    prefetch_factor=2,      # 预取数据
)
```

### 5. 内存管理

#### 及时释放显存

```python
import torch
import gc

# 训练后清理显存
del model, optimizer
torch.cuda.empty_cache()
gc.collect()
```

#### 监控显存使用

```python
def print_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f'GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')

# 在训练循环中定期调用
print_gpu_memory()
```

---

## 性能监控

### 实时监控脚本

我们提供了专门的性能监控工具：

```bash
# 持续监控模式
./scripts/wsl_performance_monitor.sh

# 单次检查
./scripts/wsl_performance_monitor.sh -o

# 帮助信息
./scripts/wsl_performance_monitor.sh -h
```

### 监控指标

脚本会实时显示：
- ✅ CPU 使用率和负载
- ✅ 内存和 Swap 使用情况
- ✅ GPU 利用率和显存
- ✅ 磁盘空间
- ✅ 网络流量
- ✅ 训练进程状态
- ✅ 性能优化建议

### 手动监控命令

```bash
# 内存监控
watch -n 1 free -h

# GPU 监控
watch -n 1 nvidia-smi

# CPU 监控
htop

# 磁盘 I/O
iostat -x 1

# 进程监控
top -u $USER
```

---

## 常见问题

### Q0: 配置 memory=50GB 后，Windows 是不是只剩 14GB 可用了？

**答案：不是！这是最常见的误解。**

**详细说明**：
- `memory=50GB` 只是 WSL2 的**上限**，不是固定占用
- WSL2 使用动态内存分配，实际占用取决于工作负载
- **Windows 可以使用 WSL2 未使用的内存**

**实际情况**：
```
WSL2 空闲时：
  - WSL2 实际占用：1-2GB
  - Windows 可用：62-63GB ✅

WSL2 训练时：
  - WSL2 实际占用：30-40GB（视模型而定）
  - Windows 可用：24-34GB ✅

训练完成后（5-10分钟）：
  - WSL2 自动释放：回到 5GB 左右
  - Windows 可用：59GB ✅
```

**查看实际占用**：
- Windows：任务管理器 → 性能 → 查看 "Vmmem" 进程
- WSL2：`free -h` 命令查看 "used" 列

**结论**：配置大内存上限是安全的，不会"浪费"资源。

---

### Q1: 配置后 WSL 启动变慢？

**原因**：WSL2 需要预分配部分内存

**解决方案**：
- 首次启动等待 10-20 秒
- 后续启动会更快
- 使用 `autoMemoryReclaim=gradual` 优化

### Q2: Windows 响应变慢？

**原因**：WSL2 占用过多资源

**解决方案**：
```ini
# 降低资源分配
memory=40GB        # 从 50GB 降低
processors=20      # 从 24 降低
```

### Q3: 训练时出现 OOM？

**可能原因**：
1. Batch size 过大
2. 模型参数过多
3. 数据预处理占用内存

**解决方案**：
```bash
# 1. 降低 batch size
# 2. 使用梯度累积
# 3. 启用混合精度训练
# 4. 增加 swap 空间（临时方案）
```

### Q4: GPU 利用率低？

**可能原因**：
1. 数据加载是瓶颈
2. CPU 预处理过慢
3. 模型太小

**解决方案**：
```python
# 增加 DataLoader worker 数量
dataloader = DataLoader(
    dataset,
    num_workers=12,     # 增加到 12
    prefetch_factor=3,  # 增加预取
    pin_memory=True,
)

# 使用 GPU 数据增强
import kornia.augmentation as K
transforms = K.RandomHorizontalFlip()  # 在 GPU 上执行
```

### Q5: 编译大型项目时仍然崩溃？

**解决方案**：
```bash
# 进一步限制并行编译
export MAX_JOBS=1
export MAKEFLAGS="-j1"

# 临时增加 swap
# 修改 .wslconfig：swap=32GB
```

---

## 高级优化

### 1. CUDA 内存池优化

```python
import torch
import os

# 启用 CUDA 内存池
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# 或更激进的配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
```

### 2. cuDNN 自动调优

```python
import torch.backends.cudnn as cudnn

# 启用 cuDNN 自动调优（首次运行慢，后续更快）
cudnn.benchmark = True

# 确定性训练（可重现但稍慢）
cudnn.deterministic = True
cudnn.benchmark = False
```

### 3. 异步数据传输

```python
# 使用 non_blocking 加速数据传输
data = data.to(device, non_blocking=True)
target = target.to(device, non_blocking=True)
```

### 4. 模型并行

```python
# 对于超大模型，使用模型并行
from torch.nn.parallel import DataParallel

# 如果有多GPU
if torch.cuda.device_count() > 1:
    model = DataParallel(model)

# 或使用 Distributed Data Parallel (更快)
from torch.nn.parallel import DistributedDataParallel as DDP
```

### 5. 检查点优化

```python
# 使用压缩保存检查点
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'checkpoint.pt', _use_new_zipfile_serialization=True)
```

### 6. Profile 性能瓶颈

```python
from torch.profiler import profile, ProfilerActivity

# 分析性能
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    output = model(data)
    loss = criterion(output, target)
    loss.backward()

# 查看结果
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 7. WSL2 内核参数微调

```bash
# 在 WSL 中永久修改内核参数
sudo nano /etc/sysctl.conf

# 添加以下内容：
vm.swappiness=10
vm.max_map_count=262144
vm.overcommit_memory=1
fs.file-max=2097152

# 应用设置
sudo sysctl -p
```

### 8. Docker 优化（如使用容器）

```bash
# Docker 配置文件：~/.docker/daemon.json
{
  "data-root": "/home/user/docker",  # 放在 WSL2 文件系统
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

---

## 性能检查清单

训练前检查这些项目：

### 系统配置
- [ ] .wslconfig 已配置且生效（`wsl --shutdown` 后重启）
- [ ] 可用内存 ≥ 40GB (`free -h`)
- [ ] GPU 可用 (`nvidia-smi`)
- [ ] CUDA 版本正确 (`nvcc --version`)

### 环境设置
- [ ] Python 虚拟环境已激活
- [ ] PyTorch 已安装 CUDA 版本
- [ ] 数据集在 WSL2 文件系统（非 /mnt/c）
- [ ] 必要的库已安装

### 训练参数
- [ ] Batch size 适合显存
- [ ] 启用混合精度训练
- [ ] DataLoader num_workers 设置合理
- [ ] 监控脚本已运行

### 性能监控
- [ ] GPU 利用率 > 80%
- [ ] 显存使用 < 90%
- [ ] CPU 不是瓶颈
- [ ] 无 Swap 抖动

---

## 配置文件快速切换

### 不同场景配置

我们提供了多个预设配置：

1. **大模型训练** (当前)：50GB RAM, 24 CPU, 20GB Swap
2. **模型推理服务**：40GB RAM, 20 CPU, 16GB Swap
3. **日常开发**：32GB RAM, 16 CPU, 16GB Swap
4. **极限性能**：56GB RAM, 28 CPU, 24GB Swap

### 切换方法

```powershell
# Windows PowerShell
# 1. 备份当前配置
copy C:\Users\YourName\.wslconfig C:\Users\YourName\.wslconfig.backup

# 2. 复制新配置
copy path\to\preset\.wslconfig C:\Users\YourName\.wslconfig

# 3. 重启 WSL
wsl --shutdown
```

---

## 参考资源

### 官方文档
- [Microsoft WSL 文档](https://learn.microsoft.com/en-us/windows/wsl/)
- [NVIDIA CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [PyTorch 文档](https://pytorch.org/docs/stable/index.html)

### 性能调优
- [WSL2 性能最佳实践](https://learn.microsoft.com/en-us/windows/wsl/compare-versions#performance)
- [CUDA 性能优化](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PyTorch 性能调优](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

### 社区讨论
- [WSL GitHub Issues](https://github.com/microsoft/WSL/issues)
- [PyTorch Forums](https://discuss.pytorch.org/)

---

## 更新日志

### 2025-10-13 - v1.0
- ✅ 创建优化配置文件（50GB RAM, 24 CPU）
- ✅ 开发性能监控脚本
- ✅ 编写完整的最佳实践文档
- ✅ 提供多场景配置模板

---

## 贡献者

- 主要作者：Claude Code Analysis Team
- 测试平台：64GB RAM, i9-13900KF, RTX 5070 Ti
- 测试工作负载：大模型训练、推理、编译

---

## 许可证

本文档遵循 MIT License，可自由使用和分发。

---

**最后更新**：2025-10-13
**文档版本**：1.0
**适用系统**：Windows 11 + WSL2 (Ubuntu 22.04+)
