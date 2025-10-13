# CUDA Toolkit 手动安装指南

## 当前问题

TripoSR 需要 `torchmcubes` 库来生成 3D 网格。该库需要 CUDA toolkit 才能编译 CUDA 版本。

**当前状态**:
- ✅ PyTorch 2.5.1+cu121 (GPU 加速) 已安装
- ✅ 所有其他依赖已安装
- ❌ torchmcubes 未安装 (需要 CUDA toolkit)

## 手动安装 CUDA Toolkit 12.6

请按顺序执行以下命令：

### 步骤 1: 下载 CUDA 仓库密钥

```bash
cd /home/doer/repos/TripoSR
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
```

### 步骤 2: 安装仓库密钥 (需要 sudo)

```bash
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```

### 步骤 3: 更新包列表 (需要 sudo)

```bash
sudo apt-get update
```

### 步骤 4: 安装 CUDA Toolkit 12.6 (需要 sudo)

这一步会下载约 2-3GB 数据，需要 5-10 分钟：

```bash
sudo apt-get install -y cuda-toolkit-12-6
```

### 步骤 5: 清理下载的 deb 文件

```bash
rm cuda-keyring_1.1-1_all.deb
```

### 步骤 6: 配置环境变量

将 CUDA 添加到你的 PATH：

```bash
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 步骤 7: 验证 CUDA 安装

```bash
nvcc --version
```

你应该看到类似这样的输出：
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.6, ...
```

### 步骤 8: 重新安装 TripoSR 以编译 torchmcubes

现在 CUDA toolkit 已经安装，我们需要重新编译 torchmcubes：

```bash
cd /home/doer/repos/TripoSR

# 删除现有虚拟环境
rm -rf .venv

# 重新运行安装脚本
./install_triposr.sh
```

这次 torchmcubes 应该能成功编译 CUDA 版本。

## 一键复制命令

如果你想快速执行所有命令，可以复制这个完整的命令序列：

```bash
# 下载和安装 CUDA toolkit
cd /home/doer/repos/TripoSR && \
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb && \
sudo dpkg -i cuda-keyring_1.1-1_all.deb && \
sudo apt-get update && \
sudo apt-get install -y cuda-toolkit-12-6 && \
rm cuda-keyring_1.1-1_all.deb && \
echo 'export PATH=/usr/local/cuda-12.6/bin:$PATH' >> ~/.bashrc && \
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc && \
source ~/.bashrc && \
nvcc --version
```

然后：

```bash
# 重新安装 TripoSR
cd /home/doer/repos/TripoSR && \
rm -rf .venv && \
./install_triposr.sh
```

## 验证完整安装

安装完成后，测试 TripoSR：

```bash
cd /home/doer/repos/TripoSR
source .venv/bin/activate

# 检查所有依赖
python -c "
import torch
import torchmcubes
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('torchmcubes imported successfully')
"

# 运行示例
python run.py examples/chair.png --output-dir output/
```

## 如果 CUDA 安装失败

如果上述步骤失败，可能的原因：

1. **网络问题**: CUDA toolkit 下载需要稳定的网络连接
2. **磁盘空间**: 确保有至少 5GB 可用空间
3. **WSL 版本**: 确保使用 WSL2 (不是 WSL1)

检查 WSL 版本：
```bash
wsl --version
```

## 替代方案：仅使用 CPU 版本的 torchmcubes

如果你不想安装 CUDA toolkit，理论上可以强制安装 CPU 版本，但这会导致 PyTorch 运行在 GPU 而网格生成在 CPU，性能会大幅下降。

**不推荐此方案**，因为你有强大的 RTX 5070 Ti GPU。

## 估计时间

- 下载 CUDA toolkit: 5-10 分钟 (取决于网络速度)
- 安装 CUDA toolkit: 2-3 分钟
- 重新安装 TripoSR: 3-5 分钟
- **总计**: 约 10-18 分钟

## 磁盘空间需求

- CUDA toolkit: 约 3-4GB
- TripoSR 虚拟环境: 约 2-3GB
- **总计**: 约 5-7GB

## 下一步

1. 执行上述命令安装 CUDA toolkit
2. 重新运行 `./install_triposr.sh`
3. 测试 TripoSR

完成后，你将拥有完整的 GPU 加速 3D 重建系统！
