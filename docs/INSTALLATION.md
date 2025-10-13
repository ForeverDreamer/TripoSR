# TripoSR 安装和配置指南

本指南提供了在 WSL2 环境中使用 uv 和 Python 3.11 安装配置 TripoSR 的完整流程。

## 目录
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [详细步骤](#详细步骤)
- [使用方法](#使用方法)
- [故障排除](#故障排除)

## 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU with 至少 6GB VRAM (推荐 8GB+)
  - 支持 CUDA 11.x 或 12.x
  - CPU 模式可用但速度会慢得多
- **内存**: 至少 8GB RAM (推荐 16GB+)
- **存储**: 至少 10GB 可用空间 (用于模型和依赖)

### 软件要求
- **操作系统**: WSL2 (Windows Subsystem for Linux 2)
  - Ubuntu 20.04+ 或其他兼容的 Linux 发行版
- **Python**: 3.8+ (本指南使用 Python 3.11)
- **CUDA**: 11.x 或 12.x (如果使用 GPU)
- **NVIDIA 驱动**: 在 Windows 主机上安装最新的 NVIDIA 驱动

## 快速开始

### 1. 检查系统信息

运行系统信息收集脚本来验证你的环境：

```bash
cd /home/doer/repos/TripoSR
./check_system_info.sh
```

这个脚本会检查：
- WSL 版本和配置
- Windows 主机信息
- NVIDIA GPU 和 CUDA 版本
- Python 版本
- uv 包管理器

### 2. 安装 TripoSR

运行安装脚本：

```bash
./install_triposr.sh
```

这个脚本会：
1. 检查并安装 uv (如果未安装)
2. 检测 CUDA 版本
3. 创建 Python 3.11 虚拟环境
4. 安装匹配的 PyTorch 版本
5. 安装所有 TripoSR 依赖
6. 验证安装

安装过程大约需要 5-10 分钟，取决于网络速度。

### 3. 测试安装

激活虚拟环境并运行示例：

```bash
source .venv/bin/activate
python run.py examples/chair.png --output-dir output/
```

## 详细步骤

### 步骤 1: 准备 WSL2 环境

确保你的 WSL2 已正确安装并配置：

```bash
# 检查 WSL 版本
wsl --version

# 检查 NVIDIA GPU 支持
nvidia-smi
```

如果 `nvidia-smi` 不工作，你可能需要：
1. 更新 Windows 到最新版本
2. 安装最新的 NVIDIA 驱动
3. 确保 WSL2 内核支持 GPU

### 步骤 2: 系统信息检查

```bash
./check_system_info.sh
```

**关键检查项**:
- ✓ NVIDIA GPU 检测到 CUDA 支持
- ✓ Python 3.11 可用
- ✓ uv 包管理器已安装
- ✓ 足够的内存 (至少 8GB)

### 步骤 3: 运行安装脚本

```bash
./install_triposr.sh
```

**安装过程**:
1. **uv 安装**: 如果未安装，自动下载并安装 uv
2. **CUDA 检测**: 自动检测 CUDA 版本并选择匹配的 PyTorch
3. **虚拟环境**: 创建隔离的 Python 3.11 环境
4. **PyTorch 安装**: 安装支持 CUDA 的 PyTorch
   - CUDA 12.x → PyTorch with CUDA 12.1
   - CUDA 11.x → PyTorch with CUDA 11.8
   - 无 GPU → CPU-only PyTorch
5. **依赖安装**: 安装所有 requirements.txt 中的包
6. **验证**: 测试所有导入和 CUDA 功能

### 步骤 4: 验证安装

安装完成后，脚本会显示摘要信息：

```
========================================
TripoSR has been successfully installed!
========================================

System Information:
  Python: Python 3.11.x
  PyTorch: 2.x.x+cuXXX
  CUDA: Available (12.x)
  GPU: NVIDIA GeForce RTX XXXX
```

## 使用方法

### 基本使用

#### 1. 激活虚拟环境

```bash
cd /home/doer/repos/TripoSR
source .venv/bin/activate
```

#### 2. 单图像重建

```bash
python run.py path/to/image.png --output-dir output/
```

#### 3. 多图像批处理

```bash
python run.py image1.png image2.png image3.png --output-dir output/
```

#### 4. 使用辅助脚本

为了方便，已创建 `run_triposr.sh` 辅助脚本：

```bash
./run_triposr.sh examples/chair.png --output-dir output/
```

这个脚本会自动激活虚拟环境并运行 TripoSR。

### 高级选项

#### 纹理烘焙

生成带有纹理贴图的 3D 模型：

```bash
python run.py image.png --output-dir output/ \
  --bake-texture \
  --texture-resolution 1024
```

纹理分辨率选项：512, 1024, 2048, 4096

#### Gradio Web 界面

启动交互式 Web 应用：

```bash
python gradio_app.py
```

然后在浏览器中打开显示的 URL (通常是 `http://127.0.0.1:7860`)。

#### 查看所有选项

```bash
python run.py --help
```

### 输出格式

默认输出格式：
- **GLB**: Binary glTF 格式 (推荐，支持大多数 3D 软件)
- **OBJ**: 带顶点颜色或纹理的传统格式
- **Mesh**: 包含几何数据的 3D 模型

## 故障排除

### 常见问题

#### 1. `torchmcubes` CUDA 错误

**错误信息**:
```
AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'
```

**解决方案**:
```bash
source .venv/bin/activate
pip uninstall torchmcubes
pip install --upgrade setuptools
pip install git+https://github.com/tatsy/torchmcubes.git
```

确保：
- PyTorch CUDA 版本与系统 CUDA 版本匹配
- setuptools >= 49.6.0

#### 2. CUDA 版本不匹配

**症状**: PyTorch 检测不到 CUDA

**检查**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**解决方案**:
```bash
# 检查系统 CUDA 版本
nvidia-smi

# 卸载当前 PyTorch
pip uninstall torch torchvision

# 安装匹配的版本
# 对于 CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 对于 CUDA 11.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 3. 内存不足 (OOM)

**错误信息**:
```
CUDA out of memory
```

**解决方案**:
1. 关闭其他使用 GPU 的程序
2. 减小输入图像尺寸
3. 尝试 CPU 模式：
   ```bash
   python run.py image.png --device cpu --output-dir output/
   ```

#### 4. 模型下载失败

**症状**: 首次运行时模型下载失败

**解决方案**:
```bash
# 设置 HuggingFace 镜像 (如果在中国)
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载模型
huggingface-cli download stabilityai/TripoSR --local-dir ./models/TripoSR
```

#### 5. WSL 中 nvidia-smi 不工作

**解决方案**:
1. 确保 Windows 有最新的 NVIDIA 驱动
2. 更新 WSL2 内核：
   ```bash
   wsl --update
   ```
3. 重启 WSL：
   ```bash
   wsl --shutdown
   # 然后重新打开 WSL 终端
   ```

### 性能优化

#### GPU 加速

确保使用 GPU：
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

#### 批处理优化

处理多个图像时，可以调整批大小（如果有足够的 VRAM）。

## 脚本说明

### check_system_info.sh

检查系统配置和兼容性的脚本。

**功能**:
- WSL 和 Linux 发行版信息
- Windows 主机详细信息（通过 PowerShell）
- NVIDIA GPU 和 CUDA 版本检测
- Python 环境检查
- 内存和磁盘空间
- 安装建议和警告

**使用**:
```bash
./check_system_info.sh
```

### check_system_info.ps1

Windows PowerShell 版本的系统检查脚本。

**从 Windows PowerShell 运行**:
```powershell
cd C:\path\to\TripoSR
.\check_system_info.ps1
```

### install_triposr.sh

自动安装和配置 TripoSR 的脚本。

**功能**:
- 安装/检查 uv 包管理器
- 检测 CUDA 版本
- 创建 Python 3.11 虚拟环境
- 安装匹配的 PyTorch
- 安装所有依赖
- 验证安装
- 创建辅助脚本

**使用**:
```bash
./install_triposr.sh
```

**重新安装**:
```bash
# 删除虚拟环境
rm -rf .venv

# 重新运行安装
./install_triposr.sh
```

### run_triposr.sh

便捷的启动脚本，自动激活环境。

**使用**:
```bash
./run_triposr.sh image.png --output-dir output/
```

等同于：
```bash
source .venv/bin/activate
python run.py image.png --output-dir output/
```

## 参考资源

- **TripoSR GitHub**: https://github.com/VAST-AI-Research/TripoSR
- **TripoSR 论文**: https://arxiv.org/abs/2403.02151
- **Hugging Face Demo**: https://huggingface.co/spaces/stabilityai/TripoSR
- **Hugging Face Model**: https://huggingface.co/stabilityai/TripoSR
- **uv 文档**: https://docs.astral.sh/uv/
- **PyTorch 安装指南**: https://pytorch.org/get-started/locally/

## 技术规格

### 模型信息
- **模型大小**: ~1.5GB
- **架构**: Large Reconstruction Model (LRM)
- **输入**: 单张 RGB 图像
- **输出**: 3D 网格 (GLB/OBJ)
- **推理时间**: <0.5 秒 (NVIDIA A100), ~2-5 秒 (消费级 GPU)

### 系统需求摘要
| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU VRAM | 6GB | 8GB+ |
| 系统 RAM | 8GB | 16GB+ |
| CUDA | 11.x | 12.x |
| Python | 3.8 | 3.11 |
| 存储 | 5GB | 10GB+ |

## 许可证

TripoSR 在 MIT 许可证下发布。详见仓库中的 LICENSE 文件。

## 支持

如有问题或需要帮助：
1. 查看本文档的故障排除部分
2. 检查 [GitHub Issues](https://github.com/VAST-AI-Research/TripoSR/issues)
3. 加入 [Discord 社区](https://discord.gg/mvS9mCfMnQ)

---

**最后更新**: 2025-10-13
**版本**: 1.0.0
