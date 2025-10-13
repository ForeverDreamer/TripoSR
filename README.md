# TripoSR - Fast 3D Object Reconstruction

<div align="center">

[![](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![](https://img.shields.io/badge/CUDA-11.8+-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[GitHub](https://github.com/VAST-AI-Research/TripoSR) | [Paper](https://arxiv.org/abs/2403.02151) | [HuggingFace](https://huggingface.co/stabilityai/TripoSR) | [Demo](https://huggingface.co/spaces/stabilityai/TripoSR)

</div>

**TripoSR** 是一个快速的前馈式3D重建模型，能从单张图像生成高质量3D模型，推理速度小于0.5秒（NVIDIA A100）。

## ✨ 特性

- 🚀 **极速推理**: RTX 3080 约 2-4秒/图像
- 🎯 **高质量输出**: 优于其他开源方案
- 💼 **易于使用**: 一键安装，简单API
- 🔧 **灵活配置**: 支持纹理烘焙、批处理等
- 🌐 **Web界面**: 内置Gradio交互界面

## 📦 快速开始

### 一键安装

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
./setup.sh
```

### 基本使用

```bash
# 激活环境
source .venv/bin/activate

# 单图像重建
python run.py examples/chair.png --output-dir output/

# 启动Web界面
python gradio_app.py
```

详细说明见 [快速开始指南](docs/QUICK_START.md)

## 📋 系统要求

### 最低要求
- **GPU**: NVIDIA GPU with 6GB+ VRAM（或CPU，但会很慢）
- **内存**: 8GB+ RAM
- **Python**: 3.8+
- **CUDA**: 11.8+ (GPU模式)
- **磁盘**: 10GB+ 可用空间

### 推荐配置
- **GPU**: RTX 3080 / RTX 4080 或更高
- **内存**: 16GB+ RAM
- **Python**: 3.11
- **CUDA**: 12.x

## 🎮 使用示例

### 命令行

```bash
# 基础用法
python run.py image.png --output-dir output/

# 批量处理
python run.py img1.png img2.png img3.png --output-dir output/

# 生成纹理
python run.py image.png --bake-texture --texture-resolution 1024 --output-dir output/

# 使用CPU
python run.py image.png --device cpu --output-dir output/

# 自定义分辨率
python run.py image.png --mc-resolution 512 --output-dir output/
```

### Python API

```python
import torch
from tsr.system import TSR
from PIL import Image

# 加载模型
model = TSR.from_pretrained("stabilityai/TripoSR", device="cuda:0")

# 加载图像
image = Image.open("examples/chair.png")

# 生成3D模型
with torch.no_grad():
    scene_codes = model([image])
    meshes = model.extract_mesh(scene_codes)

# 保存
meshes[0].export("output/mesh.obj")
```

## 📚 文档

| 文档 | 说明 |
|------|------|
| [快速开始](docs/QUICK_START.md) | 5分钟上手指南 |
| [安装指南](docs/INSTALLATION.md) | 详细安装步骤 |
| [CUDA安装](docs/CUDA_INSTALL.md) | CUDA Toolkit安装 |
| [故障排除](docs/TROUBLESHOOTING.md) | 常见问题解决方案 |
| [原始README](docs/ORIGINAL_README.md) | 官方原始文档 |

## 🛠️ 项目结构

```
TripoSR/
├── setup.sh                    # 一键安装脚本
├── run.py                      # 命令行工具
├── gradio_app.py              # Web界面
├── requirements.txt
│
├── docs/                       # 文档目录
│   ├── QUICK_START.md
│   ├── INSTALLATION.md
│   ├── CUDA_INSTALL.md
│   └── TROUBLESHOOTING.md
│
├── scripts/                    # 工具脚本
│   ├── check_system.sh        # 系统检查
│   ├── install.sh             # 安装脚本
│   └── utils/
│       └── common.sh          # 通用函数库
│
├── tsr/                        # 核心代码
└── examples/                   # 示例图像
```

## 🔧 高级配置

### 环境变量

```bash
# Python版本
export PYTHON_VERSION=3.11

# 虚拟环境名称
export VENV_NAME=.venv

# HuggingFace镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com
```

### 配置选项

```bash
# 自定义Python版本
./setup.sh --python 3.10

# 跳过系统检查
./setup.sh --skip-check

# 自动确认
./setup.sh -y
```

## 📊 性能基准

| GPU | 分辨率 | 速度 | VRAM |
|-----|--------|------|------|
| A100 | 256 | ~0.5s | 6GB |
| RTX 4090 | 256 | ~1-2s | 6GB |
| RTX 3080 | 256 | ~2-4s | 6GB |
| RTX 2080 | 256 | ~4-6s | 6GB |
| CPU | 256 | ~60-120s | - |

## 🐛 故障排除

### 常见问题

1. **GPU未检测到**: 检查NVIDIA驱动和CUDA安装
2. **torchmcubes错误**: 需要CUDA toolkit编译支持
3. **内存不足**: 降低 `--mc-resolution` 或使用CPU

详细解决方案见 [故障排除文档](docs/TROUBLESHOOTING.md)

### 系统检查

```bash
bash scripts/check_system.sh
```

## 🤝 贡献

欢迎贡献！请参考[原始README](docs/ORIGINAL_README.md)了解项目详情。

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [Tripo AI](https://www.tripo3d.ai/)
- [Stability AI](https://stability.ai/)
- [Large Reconstruction Model (LRM)](https://yiconghong.me/LRM/)

## 📞 支持

- **GitHub Issues**: [提交问题](https://github.com/VAST-AI-Research/TripoSR/issues)
- **Discord**: [加入社区](https://discord.gg/mvS9mCfMnQ)
- **文档**: [docs/](docs/)

## 📈 更新日志

### v1.0.1 (2025-10-13)
- ✨ 新增一键安装脚本
- 📚 重构文档结构
- 🔧 优化安装流程
- 🛠️ 添加通用函数库
- 📝 完善故障排除指南

### v1.0.0 (2024-03-XX)
- 🎉 初始发布

---

<div align="center">

**[⬆ 回到顶部](#triposr---fast-3d-object-reconstruction)**

Made with ❤️ by [Tripo AI](https://www.tripo3d.ai/) & [Stability AI](https://stability.ai/)

</div>
