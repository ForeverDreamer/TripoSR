# TripoSR 快速开始指南

## 一键安装

```bash
./setup.sh
```

这将自动完成：
- 系统兼容性检查
- CUDA 环境配置
- Python 虚拟环境创建
- 所有依赖安装

## 前提条件

- **Python**: 3.8+ (推荐 3.11)
- **GPU**: NVIDIA GPU with 6GB+ VRAM (推荐)
- **内存**: 8GB+ RAM
- **磁盘**: 10GB+ 可用空间

## 基本使用

### 1. 激活环境

```bash
source .venv/bin/activate
```

### 2. 单图像重建

```bash
python run.py examples/chair.png --output-dir output/
```

### 3. 启动 Web 界面

```bash
python gradio_app.py
```

访问 http://localhost:7860

### 4. 批量处理

```bash
python run.py image1.png image2.png image3.png --output-dir output/
```

### 5. 生成纹理

```bash
python run.py image.png --bake-texture --texture-resolution 1024 --output-dir output/
```

## 常见命令选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--device` | 设备 (cuda:0/cpu) | cuda:0 |
| `--mc-resolution` | 网格分辨率 | 256 |
| `--bake-texture` | 生成纹理贴图 | false |
| `--texture-resolution` | 纹理分辨率 | 1024 |
| `--no-remove-bg` | 不移除背景 | false |
| `--output-dir` | 输出目录 | output/ |

## 性能预期

| 硬件 | 速度 | VRAM使用 |
|------|------|----------|
| RTX 4090 | ~1-2秒/图 | ~6GB |
| RTX 3080 | ~2-4秒/图 | ~6GB |
| RTX 2080 | ~4-6秒/图 | ~6GB |
| CPU | ~60-120秒/图 | N/A |

## 故障排除

### GPU未检测到

```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

### torchmcubes错误

需要CUDA toolkit:
```bash
sudo apt-get install -y cuda-toolkit-12-6
source ~/.bashrc
./setup.sh
```

详细文档见 [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 下一步

- 📖 完整文档: [docs/](.)
- 🔧 详细安装: [INSTALLATION.md](INSTALLATION.md)
- 💡 使用示例: [USAGE.md](USAGE.md)
- 🐛 故障排除: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
