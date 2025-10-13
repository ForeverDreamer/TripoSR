# TripoSR 快速参考卡

## 🚀 一键安装

```bash
./setup.sh
```

## 📝 常用命令

### 基本使用
```bash
source .venv/bin/activate                          # 激活环境
python run.py image.png --output-dir output/       # 重建3D模型
python gradio_app.py                               # 启动Web界面
```

### 高级选项
```bash
# 批量处理
python run.py img1.png img2.png img3.png --output-dir output/

# 生成纹理
python run.py image.png --bake-texture --texture-resolution 1024

# 使用CPU
python run.py image.png --device cpu

# 自定义分辨率
python run.py image.png --mc-resolution 512
```

## 🔧 脚本工具

### 系统检查
```bash
bash scripts/check_system.sh              # 基本检查
bash scripts/check_system.sh --verbose    # 详细信息
bash scripts/check_system.sh -o report.txt # 保存报告
```

### 安装选项
```bash
./setup.sh                      # 标准安装
./setup.sh --python 3.10        # 指定Python版本
./setup.sh --skip-check         # 跳过系统检查
./setup.sh -y                   # 自动确认
```

## 📚 文档快速链接

| 需求 | 文档 |
|------|------|
| 第一次使用 | [QUICK_START.md](docs/QUICK_START.md) |
| 安装问题 | [INSTALLATION.md](docs/INSTALLATION.md) |
| CUDA配置 | [CUDA_INSTALL.md](docs/CUDA_INSTALL.md) |
| 遇到错误 | [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) |
| 优化详情 | [OPTIMIZATION_SUMMARY.md](docs/OPTIMIZATION_SUMMARY.md) |

## 🐛 快速故障排除

### GPU未检测
```bash
nvidia-smi  # 检查驱动
python -c "import torch; print(torch.cuda.is_available())"
```

### torchmcubes错误
```bash
sudo apt-get install -y cuda-toolkit-12-6
source ~/.bashrc
./setup.sh
```

### 内存不足
```bash
python run.py image.png --mc-resolution 128  # 降低分辨率
python run.py image.png --device cpu          # 使用CPU
```

## 💡 性能优化

```bash
# 禁用背景移除（如已预处理）
python run.py image.png --no-remove-bg

# 调整chunk size
python run.py image.png --chunk-size 4096
```

## 📞 获取帮助

```bash
python run.py --help                    # 查看所有选项
bash scripts/check_system.sh --help    # 系统检查帮助
./setup.sh --help                       # 安装选项
```

## 🌐 在线资源

- **GitHub**: https://github.com/VAST-AI-Research/TripoSR
- **Paper**: https://arxiv.org/abs/2403.02151
- **HuggingFace**: https://huggingface.co/stabilityai/TripoSR
- **Demo**: https://huggingface.co/spaces/stabilityai/TripoSR
- **Discord**: https://discord.gg/mvS9mCfMnQ

---

**提示**: 将此文件添加到书签，方便随时查阅！
