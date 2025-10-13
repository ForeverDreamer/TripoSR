# TripoSR 故障排除指南

## 常见问题

### 1. torchmcubes CUDA 错误

**错误**: `AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'`

**原因**: torchmcubes 未使用 CUDA 编译

**解决方案**:

```bash
# 1. 确保CUDA toolkit已安装
sudo apt-get install -y cuda-toolkit-12-6

# 2. 配置环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 3. 重新安装
rm -rf .venv
./setup.sh
```

### 2. CUDA 版本不匹配

**警告**: `CUDA capability sm_XX is not compatible`

**说明**: 你的 GPU 比 PyTorch 版本新，通常可以正常运行

**如需最新支持**:

```bash
# 安装 PyTorch nightly
source .venv/bin/activate
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu121
```

### 3. GPU 内存不足 (OOM)

**错误**: `CUDA out of memory`

**解决方案**:

```bash
# 方法1: 降低分辨率
python run.py image.png --mc-resolution 128

# 方法2: 减小chunk size
python run.py image.png --chunk-size 4096

# 方法3: 使用CPU
python run.py image.png --device cpu
```

### 4. 模型下载失败

**错误**: `HuggingFace hub connection error`

**解决方案**:

```bash
# 使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或手动下载
git lfs install
git clone https://huggingface.co/stabilityai/TripoSR models/TripoSR

# 使用本地模型
python run.py image.png --pretrained-model-name-or-path models/TripoSR
```

### 5. nvidia-smi 在 WSL 中不工作

**问题**: WSL 中找不到 GPU

**解决方案**:

```bash
# 1. 确保Windows有最新NVIDIA驱动
# 在Windows PowerShell中运行:
nvidia-smi

# 2. 更新WSL内核
wsl --update

# 3. 重启WSL
wsl --shutdown
# 然后重新打开WSL终端
```

### 6. Python 版本问题

**错误**: `Python 3.X is required`

**解决方案**:

```bash
# 安装Python 3.11
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# 使用特定版本
./setup.sh --python 3.11
```

### 7. 依赖安装失败

**错误**: 各种包安装失败

**解决方案**:

```bash
# 更新pip和setuptools
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# 安装系统依赖
sudo apt-get install -y build-essential python3-dev

# 重新安装
pip install -r requirements.txt
```

### 8. rembg 背景移除问题

**错误**: `onnxruntime` 相关错误

**解决方案**:

```bash
source .venv/bin/activate
pip install onnxruntime
```

### 9. Gradio 端口占用

**错误**: `Port 7860 is already in use`

**解决方案**:

```bash
# 使用不同端口
python gradio_app.py --server-port 7861

# 或结束占用进程
lsof -ti:7860 | xargs kill -9
```

### 10. 权限问题

**错误**: `Permission denied`

**解决方案**:

```bash
# 添加执行权限
chmod +x setup.sh scripts/*.sh

# 或使用bash运行
bash setup.sh
```

## 性能优化

### 加速推理

```bash
# 使用fp16 (如果支持)
export TORCH_DTYPE=float16
python run.py image.png

# 禁用背景移除（如果已预处理）
python run.py image.png --no-remove-bg
```

### 降低内存使用

```bash
# 减小网格分辨率
python run.py image.png --mc-resolution 128

# 减小chunk size
python run.py image.png --chunk-size 2048
```

## 调试模式

```bash
# 启用详细日志
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

python run.py image.png
```

## 获取帮助

1. 查看完整选项: `python run.py --help`
2. 系统检查: `bash scripts/check_system.sh`
3. GitHub Issues: https://github.com/VAST-AI-Research/TripoSR/issues
4. Discord: https://discord.gg/mvS9mCfMnQ

## 报告问题

提交issue时请包含:

```bash
# 系统信息
bash scripts/check_system.sh --output system_info.txt

# Python环境
source .venv/bin/activate
pip list > pip_list.txt

# 错误信息（完整的 traceback）
```
