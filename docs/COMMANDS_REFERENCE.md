# TripoSR 常用命令和参数参考手册

本文档提供 TripoSR 的常用命令和参数说明，特别优化了 WSL 环境下输出到 Windows 宿主机路径的使用场景，方便 Blender 直接导入使用。

## 目录

- [快速开始](#快速开始)
- [WSL 路径转换说明](#wsl-路径转换说明)
- [常用命令示例](#常用命令示例)
- [完整参数说明](#完整参数说明)
- [输出格式选择](#输出格式选择)
- [Blender 导入指南](#blender-导入指南)
- [性能优化建议](#性能优化建议)
- [故障排除](#故障排除)

---

## 快速开始

### 基本用法（输出到 Windows 路径）

```bash
# 激活虚拟环境
source .venv/bin/activate

# 单图像重建，输出到 Windows D:\VideoCreation\Tests
python run.py examples/chair.png --output-dir /mnt/d/VideoCreation/Tests/chair/

# 带纹理输出（推荐用于 Blender）
python run.py examples/chair.png \
  --output-dir /mnt/d/VideoCreation/Tests/chair/ \
  --bake-texture \
  --texture-resolution 2048 \
  --model-save-format glb
```

---

## WSL 路径转换说明

### Windows 路径 ↔ WSL 路径对照表

| Windows 路径 | WSL 路径 | 说明 |
|-------------|---------|------|
| `D:\VideoCreation\Tests` | `/mnt/d/VideoCreation/Tests` | 推荐输出位置 |
| `C:\Users\YourName\Documents` | `/mnt/c/Users/YourName/Documents` | 用户文档目录 |
| `E:\Projects` | `/mnt/e/Projects` | 其他盘符 |

### 转换规则

1. **盘符转换**: `D:` → `/mnt/d`（小写）
2. **路径分隔符**: `\` → `/`
3. **完整示例**:
   - Windows: `D:\VideoCreation\Tests\output`
   - WSL: `/mnt/d/VideoCreation/Tests/output`

### 注意事项

- WSL 路径区分大小写
- Windows 可以直接访问 WSL 路径生成的文件
- Blender 使用 Windows 路径（`D:\VideoCreation\Tests`）导入

---

## 常用命令示例

### 1. 基础重建（无纹理）

```bash
# 输出 OBJ 格式（仅顶点颜色）
python run.py input.png --output-dir /mnt/d/VideoCreation/Tests/output1/

# 输出 GLB 格式（仅顶点颜色）
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output1/ \
  --model-save-format glb
```

**输出文件**:
- `0/mesh.obj` 或 `0/mesh.glb`
- `0/input.png`（处理后的输入图像）

---

### 2. 高质量纹理重建（推荐用于 Blender）

```bash
# OBJ + 纹理贴图
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output2/ \
  --bake-texture \
  --texture-resolution 2048

# GLB + 纹理贴图（推荐，单文件包含所有数据）
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output2/ \
  --bake-texture \
  --texture-resolution 2048 \
  --model-save-format glb
```

**输出文件**:
- **OBJ 格式**: `0/mesh.obj` + `0/texture.png`
- **GLB 格式**: `0/mesh.glb`（纹理已嵌入）

---

### 3. 批量处理

```bash
# 批量处理多张图像
python run.py img1.png img2.png img3.png \
  --output-dir /mnt/d/VideoCreation/Tests/batch/ \
  --bake-texture \
  --texture-resolution 2048 \
  --model-save-format glb
```

**输出结构**:
```
/mnt/d/VideoCreation/Tests/batch/
├── 0/
│   ├── mesh.glb
│   └── input.png
├── 1/
│   ├── mesh.glb
│   └── input.png
└── 2/
    ├── mesh.glb
    └── input.png
```

---

### 4. 高分辨率网格

```bash
# 提高网格分辨率（更平滑但更慢）
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/highres/ \
  --mc-resolution 512 \
  --bake-texture \
  --texture-resolution 4096 \
  --model-save-format glb
```

---

### 5. 渲染视频预览

```bash
# 生成 NeRF 渲染的 360 度旋转视频
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/with_video/ \
  --bake-texture \
  --render
```

**输出文件**:
- `0/mesh.obj` + `0/texture.png`
- `0/render.mp4`（360 度旋转视频）
- `0/render_000.png` 到 `0/render_029.png`（30 帧）

---

### 6. CPU 模式（无 GPU）

```bash
# 使用 CPU 运行（速度较慢）
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/cpu_output/ \
  --device cpu \
  --bake-texture
```

---

### 7. 低显存优化

```bash
# 减小 chunk size 降低显存占用
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/lowmem/ \
  --chunk-size 4096 \
  --mc-resolution 256 \
  --bake-texture \
  --texture-resolution 1024
```

---

### 8. 不移除背景（已处理的图像）

```bash
# 输入图像已经去除背景
python run.py preprocessed.png \
  --output-dir /mnt/d/VideoCreation/Tests/nobg/ \
  --no-remove-bg \
  --bake-texture
```

---

## 完整参数说明

### 必需参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `image` | string(s) | 输入图像路径，支持多个 | `input.png` 或 `*.png` |

### 输出相关参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--output-dir` | `output/` | 输出目录路径（WSL 格式） | `/mnt/d/VideoCreation/Tests/` |
| `--model-save-format` | `obj` | 输出格式：`obj` 或 `glb` | `glb`（用于 Blender） |

### 纹理相关参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--bake-texture` | false | 烘焙纹理贴图（而非顶点颜色） | 添加此参数（Blender 推荐） |
| `--texture-resolution` | `2048` | 纹理分辨率（仅与 `--bake-texture` 一起使用） | `2048` 或 `4096` |

### 模型质量参数

| 参数 | 默认值 | 说明 | 推荐值 |
|------|--------|------|--------|
| `--mc-resolution` | `256` | Marching Cubes 网格分辨率 | `256`（标准）或 `512`（高质量） |
| `--chunk-size` | `8192` | 评估块大小（影响显存占用） | `8192`（标准）或 `4096`（低显存） |

### 预处理参数

| 参数 | 默认值 | 说明 | 使用场景 |
|------|--------|------|----------|
| `--no-remove-bg` | false | 不自动移除背景 | 输入已去除背景时使用 |
| `--foreground-ratio` | `0.85` | 前景占图像比例（0.0-1.0） | 调整物体在图像中的大小 |

### 渲染参数

| 参数 | 默认值 | 说明 | 输出 |
|------|--------|------|------|
| `--render` | false | 生成 NeRF 渲染的视频 | `render.mp4` + 30 张渲染图 |

### 系统参数

| 参数 | 默认值 | 说明 | 选项 |
|------|--------|------|------|
| `--device` | `cuda:0` | 运行设备 | `cuda:0`, `cuda:1`, `cpu` |
| `--pretrained-model-name-or-path` | `stabilityai/TripoSR` | 模型路径 | HuggingFace ID 或本地路径 |

---

## 输出格式选择

### OBJ vs GLB 格式对比

| 特性 | OBJ | GLB | Blender 推荐 |
|------|-----|-----|--------------|
| **文件结构** | 多文件（.obj + .mtl + .png） | 单文件（所有数据嵌入） | **GLB** |
| **纹理支持** | 需要单独的 PNG 文件 | 纹理嵌入在 GLB 中 | **GLB** |
| **文件大小** | 较小 | 较大 | - |
| **兼容性** | 广泛支持 | 现代工具支持 | 两者都好 |
| **易用性** | 需要保持文件在同一目录 | 单文件，不会丢失纹理 | **GLB** |
| **编辑性** | 易于编辑顶点数据 | 较难手动编辑 | OBJ |

### 推荐配置

#### 用于 Blender（最佳选择）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/for_blender/ \
  --bake-texture \
  --texture-resolution 2048 \
  --model-save-format glb \
  --mc-resolution 256
```

**优势**:
- 单文件，方便管理
- 纹理不会丢失
- Blender 完美支持 GLB 导入
- 包含完整材质信息

#### 用于快速预览（性能优先）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/preview/ \
  --model-save-format obj \
  --mc-resolution 256
```

**优势**:
- 处理速度快（无纹理烘焙）
- 文件小
- 顶点颜色预览足够

#### 用于高质量渲染（质量优先）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/highquality/ \
  --bake-texture \
  --texture-resolution 4096 \
  --model-save-format glb \
  --mc-resolution 512 \
  --chunk-size 4096
```

**注意**: 需要较高显存（8GB+）

---

## Blender 导入指南

### 1. 导入 GLB 格式（推荐）

**步骤**:

1. 打开 Blender
2. `File` → `Import` → `glTF 2.0 (.glb/.gltf)`
3. 导航到 `D:\VideoCreation\Tests\your_output\0\`
4. 选择 `mesh.glb`
5. 点击 `Import glTF 2.0`

**自动导入**:
- 模型几何体
- 纹理贴图
- UV 映射
- 材质设置

**着色器设置**:
- 切换到 `Shading` 工作区
- 材质已自动设置为 `Principled BSDF`
- 纹理已连接到 `Base Color`

---

### 2. 导入 OBJ 格式（带纹理）

**步骤**:

1. 确保 `mesh.obj` 和 `texture.png` 在同一目录
2. 打开 Blender
3. `File` → `Import` → `Wavefront (.obj)`
4. 导航到 `D:\VideoCreation\Tests\your_output\0\`
5. 选择 `mesh.obj`
6. 点击 `Import OBJ`

**手动连接纹理（如果需要）**:

1. 切换到 `Shading` 工作区
2. 选择导入的对象
3. 在 `Shader Editor` 中:
   - 添加 `Image Texture` 节点（`Shift + A` → `Texture` → `Image Texture`）
   - 打开 `texture.png`
   - 连接 `Color` 输出到 `Principled BSDF` 的 `Base Color` 输入

---

### 3. 导入 OBJ 格式（仅顶点颜色）

**步骤**:

1. 打开 Blender
2. `File` → `Import` → `Wavefront (.obj)`
3. 导航到 `D:\VideoCreation\Tests\your_output\0\`
4. 选择 `mesh.obj`
5. 点击 `Import OBJ`

**启用顶点颜色显示**:

1. 切换到 `Shading` 工作区
2. 选择导入的对象
3. 在 `Shader Editor` 中:
   - 添加 `Attribute` 节点（`Shift + A` → `Input` → `Attribute`）
   - 在 `Name` 字段输入 `Col`（顶点颜色属性名）
   - 连接 `Color` 输出到 `Principled BSDF` 的 `Base Color` 输入
4. 切换视口着色为 `Material Preview` 或 `Rendered`

---

### 4. Blender 导入设置优化

**推荐的导入选项**:

| 选项 | GLB | OBJ | 说明 |
|------|-----|-----|------|
| **Split by Object** | 自动 | 勾选 | 按对象分割网格 |
| **Split by Group** | 自动 | 可选 | 按组分割 |
| **Clamp Size** | - | `0.0` | 不限制尺寸 |
| **Forward / Up Axis** | 自动 | `Y Forward`, `Z Up` | 坐标系设置 |

---

### 5. 导入后的常见调整

#### 缩放调整

```python
# 在 Blender Python Console 中
import bpy
obj = bpy.context.active_object
obj.scale = (10, 10, 10)  # 放大 10 倍
```

或手动:
1. 选择对象
2. 按 `S` 键（缩放）
3. 输入数值（如 `10`）
4. 按 `Enter`

#### 旋转调整

```python
# 旋转 90 度（X 轴）
import bpy
obj = bpy.context.active_object
obj.rotation_euler[0] = 1.5708  # 90 度（弧度）
```

#### 材质增强

1. 在 `Shading` 工作区
2. 调整 `Principled BSDF` 参数:
   - `Roughness`: 0.5-0.8（表面粗糙度）
   - `Metallic`: 0.0-0.5（金属度）
   - `Specular`: 0.5（高光强度）

---

## 性能优化建议

### 根据 GPU 显存选择参数

#### 6GB VRAM（RTX 3060, RTX 2060）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --chunk-size 4096 \
  --mc-resolution 256 \
  --bake-texture \
  --texture-resolution 2048
```

#### 8GB VRAM（RTX 3070, RTX 4060 Ti）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --chunk-size 8192 \
  --mc-resolution 256 \
  --bake-texture \
  --texture-resolution 2048
```

#### 12GB+ VRAM（RTX 3090, RTX 4080, RTX 4090）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --chunk-size 8192 \
  --mc-resolution 512 \
  --bake-texture \
  --texture-resolution 4096
```

---

### 速度 vs 质量权衡

| 配置 | 速度 | 质量 | 显存 | 适用场景 |
|------|------|------|------|----------|
| `--mc-resolution 128` | 最快 | 低 | 低 | 快速原型 |
| `--mc-resolution 256` | 快 | 中 | 中 | **标准使用（推荐）** |
| `--mc-resolution 512` | 慢 | 高 | 高 | 最终产品 |
| `--mc-resolution 1024` | 很慢 | 极高 | 很高 | 专业级别 |

---

## 故障排除

### 问题 1: Blender 导入后无纹理

**原因**: OBJ 文件找不到纹理文件

**解决方案**:
1. 确保 `mesh.obj`、`mesh.mtl` 和 `texture.png` 在同一目录
2. 使用 GLB 格式（纹理嵌入）:
   ```bash
   --model-save-format glb --bake-texture
   ```

---

### 问题 2: Blender 导入 GLB 后材质错误

**解决方案**:
1. 切换视口着色为 `Material Preview` 或 `Rendered`（顶部右侧图标）
2. 检查 Blender 版本（推荐 3.0+）
3. 确保启用了 glTF 2.0 插件:
   - `Edit` → `Preferences` → `Add-ons`
   - 搜索 `glTF`
   - 勾选 `Import-Export: glTF 2.0 format`

---

### 问题 3: 模型在 Blender 中太小或太大

**解决方案**:
1. 导入时调整 `Scale` 选项
2. 导入后手动缩放（`S` 键）
3. 重新生成时调整输入图像的前景比例:
   ```bash
   --foreground-ratio 0.7  # 物体更小
   --foreground-ratio 0.95  # 物体更大
   ```

---

### 问题 4: Windows 找不到 WSL 输出的文件

**检查路径**:
1. 在 WSL 中运行: `ls /mnt/d/VideoCreation/Tests/`
2. 在 Windows 资源管理器中访问: `D:\VideoCreation\Tests\`
3. 确保目录存在:
   ```bash
   mkdir -p /mnt/d/VideoCreation/Tests
   ```

---

### 问题 5: 显存不足错误

**错误信息**: `CUDA out of memory`

**解决方案**:
1. 减小 chunk size:
   ```bash
   --chunk-size 2048
   ```
2. 降低网格分辨率:
   ```bash
   --mc-resolution 128
   ```
3. 降低纹理分辨率:
   ```bash
   --texture-resolution 1024
   ```
4. 使用 CPU 模式:
   ```bash
   --device cpu
   ```

---

### 问题 6: 纹理分辨率不够清晰

**解决方案**:
1. 提高纹理分辨率:
   ```bash
   --texture-resolution 4096  # 或 8192（需要大显存）
   ```
2. 提高网格分辨率:
   ```bash
   --mc-resolution 512
   ```
3. 使用高质量输入图像（推荐 1024×1024 以上）

---

## 完整工作流示例

### 场景: 从照片到 Blender 的完整流程

**1. 准备输入图像**

```bash
# 假设有一张产品照片 product.jpg
# 放在 D:\VideoCreation\Input\ 目录
```

**2. 运行 TripoSR（在 WSL 中）**

```bash
# 激活环境
cd /home/doer/repos/TripoSR
source .venv/bin/activate

# 运行重建（输出到 Windows 路径）
python run.py /mnt/d/VideoCreation/Input/product.jpg \
  --output-dir /mnt/d/VideoCreation/Tests/product_model/ \
  --bake-texture \
  --texture-resolution 2048 \
  --model-save-format glb \
  --mc-resolution 256 \
  --render
```

**3. 查看输出（在 Windows 中）**

```
D:\VideoCreation\Tests\product_model\0\
├── mesh.glb          ← 3D 模型（用于 Blender）
├── input.png         ← 处理后的输入图像
├── render.mp4        ← 360 度预览视频
└── render_*.png      ← 渲染帧图像
```

**4. 导入 Blender（在 Windows 中）**

1. 打开 Blender
2. `File` → `Import` → `glTF 2.0 (.glb/.gltf)`
3. 选择 `D:\VideoCreation\Tests\product_model\0\mesh.glb`
4. 点击 `Import glTF 2.0`
5. 切换视口着色为 `Material Preview`
6. 完成！

---

## 批处理脚本示例

### Bash 脚本：批量处理文件夹中的所有图像

创建文件 `batch_process.sh`:

```bash
#!/bin/bash

# 配置
INPUT_DIR="/mnt/d/VideoCreation/Input"
OUTPUT_DIR="/mnt/d/VideoCreation/Tests/batch_output"
TEXTURE_RES=2048
MC_RES=256

# 激活环境
source .venv/bin/activate

# 处理所有 PNG 和 JPG 文件
for img in "$INPUT_DIR"/*.{png,jpg,jpeg}; do
    [ -f "$img" ] || continue

    filename=$(basename "$img")
    name="${filename%.*}"

    echo "Processing: $filename"

    python run.py "$img" \
        --output-dir "$OUTPUT_DIR/$name/" \
        --bake-texture \
        --texture-resolution $TEXTURE_RES \
        --model-save-format glb \
        --mc-resolution $MC_RES
done

echo "Batch processing completed!"
```

**使用方法**:

```bash
chmod +x batch_process.sh
./batch_process.sh
```

---

## 高级技巧

### 1. 自定义模型路径

```bash
# 使用本地模型（避免重复下载）
python run.py input.png \
  --pretrained-model-name-or-path /path/to/local/model \
  --output-dir /mnt/d/VideoCreation/Tests/output/
```

### 2. 处理透明背景的 PNG

```bash
# 输入已经是透明背景
python run.py transparent_input.png \
  --no-remove-bg \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --bake-texture
```

### 3. 多 GPU 使用

```bash
# 使用第二块 GPU
python run.py input.png \
  --device cuda:1 \
  --output-dir /mnt/d/VideoCreation/Tests/output/
```

### 4. 环境变量配置

```bash
# 设置 HuggingFace 镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 设置缓存目录
export HF_HOME=/path/to/cache

# 运行
python run.py input.png --output-dir /mnt/d/VideoCreation/Tests/output/
```

---

## 快速参考卡

### 最常用的三个命令

#### 1. 标准质量（推荐）

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --bake-texture \
  --model-save-format glb
```

#### 2. 高质量

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --bake-texture \
  --texture-resolution 4096 \
  --mc-resolution 512 \
  --model-save-format glb
```

#### 3. 快速预览

```bash
python run.py input.png \
  --output-dir /mnt/d/VideoCreation/Tests/output/ \
  --model-save-format obj
```

---

## 相关文档

- [快速开始指南](QUICK_START.md)
- [安装指南](INSTALLATION.md)
- [故障排除](TROUBLESHOOTING.md)
- [原始 README](ORIGINAL_README.md)

---

## 更新日志

- **2025-10-13**: 初始版本，添加完整的命令参考和 Blender 集成指南

---

**文档维护**: 本文档基于 TripoSR 最新版本编写，如有问题请提交 Issue。
