# MSFT Cross-Image FractalGen — Colab / Server Guide

> **从零开始在 Colab 或任何 GPU 服务器上运行完整实验**

---

## 目录

1. [实验概述](#1-实验概述)
2. [Colab 一键运行](#2-colab-一键运行)
3. [服务器逐步骤运行](#3-服务器逐步骤运行)
4. [监控与评估](#4-监控与评估)
5. [常见问题排查](#5-常见问题排查)
6. [自定义实验](#6-自定义实验)

---

## 1. 实验概述

### 做什么

用**残缺的多尺度数据**（16×16 粗图 + 4×4 高清patch，来自同类但不同图）训练 FractalGen，
验证是否能恢复 pixel-to-pixel 的 64×64 图像生成能力。

### 三阶段训练

| Phase | 内容 | 数据 | 冻结 | epochs | 时间(5090) |
|-------|------|------|------|--------|-------------|
| **1** | L0 预训练 | 仅 16×16 粗图（上采样到 64×64） | L1, PixLoss | 200 | ~1h |
| **2** | L1+PixelLoss 预训练 | 粗图 + 4×4 高清patch（跨图） | L0 frozen | 200 | ~2h |
| **3** | 联合微调 + 跨层一致 | 粗图 + 4×4 高清patch（跨图） | 全部解冻 | 100 | ~3h |

**总计**：5090 上约 6 小时完成全部三阶段训练。

### 预期输出

- `output/msft_phase3_*/samples/epoch0200.png`：生成的 64×64 图像
- `output/msft_phase3_*/phase3_best.pth`：最佳模型权重
- TensorBoard logs：`logs/msft_phase*/`

---

## 2. Colab 一键运行

### 2.1 打开 Colab

新建 notebook，选择 **T4 GPU**（免费）或 **A100**（Colab Pro）。

> ⚠️ T4 用 `--model_size tiny`，5090/A100 用 `--model_size 5090`。

### 2.2 Cell 1: 环境准备

```python
# ═══════════════════════════════════════════════════════
# Cell 1: 挂载 Google Drive + 安装依赖
# ═══════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

# 安装依赖
!pip install -q torch torchvision tensorboard timm pillow tqdm numpy scipy

# Clone 项目（你需要先把项目 push 到 GitHub）
# 或者直接从 Google Drive 复制
# !git clone https://github.com/YOUR_USER/msft-cross-image.git
# %cd msft-cross-image

# 如果代码在 Google Drive:
# %cd /content/drive/MyDrive/msft-cross-image

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 2.3 Cell 2: 下载 Tiny ImageNet（Stanford 源）

```python
# ═══════════════════════════════════════════════════════
# Cell 2: 下载 Tiny ImageNet (200 类, 100K 训练集)
# 从 Stanford 源下载，国内直接可访问
# ═══════════════════════════════════════════════════════

import os, urllib.request, zipfile, shutil

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

ZIP_PATH = f"{DATA_DIR}/tiny-imagenet-200.zip"
SRC = f"{DATA_DIR}/tiny-imagenet-200"

# ── 下载 (500MB) ──
if not os.path.exists(ZIP_PATH):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    print("Downloading Tiny ImageNet (500MB)...")
    urllib.request.urlretrieve(url, ZIP_PATH)
else:
    print("Zip already downloaded.")

# ── 解压 ──
if not os.path.exists(SRC):
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(DATA_DIR)
else:
    print("Already extracted.")

# ── 重组 train: train/class/images/* → train/class/* ──
print("Reorganizing train...")
for cls in os.listdir(f"{SRC}/train"):
    img_dir = f"{SRC}/train/{cls}/images"
    if os.path.exists(img_dir):
        for img in os.listdir(img_dir):
            shutil.move(f"{img_dir}/{img}", f"{SRC}/train/{cls}/{img}")
        os.rmdir(img_dir)

# ── 重组 val: val/images/* → val/class/* ──
print("Reorganizing val...")
val_dir = f"{SRC}/val"
with open(f"{val_dir}/val_annotations.txt") as f:
    annotations = [l.strip().split('\t') for l in f]

for img_name, class_id, *_ in annotations:
    dst_dir = f"{val_dir}/{class_id}"
    os.makedirs(dst_dir, exist_ok=True)
    shutil.move(f"{val_dir}/images/{img_name}", f"{dst_dir}/{img_name}")

shutil.rmtree(f"{val_dir}/images")
os.remove(f"{val_dir}/val_annotations.txt")

n_train = len(os.listdir(f"{SRC}/train"))
n_val = len(os.listdir(f"{SRC}/val"))
print(f"Done! Train: {n_train} classes, Val: {n_val} classes")
```

> `--data_root` 传入 `./data/tiny-imagenet-200`。

### 2.4 Cell 3: Phase 1 — L0 粗图预训练

```python
# ═══════════════════════════════════════════════════════
# Cell 3: Phase 1 — L0 pretraining on coarse images
# ═══════════════════════════════════════════════════════

# 选择模型大小
# T4 (16GB): 用 tiny
# A100/V100/5090 (32GB+): 用 5090

GPU_MEM = torch.cuda.get_device_properties(0).total_mem / 1e9 if torch.cuda.is_available() else 0
if GPU_MEM < 20:
    MODEL = 'tiny'
    BATCH = 32
    print(f"⚠️ Low VRAM ({GPU_MEM:.0f}GB). Using tiny model.")
else:
    MODEL = '5090'
    BATCH = 64
    print(f"✓ VRAM: {GPU_MEM:.0f}GB. Using 5090 model.")

!python train_msft.py \
    --phase 1 \
    --data_root ./data/tiny-imagenet-200 \
    --model_size {MODEL} \
    --batch_size {BATCH} \
    --epochs 200 \
    --lr 5e-4 \
    --pairing_mode same_class \
    --output_dir ./output
```

### 2.5 Cell 4: Phase 2 — L1+PixelLoss 预训练

```python
# ═══════════════════════════════════════════════════════
# Cell 4: Phase 2 — L1+PixelLoss pretraining (L0 frozen)
# ═══════════════════════════════════════════════════════

import glob
import os

# 自动找到 Phase 1 的最佳 checkpoint
phase1_dirs = sorted(glob.glob('./output/msft_phase1_*'))
if phase1_dirs:
    latest = phase1_dirs[-1]
    ckpt_path = f'{latest}/phase1_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'{latest}/phase1_final.pth'
    print(f"Resuming from: {ckpt_path}")
else:
    ckpt_path = None
    print("No Phase 1 checkpoint found!")

!python train_msft.py \
    --phase 2 \
    --data_root ./data/tiny-imagenet-200 \
    --model_size {MODEL} \
    --batch_size {max(BATCH//2, 16)} \
    --epochs 200 \
    --lr 1e-4 \
    --pairing_mode same_class \
    --resume {ckpt_path} \
    --output_dir ./output
```

### 2.6 Cell 5: Phase 3 — 联合微调

```python
# ═══════════════════════════════════════════════════════
# Cell 5: Phase 3 — Joint fine-tuning with consistency loss
# ═══════════════════════════════════════════════════════

phase2_dirs = sorted(glob.glob('./output/msft_phase2_*'))
if phase2_dirs:
    latest = phase2_dirs[-1]
    ckpt_path = f'{latest}/phase2_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = f'{latest}/phase2_final.pth'
    print(f"Resuming from: {ckpt_path}")

!python train_msft.py \
    --phase 3 \
    --data_root ./data/tiny-imagenet-200 \
    --model_size {MODEL} \
    --batch_size {max(BATCH//4, 8)} \
    --epochs 100 \
    --lr 2e-5 \
    --pairing_mode same_class \
    --resume {ckpt_path} \
    --output_dir ./output
```

### 2.7 Cell 6: 生成样本 & 可视化

```python
# ═══════════════════════════════════════════════════════
# Cell 6: Generate and visualize samples
# ═══════════════════════════════════════════════════════

import sys
sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from models.fractalgen_msft import fractalmar_msft_5090, fractalmar_msft_tiny

# Load best model
phase3_dirs = sorted(glob.glob('./output/msft_phase3_*'))
ckpt_path = f'{phase3_dirs[-1]}/phase3_best.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if GPU_MEM < 20:
    model = fractalmar_msft_tiny(class_num=200)
else:
    model = fractalmar_msft_5090(class_num=200)

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.to(device)
model.eval()

# Generate
print("Generating 16 images...")
class_labels = torch.randint(0, 200, (16,)).to(device)

with torch.no_grad():
    images = model.generate(
        class_labels=class_labels,
        num_iter_l0=64,
        num_iter_l1=16,
        cfg=1.0,
        temperature=1.0,
    )

# Denormalize and display
mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
images = images * std + mean
images = images.clamp(0, 1)

grid = make_grid(images.cpu(), nrow=4, padding=2)
plt.figure(figsize=(10, 10))
plt.imshow(grid.permute(1, 2, 0))
plt.axis('off')
plt.title('MSFT Cross-Image FractalGen — Generated 64×64 Images')
plt.show()
```

### 2.8 Cell 7: 保存结果到 Google Drive

```python
# ═══════════════════════════════════════════════════════
# Cell 7: Save results to Google Drive
# ═══════════════════════════════════════════════════════

import shutil

DRIVE_SAVE_PATH = '/content/drive/MyDrive/msft-cross-image-results'

# Copy outputs
shutil.copytree('./output', f'{DRIVE_SAVE_PATH}/output', dirs_exist_ok=True)
shutil.copytree('./logs', f'{DRIVE_SAVE_PATH}/logs', dirs_exist_ok=True)

print(f"Results saved to: {DRIVE_SAVE_PATH}")
```

---

## 3. 服务器逐步骤运行

### 3.1 环境准备

```bash
# Ubuntu 22.04 / 24.04
# 需要: Python 3.10+, CUDA 12.x, 至少 16GB VRAM

# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard timm pillow tqdm numpy scipy

# 3. Clone 或复制项目代码
git clone https://github.com/SpencerRaw/msft-cross-image.git msft-cross-image
cd msft-cross-image
```

### 3.2 下载数据

```bash
# 方法 A: Stanford 源 (国内可用)
python -c "
import os, urllib.request, zipfile, shutil

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)
SRC = f'{DATA_DIR}/tiny-imagenet-200'

# 下载+解压
if not os.path.exists(SRC):
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    print('Downloading...')
    urllib.request.urlretrieve(url, f'{DATA_DIR}/tiny-imagenet-200.zip')
    print('Extracting...')
    from zipfile import ZipFile
    with ZipFile(f'{DATA_DIR}/tiny-imagenet-200.zip', 'r') as zf:
        zf.extractall(DATA_DIR)

# 重组 train
for cls in os.listdir(f'{SRC}/train'):
    img_dir = f'{SRC}/train/{cls}/images'
    if os.path.exists(img_dir):
        for img in os.listdir(img_dir):
            shutil.move(f'{img_dir}/{img}', f'{SRC}/train/{cls}/{img}')
        os.rmdir(img_dir)

# 重组 val
val_dir = f'{SRC}/val'
with open(f'{val_dir}/val_annotations.txt') as f:
    ann = [l.strip().split('\t') for l in f]
for img_name, class_id, *_ in ann:
    dst = f'{val_dir}/{class_id}'; os.makedirs(dst, exist_ok=True)
    shutil.move(f'{val_dir}/images/{img_name}', f'{dst}/{img_name}')
shutil.rmtree(f'{val_dir}/images')
os.remove(f'{val_dir}/val_annotations.txt')
print(f'Done! {len(os.listdir(SRC + \"/train\"))} train, {len(os.listdir(SRC + \"/val\"))} val classes')
"
```

### 3.3 三阶段训练

```bash
# ── Phase 1: L0 粗图预训练 (~1h on 5090) ──
python train_msft.py \
    --phase 1 \
    --data_root ./data/tiny-imagenet-200 \
    --model_size 5090 \
    --batch_size 64 \
    --epochs 200 \
    --lr 5e-4 \
    --pairing_mode same_class \
    --output_dir ./output

# ── Phase 2: L1+PixelLoss (~2h on 5090) ──
# 自动找到 Phase 1 的 best checkpoint
PHASE1_CKPT=$(ls -t ./output/msft_phase1_*/phase1_best.pth 2>/dev/null | head -1)
python train_msft.py \
    --phase 2 \
    --data_root ./data/tiny-imagenet-200 \
    --model_size 5090 \
    --batch_size 32 \
    --epochs 200 \
    --lr 1e-4 \
    --pairing_mode same_class \
    --resume "$PHASE1_CKPT" \
    --output_dir ./output

# ── Phase 3: 联合微调 (~3h on 5090) ──
PHASE2_CKPT=$(ls -t ./output/msft_phase2_*/phase2_best.pth 2>/dev/null | head -1)
python train_msft.py \
    --phase 3 \
    --data_root ./data/tiny-imagenet-200 \
    --model_size 5090 \
    --batch_size 16 \
    --epochs 100 \
    --lr 2e-5 \
    --pairing_mode same_class \
    --resume "$PHASE2_CKPT" \
    --output_dir ./output
```

### 3.4 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir ./logs --port 6006 --bind_all

# 在浏览器打开: http://<server-ip>:6006
```

---

## 4. 监控与评估

### 4.1 TensorBoard 指标

| 指标 | 正常范围 | 异常信号 |
|------|---------|---------|
| `loss/l0` | Phase1: 4.5→3.0; Phase2-3: 稳定在 3.0-4.0 | 不下降 → LR 太低/数据问题 |
| `loss/pixel` | Phase2-3: 5.5→4.0 | 震荡 → batch 太小/LR 太高 |
| `loss/consistency` | Phase3: 0.5→0.1 | >1.0 → 跨图失配严重，增大 λ |
| `loss/total` | 持续下降 | 突然跳升 → 梯度爆炸 |

### 4.2 生成样本检查

每 20 epochs 自动保存到 `output/msft_phaseX_*/samples/epoch*.png`。

**正常进度**：
- Epoch 20: 模糊色块，但整体布局合理
- Epoch 60: 开始出现可辨认的物体轮廓
- Epoch 120: 细节出现，类别特征明显
- Epoch 200: 清晰图像，但可能略粗糙（这是正常的数据残缺限制）

### 4.3 判断实验是否成功

| 检查项 | 标准 |
|--------|------|
| 生成多样性 | 不同 class 产生明显不同的图像 |
| 结构合理性 | 物体在画面中位置合理，不全是噪声 |
| 跨尺度一致性 | 粗结构和细结构协调（不会出现"天空纹理在物体上"） |
| Pixel loss 收敛 | < 4.0 bits/dim（Tiny ImageNet 级别合理） |

---

## 5. 常见问题排查

### 5.1 CUDA Out of Memory

```bash
# 减小 batch size
--batch_size 16  # 甚至 8

# 使用 gradient checkpointing
# 在 config 中设置 grad_checkpointing=True

# 从 Python 中修改:
# model.generator_l0.grad_checkpointing = True
```

### 5.2 下载数据集失败

```python
# 备选方案: 手动下载
# wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
# unzip tiny-imagenet-200.zip

# 然后转换为 ImageFolder 格式:
# python convert_tiny_imagenet.py
```

### 5.3 训练不收敛 / Loss 不下降

```bash
# 诊断步骤
# 1. 检查数据是否正确加载
python -c "
from data.msft_dataset import MSFT2LevelDataset
ds = MSFT2LevelDataset('./data/tiny-imagenet-200', train=True)
sample = ds[0]
print('Coarse shape:', sample['coarse'].shape)  # (3, 16, 16)
print('Fine shape:', sample['fine'].shape)       # (8, 3, 4, 4)
print('Positions:', sample['positions'])          # (8, 2)
print('Label:', sample['label'])
"

# 2. 尝试 same_image 模式（排除跨图问题）
python train_msft.py --phase 1 --pairing_mode same_image ...

# 如果 same_image 收敛但 same_class 不收敛 → 跨图失配太大
# 解决: 增大 consistency_weight (0.1 → 0.5)
```

### 5.4 生成图像全黑/全噪

```bash
# 检查 normalization
# FractalGen 内部使用 ImageNet normalization
# 生成后需要 denormalize:
# images = images * std + mean

# 检查 temperature
# temperature=1.0 是默认值
# 太高 (>2.0) → 噪声
# 太低 (<0.5) → 全黑/模糊
```

---

## 6. 自定义实验

### 6.1 修改配对模式

```bash
# same_class (默认): 粗图和细图来自同类不同图 → 模拟跨尺度物理
--pairing_mode same_class

# same_image: 粗图和细图来自同一张图 → 空间对齐，验证架构
--pairing_mode same_image

# random: 粗图和细图完全随机 → 负对照
--pairing_mode random
```

### 6.2 调整模型大小

```python
# 在 models/fractalgen_msft.py 中修改 factory functions:

def fractalmar_msft_5090(class_num=200, **kwargs):
    return FractalGenMSFT(
        embed_dim_list=(512, 256, 64),   # 增大 → 更强但更慢
        num_blocks_list=(12, 4, 2),      # 增大 → 更深
        num_heads_list=(8, 4, 2),
        ...
    )
```

### 6.3 使用完整 ImageNet (1000类)

```bash
# 1. 下载 ImageNet-1k-64
# HuggingFace: benjamin-paine/imagenet-1k-64x64

# 2. 修改 class_num
--model_size 5090  # class_num 通过 config 传入

# 3. 在 create_model 中传入 class_num=1000
```

### 6.4 对比实验矩阵

建议跑以下实验来验证跨图学习的核心假设：

| 实验 | pairing_mode | consistency_loss | 预期结果 |
|------|-------------|-----------------|---------|
| A (上界) | same_image | none | 最好，空间对齐无歧义 |
| B (我们的方法) | same_class | contrastive | 好，一致性loss弥合跨图gap |
| C (消融) | same_class | none | 中等，仅靠类别条件 |
| D (下界) | random | none | 差，无有效条件信号 |

---

## 附录: 项目文件结构

```
msft-cross-image/
├── models/
│   ├── __init__.py
│   └── fractalgen_msft.py      # 核心模型 (FractalGenMSFT + 一致性loss)
├── data/
│   ├── __init__.py
│   └── msft_dataset.py         # 数据集 (MSFT2LevelDataset + collate)
├── train_msft.py               # 训练脚本 (三阶段)
├── COLAB_GUIDE.md              # 本文件
└── README.md                   # 项目说明

依赖:
../fractalgen/                  # 原版 FractalGen 代码
├── models/mar.py
├── models/ar.py
├── models/pixelloss.py
```
