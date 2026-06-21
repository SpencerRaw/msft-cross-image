# MSFT Cross-Image FractalGen

**用残缺跨尺度数据训练 FractalGen —— 验证「类条件空间先验」能否替代空间确定性对齐。**

物理类比：宏观测量（16×16 粗图）和微观测量（4×4 高清patch）来自同一体系类型但不同实验 → 能否重建一致的跨尺度生成模型？

## 核心思想

```
原版 FractalGen:  64×64 原图 → L0(256 tokens) → L1(16 tokens/patch) → pixels
                   ↑ 所有层级共享同一张图的确定性空间对应

MSFT 变体:        16×16 粗图A → L0 → conditions[pos]
                  4×4 高精patchB → L1 → pixels
                   ↑ A和B来自同类但不同图，通过「类条件空间先验」+「对比一致性loss」弥合
```

## 架构

```
L0 (MAR, 256 tokens)
  ├── Input: 16×16 → nearest upsample → 64×64
  ├── Patchify: 4×4 → 16×16 grid
  └── Output: 5 conditions per position

Cross-Level Bridge (NEW)
  ├── 从 L0 condition grid 中按空间位置提取条件
  ├── 喂给 L1（4×4 patch 来自同类不同图）
  └── 对比 loss 确保 L0 条件分布 ≈ L1 patch 表示分布

L1 (MAR, 16 tokens)
  ├── Input: 4×4 patch → 16 pixel tokens
  └── Output: conditions for PixelLoss

PixelLoss
  └── RGB 256-way cross-entropy
```

## 新增 Loss

| Loss | 位置 | 作用 |
|------|------|------|
| L0 MAR loss | L0 | 粗图自重建（标准） |
| PixelLoss | L1→output | 像素级生成（标准） |
| **CrossLevelContrastiveLoss** 🆕 | L0↔L1 | 同类condition和patch靠近，异类远离 |
| CrossLevelMMDLoss 🆕 | L0↔L1 | 轻量替代：condition/patch分布匹配 |

## 快速开始

### Colab (推荐)

打开 [COLAB_GUIDE.md](./COLAB_GUIDE.md)，逐个 cell 运行。

### 服务器

```bash
# 1. 环境
pip install torch torchvision tensorboard timm pillow tqdm numpy scipy datasets

# 2. 下载数据
python -c "..."  # 见 COLAB_GUIDE.md

# 3. 三阶段训练
python train_msft.py --phase 1 --data_root ./data/tiny-imagenet-200 --model_size 5090
python train_msft.py --phase 2 --data_root ./data/tiny-imagenet-200 --model_size 5090
python train_msft.py --phase 3 --data_root ./data/tiny-imagenet-200 --model_size 5090
```

## 模型规格

| 版本 | Params | 适用 GPU | Phase1/2/3 总时间 |
|------|--------|---------|-------------------|
| `tiny` | ~4M | T4 (免费Colab) | ~8h |
| `5090` | ~15M | 5090/A100 (32GB) | ~6h |
| `medium` | ~50M | A100/H100 | ~12h |

## 文件结构

```
msft-cross-image/
├── models/
│   └── fractalgen_msft.py    # 核心模型 + 一致性 loss
├── data/
│   └── msft_dataset.py       # 数据集 (2-level, cross-image)
├── train_msft.py             # 训练脚本 (三阶段)
├── COLAB_GUIDE.md            # 详细运行指南
└── README.md                 # 本文件
```

## 依赖

- 原版 FractalGen 代码: `../fractalgen/models/{mar,ar,pixelloss}.py`
- PyTorch 2.0+, torchvision, timm, tensorboard

## 引用

- FractalGen: [arxiv.org/abs/2502.17437](https://arxiv.org/abs/2502.17437)
- MAR: [arxiv.org/abs/2406.11838](https://arxiv.org/abs/2406.11838)
