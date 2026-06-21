"""
MSFT Dataset for 2-Level Cross-Image Training.

Creates paired data: 16×16 coarse images and 4×4 high-res patches from the
SAME CLASS but DIFFERENT IMAGES, analogous to physics cross-scale simulation.

Data flow:
    64×64 original → avg_pool(4) → 16×16 coarse
    64×64 original → random_crop → 4×4 high-res patches (with positions)

Cross-image mode: coarse from image A, fine patches from image B (same class).
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image


class MSFT2LevelDataset(Dataset):
    """
    2-level MSFT dataset for FractalGen cross-image training.
    
    For each sample:
        - Select an anchor image A from the dataset
        - Produce 16×16 coarse view from A (avg pool 4×4 blocks)
        - Select a DIFFERENT image B from the SAME class
        - Extract K random 4×4 patches from B, recording their positions
        
    Yields:
        coarse_16x16:  (3, 16, 16) — average-pooled from 64×64
        fine_4x4:      (K, 3, 4, 4) — K high-res patches from same-class different image
        fine_positions:(K, 2)       — (row, col) in 16×16 grid for each patch
        label:         int          — class label
    """
    
    def __init__(
        self,
        root: str,
        img_size: int = 64,
        k_patches: int = 8,           # fine patches per sample
        pairing_mode: str = 'same_class',  # 'same_class', 'same_image', 'random'
        train: bool = True,
        normalize: bool = True,
    ):
        super().__init__()
        
        self.img_size = img_size
        self.k_patches = k_patches
        self.pairing_mode = pairing_mode
        self.normalize = normalize
        
        assert pairing_mode in ('same_class', 'same_image', 'random')
        
        # ImageNet normalization
        if normalize:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]))
        
        # Load dataset
        split = 'train' if train else 'val'
        dataset_path = f"{root}/{split}" if not root.endswith(split) else root
        
        transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size) if not train else T.RandomResizedCrop(
                img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            ),
            T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
            T.ToTensor(),
        ])
        
        self.dataset = ImageFolder(dataset_path, transform=transform)
        
        # For same_class mode: build class → indices mapping
        if pairing_mode == 'same_class':
            self.class_to_indices = {}
            for i, (_, label) in enumerate(self.dataset.samples):
                self.class_to_indices.setdefault(label, []).append(i)
            
            # Verify each class has at least 2 images (needed for cross-image)
            single_img_classes = [
                c for c, indices in self.class_to_indices.items()
                if len(indices) < 2
            ]
            if single_img_classes:
                print(f"Warning: {len(single_img_classes)} classes have <2 images. "
                      f"These will fall back to same-image pairing.")
        
        print(f"[MSFT2L] Dataset: {len(self.dataset)} images, "
              f"{len(self.dataset.classes)} classes, "
              f"pairing={pairing_mode}, k={k_patches}")
    
    def register_buffer(self, name, tensor):
        """Move buffer to the right device later."""
        setattr(self, name, tensor)
    
    def _normalize(self, img: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return (img - self.mean.view(3, 1, 1)) / self.std.view(3, 1, 1)
        return img
    
    def _to_coarse(self, img: torch.Tensor) -> torch.Tensor:
        """
        Convert 64×64 image to 16×16 coarse via 4×4 avg pooling.
        Equivalent to: divide into 16×16 patches of 4×4, take mean of each.
        """
        return F.avg_pool2d(img.unsqueeze(0), kernel_size=4, stride=4).squeeze(0)
    
    def _random_patch(self, img: torch.Tensor, size: int = 4) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extract a random size×size patch from img.
        Returns (patch, (row, col)) where row, col are in the 16×16 grid.
        """
        _, h, w = img.shape
        max_start = h - size
        row_pixel = np.random.randint(0, max_start + 1)
        col_pixel = np.random.randint(0, max_start + 1)
        
        patch = img[:, row_pixel:row_pixel+size, col_pixel:col_pixel+size]
        
        # Position in 16×16 grid: divide by 4
        grid_row = row_pixel // 4
        grid_col = col_pixel // 4
        
        return patch, (grid_row, grid_col)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        img_a, label = self.dataset[idx]
        img_a = self._normalize(img_a)
        
        # ── Coarse view: from anchor image A ──
        coarse = self._to_coarse(img_a)  # (3, 16, 16)
        
        # ── Fine patches: from image B (same class, different image) ──
        if self.pairing_mode == 'same_class':
            # Select a DIFFERENT image from the same class
            same_class = self.class_to_indices[label]
            if len(same_class) >= 2:
                # Exclude current idx
                candidates = [i for i in same_class if i != idx]
                img_b_idx = candidates[np.random.randint(len(candidates))]
            else:
                img_b_idx = idx  # fallback: same image
            
            img_b, _ = self.dataset[img_b_idx]
            img_b = self._normalize(img_b)
            
        elif self.pairing_mode == 'same_image':
            img_b = img_a  # same image (spatial alignment mode)
        else:  # 'random'
            random_idx = np.random.randint(len(self.dataset))
            img_b, _ = self.dataset[random_idx]
            img_b = self._normalize(img_b)
        
        # Extract K random patches from image B
        fine_patches = []
        positions = []
        for _ in range(self.k_patches):
            patch, pos = self._random_patch(img_b)
            fine_patches.append(patch)
            positions.append(pos)
        
        fine = torch.stack(fine_patches)  # (K, 3, 4, 4)
        positions = torch.tensor(positions, dtype=torch.long)  # (K, 2)
        
        return {
            'coarse': coarse,      # (3, 16, 16)
            'fine': fine,           # (K, 3, 4, 4)
            'positions': positions, # (K, 2)
            'label': label,         # int
        }


def msft2_collate_fn(batch: list) -> Dict:
    """
    Collate function for MSFT2LevelDataset.
    
    Input: list of dicts
    Output: batched dict with:
        coarse:    (B, 3, 16, 16)
        fine:      (B*K, 3, 4, 4)  — flattened
        positions: (B*K, 2)
        labels:    (B,)
    """
    coarse = torch.stack([item['coarse'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    
    # Flatten fine patches: (B, K, 3, 4, 4) → (B*K, 3, 4, 4)
    fine = torch.cat([item['fine'] for item in batch], dim=0)
    positions = torch.cat([item['positions'] for item in batch], dim=0)
    
    return {
        'coarse': coarse,
        'fine': fine,
        'positions': positions,
        'labels': labels,
    }


def create_dataloaders(
    data_root: str,
    batch_size: int = 128,
    k_patches: int = 8,
    img_size: int = 64,
    num_workers: int = 4,
    pairing_mode: str = 'same_class',
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and val dataloaders for MSFT training.
    
    Returns (train_loader, val_loader).
    val_loader is None if no val split is found.
    """
    try:
        train_dataset = MSFT2LevelDataset(
            root=data_root,
            img_size=img_size,
            k_patches=k_patches,
            pairing_mode=pairing_mode,
            train=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=msft2_collate_fn,
            drop_last=True,
        )
    except Exception as e:
        print(f"Warning: Could not create train loader: {e}")
        train_loader = None
    
    try:
        val_dataset = MSFT2LevelDataset(
            root=data_root,
            img_size=img_size,
            k_patches=k_patches,
            pairing_mode=pairing_mode,
            train=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=msft2_collate_fn,
            drop_last=False,
        )
    except Exception:
        val_loader = None
    
    return train_loader, val_loader
