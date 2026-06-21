"""
Cross-Image MSFT FractalGen: Fractal Generative Model trained on fragmented
multi-scale data from same-class but different images.

Analogy: physics cross-scale simulation where macroscopic measurements
(16×16 coarse) and microscopic measurements (4×4 high-res patches) come from
the same system type but different experiments.

Architecture: FractalGen(64, 4, 1) — identical to original.
Input adaptation: 16×16 coarse → nearest-neighbor upsample → 64×64 for L0.
Key change: L1 receives conditions from L0 at patch positions, but the 4×4
fine patches are from a DIFFERENT image (same class). Cross-level consistency
loss bridges the semantic gap.
"""

import sys
import os
import math
from functools import partial
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mar import MAR
from .ar import AR
from .pixelloss import PixelLoss


# ═══════════════════════════════════════════════════════════════════════
# Cross-Level Consistency Loss
# ═══════════════════════════════════════════════════════════════════════

class CrossLevelContrastiveLoss(nn.Module):
    """
    Contrastive loss between L0 conditions and L1 patch representations.
    
    Pulls together conditions and patch representations from the same class
    and spatial position, pushes apart those from different classes.
    
    This bridges the gap when L0's coarse image and L1's fine patch come from
    different images of the same class.
    """
    
    def __init__(self, cond_dim: int, proj_dim: int = 128, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
        # Project both condition and patch representations to a common space
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, proj_dim * 2),
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.patch_proj = nn.Sequential(
            nn.Linear(cond_dim, proj_dim * 2),  # cond_dim used for patch embedding too
            nn.GELU(),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        
    def forward(
        self,
        cond_l0: torch.Tensor,       # (N, cond_dim) - L0 conditions at fine patch positions
        patch_feat_l1: torch.Tensor, # (N, feat_dim) - L1 patch features (before PixelLoss)
        labels: torch.Tensor,        # (N,) - class labels
    ) -> torch.Tensor:
        """
        Args:
            cond_l0: L0 condition vectors at positions where we have fine patches
            patch_feat_l1: L1 intermediate representations of the fine patches
            labels: class label for each patch
        """
        # Project to common space
        z_cond = F.normalize(self.cond_proj(cond_l0), dim=-1)
        z_patch = F.normalize(self.patch_proj(patch_feat_l1), dim=-1)
        
        # Compute similarity matrix
        # For each (cond, patch) pair, positive = same class, negative = different class
        sim = (z_cond @ z_patch.T) / self.temperature  # (N, N)
        
        # Build positive mask: same class label
        labels = labels.contiguous().view(-1, 1)
        pos_mask = (labels == labels.T).float()  # (N, N)
        
        # For numerical stability, subtract max
        sim = sim - sim.max(dim=-1, keepdim=True)[0]
        
        # InfoNCE loss (treat patch→cond and cond→patch symmetrically)
        exp_sim = torch.exp(sim)
        
        # Denominator: sum over all pairs
        denom = exp_sim.sum(dim=-1, keepdim=True)  # (N, 1)
        
        # Numerator: sum over positive pairs only
        # Use pos_mask to select positive pairs (excluding self if same sample)
        # For cross-image, a cond and its corresponding patch may be different indices
        # so we keep the diagonal + same-class off-diagonal
        log_prob = sim - torch.log(denom + 1e-8)
        
        # Mean over positive pairs
        n_pos = pos_mask.sum(dim=-1).clamp(min=1)
        loss = -(pos_mask * log_prob).sum(dim=-1) / n_pos
        
        return loss.mean()


class CrossLevelMMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss between L0 condition distribution and
    L1 patch representation distribution, computed per class.
    
    Lighter alternative to contrastive loss — no learnable parameters.
    Useful as a regularizer.
    """
    
    def __init__(self, kernel: str = 'rbf', sigma: float = 1.0):
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma
        
    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix between x and y."""
        x_norm = (x ** 2).sum(-1).view(-1, 1)
        y_norm = (y ** 2).sum(-1).view(1, -1)
        dist = x_norm + y_norm - 2.0 * (x @ y.T)
        dist = dist.clamp(min=0)
        return torch.exp(-dist / (2.0 * self.sigma ** 2))
    
    def forward(
        self,
        cond_l0: torch.Tensor,
        patch_feat_l1: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-class MMD and average.
        """
        unique_labels = labels.unique()
        total_loss = 0.0
        
        for label in unique_labels:
            mask = (labels == label)
            cond_c = cond_l0[mask]
            patch_c = patch_feat_l1[mask]
            
            if len(cond_c) < 2 or len(patch_c) < 2:
                continue
            
            # Normalize
            cond_c = F.normalize(cond_c, dim=-1)
            patch_c = F.normalize(patch_c, dim=-1)
            
            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
            k_xx = self._rbf_kernel(cond_c, cond_c).mean()
            k_yy = self._rbf_kernel(patch_c, patch_c).mean()
            k_xy = self._rbf_kernel(cond_c, patch_c).mean()
            
            mmd2 = k_xx + k_yy - 2 * k_xy
            total_loss += mmd2.clamp(min=0)
        
        return total_loss / max(len(unique_labels), 1)


# ═══════════════════════════════════════════════════════════════════════
# FractalGen with Cross-Image MSFT Support
# ═══════════════════════════════════════════════════════════════════════

class FractalGenMSFT(nn.Module):
    """
    FractalGen adapted for Multi-Scale Fragment Training with cross-image data.
    
    Training data:
      - coarse_16x16: (B, 3, 16, 16) — average-pooled from 64×64
      - fine_4x4:     (B*k, 3, 4, 4)   — k high-res patches per image
      - positions:    (B*k, 2)          — (row, col) in 16×16 grid
      - labels:       (B,)              — class labels
    
    Architecture: FractalGen(64, 4, 1) — identical to original.
    During L0 forward: upsample 16×16 → 64×64 (nearest neighbor).
    During L1 forward: conditions from L0 at fine patch positions.
    """
    
    def __init__(
        self,
        embed_dim_list=(512, 256, 64),
        num_blocks_list=(12, 4, 2),
        num_heads_list=(8, 4, 2),
        generator_type_list=("mar", "mar", "ar"),
        class_num=200,
        attn_dropout=0.1,
        proj_dropout=0.1,
        label_drop_prob=0.1,
        r_weight=1.0,
        grad_checkpointing=False,
        consistency_loss_type='contrastive',  # 'contrastive', 'mmd', or 'none'
        consistency_weight=0.1,
    ):
        super().__init__()
        
        self.embed_dim_list = embed_dim_list
        self.class_num = class_num
        self.consistency_loss_type = consistency_loss_type
        self.consistency_weight = consistency_weight
        
        # ── Level 0: processes 64×64 (upsampled from 16×16 coarse) ──
        # Class embedding at top level
        self.class_emb = nn.Embedding(class_num, embed_dim_list[0])
        self.label_drop_prob = label_drop_prob
        self.fake_latent = nn.Parameter(torch.zeros(1, embed_dim_list[0]))
        torch.nn.init.normal_(self.class_emb.weight, std=0.02)
        torch.nn.init.normal_(self.fake_latent, std=0.02)
        
        # L0 generator: MAR, patchify 64×64 → 16×16 grid of 4×4 patches → 256 tokens
        gen0_cls = MAR if generator_type_list[0] == 'mar' else AR
        self.generator_l0 = gen0_cls(
            seq_len=(64 // 4) ** 2,   # 256
            patch_size=4,
            cond_embed_dim=embed_dim_list[0],
            embed_dim=embed_dim_list[0],
            num_blocks=num_blocks_list[0],
            num_heads=num_heads_list[0],
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            guiding_pixel=False,
            num_conds=1,
            grad_checkpointing=grad_checkpointing,
        )
        
        # ── Level 1: processes individual 4×4 patches ──
        gen1_cls = MAR if generator_type_list[1] == 'mar' else AR
        self.generator_l1 = gen1_cls(
            seq_len=(4 // 1) ** 2,    # 16
            patch_size=1,
            cond_embed_dim=embed_dim_list[0],  # receives L0 conditions
            embed_dim=embed_dim_list[1],
            num_blocks=num_blocks_list[1],
            num_heads=num_heads_list[1],
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            guiding_pixel=False,
            num_conds=5,  # middle, top, right, bottom, left from L0
            grad_checkpointing=grad_checkpointing,
        )
        
        # ── Pixel Loss (Level 2) ──
        self.pixel_loss = PixelLoss(
            c_channels=embed_dim_list[1],
            depth=num_blocks_list[2],
            width=embed_dim_list[2],
            num_heads=num_heads_list[2],
            r_weight=r_weight,
        )
        
        # ── Cross-level consistency loss ──
        if consistency_loss_type == 'contrastive':
            self.consistency_loss = CrossLevelContrastiveLoss(
                cond_dim=embed_dim_list[0],
                proj_dim=min(embed_dim_list[0] // 4, 256),
            )
        elif consistency_loss_type == 'mmd':
            self.consistency_loss = CrossLevelMMDLoss()
        else:
            self.consistency_loss = None
        
        self._init_weights()
        
    def _init_weights(self):
        self.apply(self._init_weights_fn)
        
    def _init_weights_fn(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
    
    # ── Public forward: training on MSFT data ──
    
    def forward(
        self,
        coarse_16x16: torch.Tensor,
        fine_4x4: torch.Tensor,
        positions: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict:
        """
        Single training step with cross-image MSFT data.
        
        Args:
            coarse_16x16: (B, 3, 16, 16) — coarse images
            fine_4x4:     (M, 3, 4, 4)   — fine patches (M = B * k_patches_per_image)
            positions:    (M, 2)          — (row, col) in 16×16 grid
            labels:       (B,)            — class labels (0..199)
        
        Returns:
            dict with keys: loss_l0, loss_l1, loss_pixel, loss_consistency, loss_total
        """
        B = coarse_16x16.shape[0]
        M = fine_4x4.shape[0]
        device = coarse_16x16.device
        
        # ── Prepare class embeddings ──
        class_emb = self.class_emb(labels)  # (B, embed_dim[0])
        if self.training:
            drop_mask = (torch.rand(B, device=device) < self.label_drop_prob).float()
            class_emb = drop_mask.view(-1, 1) * self.fake_latent + (1 - drop_mask.view(-1, 1)) * class_emb
        
        # ── L0: Process coarse image ──
        # Upsample 16×16 → 64×64 (nearest neighbor)
        coarse_64x64 = F.interpolate(
            coarse_16x16, size=(64, 64), mode='nearest'
        )
        
        # MAR forward: patchify, mask, predict
        # Returns: (masked_patches, conditions, _)
        patches_l0, cond_list_l0, _ = self.generator_l0(
            coarse_64x64, [class_emb]
        )
        # patches_l0: (num_masked, 3, 4, 4) — the ground-truth 4×4 patches at masked positions
        # cond_list_l0: list of 5 tensors, each (num_masked, embed_dim[0])
        #   [0]=middle, [1]=top, [2]=right, [3]=bottom, [4]=left
        
        # ── Compute L0 loss via recursive chain: L0 → L1 → PixelLoss ──
        # patches_l0: (num_masked_L0, 3, 4, 4) — coarse ground-truth 4×4 patches
        # cond_list_l0: list of 5 tensors — L0 conditions for those patches
        # We must pass through L1 (which patchifies 4×4 → 16×1×1 tokens) then PixelLoss.
        patches_l1_from_l0, cond_list_l1_from_l0, _ = self.generator_l1(
            patches_l0, cond_list_l0
        )
        # patches_l1_from_l0: (num_masked_L1, 3, 1, 1) — individual pixels
        loss_l0 = self.pixel_loss(patches_l1_from_l0, cond_list_l1_from_l0)
        
        # ── L0 → L1: Extract conditions at fine patch positions ──
        # We need ALL L0 conditions, not just masked ones.
        # Rerun L0 in "all-condition" mode.
        with torch.set_grad_enabled(self.training):
            all_conds = self._get_all_l0_conditions(coarse_64x64, class_emb)
        # all_conds: list of 5 tensors, each (B, 256, embed_dim[0])
        # Reshape to (B, 16, 16, embed_dim[0]) for spatial indexing
        
        cond_grid = []
        for c in all_conds:
            cond_grid.append(c.reshape(B, 16, 16, -1))  # each: (B, 16, 16, embed_dim[0])
        
        # For each fine patch, extract its condition from the grid
        # fine_4x4 has M patches. We need to map each to its source image in the batch.
        k_per_img = M // B
        img_indices = torch.arange(B, device=device).repeat_interleave(k_per_img)
        
        extracted_conds = []
        for c_grid in cond_grid:
            # c_grid: (B, 16, 16, embed_dim[0])
            extracted = c_grid[img_indices, positions[:, 0], positions[:, 1]]  # (M, embed_dim[0])
            extracted_conds.append(extracted)
        
        # ── L1: Process fine patches with extracted conditions ──
        # L1 expects patches in shape (M, 3, 4, 4) and cond_list
        patches_l1, cond_list_l1, _ = self.generator_l1(fine_4x4, extracted_conds)
        # patches_l1: (num_masked_in_l1, 3, 1, 1) — individual pixels
        # cond_list_l1: list of 5 tensors, each (num_masked_in_l1, embed_dim[1])
        
        # ── Pixel Loss on L1 output ──
        loss_pixel = self.pixel_loss(patches_l1, cond_list_l1)
        
        # ── L1 intermediate loss (MAR's internal diff loss on L1 tokens) ──
        # This is implicitly included in the pixel loss, but we can also compute
        # a separate L1 reconstruction loss if needed.
        # For now, pixel_loss covers L1's generation quality.
        
        # ── Cross-level consistency loss ──
        if self.consistency_loss is not None and self.consistency_weight > 0:
            # Use the middle condition from L0, and a simple encoding of the fine patches
            cond_for_consistency = extracted_conds[0]  # middle condition (M, embed_dim[0])
            
            # Simple patch feature: mean + std of each channel, flattened
            fine_flat = fine_4x4.view(M, 3, 16)  # (M, 3, 16)
            patch_feat = torch.cat([
                fine_flat.mean(dim=-1),  # (M, 3)
                fine_flat.std(dim=-1),   # (M, 3)
            ], dim=-1)  # (M, 6)
            # Pad to match cond_dim for the consistency loss projector
            if patch_feat.shape[-1] < self.embed_dim_list[0]:
                patch_feat = F.pad(patch_feat, (0, self.embed_dim_list[0] - patch_feat.shape[-1]))
            elif patch_feat.shape[-1] > self.embed_dim_list[0]:
                patch_feat = patch_feat[:, :self.embed_dim_list[0]]
            
            # Broadcast labels for fine patches
            fine_labels = labels[img_indices]  # (M,)
            
            loss_consistency = self.consistency_loss(
                cond_for_consistency, patch_feat, fine_labels
            )
        else:
            loss_consistency = torch.tensor(0.0, device=device)
        
        # ── Total loss ──
        loss_total = loss_l0 + loss_pixel + self.consistency_weight * loss_consistency
        
        return {
            'loss_l0': loss_l0,
            'loss_pixel': loss_pixel,
            'loss_consistency': loss_consistency,
            'loss_total': loss_total,
        }
    
    def _get_all_l0_conditions(
        self, imgs_64: torch.Tensor, class_emb: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Get L0 conditions for ALL 256 positions (no masking).
        Used to extract conditions at fine patch positions.
        """
        B = imgs_64.shape[0]
        patches = self.generator_l0.patchify(imgs_64)  # (B, 256, 48)
        
        # No masking — predict all positions
        mask = torch.zeros(B, self.generator_l0.seq_len, device=imgs_64.device)
        
        # Use L0's predict method
        cond_list = self.generator_l0.predict(patches, mask, [class_emb])
        # Returns list of 5 tensors, each (B, 256, embed_dim[0])
        
        return cond_list
    
    # ── Generation ──
    
    @torch.no_grad()
    def generate(
        self,
        class_labels: torch.Tensor,
        num_iter_l0: int = 64,
        num_iter_l1: int = 16,
        cfg: float = 1.0,
        temperature: float = 1.0,
        filter_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """
        Generate 64×64 images pixel-by-pixel.
        
        Args:
            class_labels: (B,) — class labels (0..199)
            num_iter_l0: MAR iterations for L0 (256 positions, but usually 64-128)
            num_iter_l1: MAR iterations for L1 (16 pixels, usually 8-16)
            cfg: classifier-free guidance scale
            temperature: sampling temperature
            filter_threshold: threshold for CFG filtering
        
        Returns:
            images: (B, 3, 64, 64), normalized with ImageNet stats
        """
        self.eval()
        B = class_labels.shape[0]
        device = class_labels.device
        
        class_emb = self.class_emb(class_labels)  # (B, embed_dim[0])
        
        # For CFG, duplicate with fake latent
        if cfg != 1.0:
            class_emb_cfg = torch.cat([
                class_emb,
                self.fake_latent.repeat(B, 1)
            ], dim=0)
            B_eff = B * 2
        else:
            class_emb_cfg = class_emb
            B_eff = B
        
        # ── L0: Generate 16×16 grid of 4×4 patch tokens ──
        seq_len_l0 = self.generator_l0.seq_len  # 256
        mask_l0 = torch.ones(B_eff, seq_len_l0, device=device)
        patches_l0 = torch.zeros(B_eff, seq_len_l0, 3 * 4 * 4, device=device)
        orders_l0 = self.generator_l0.sample_orders(B_eff)
        num_iter_l0 = min(seq_len_l0, num_iter_l0)
        
        for step in range(num_iter_l0):
            cur_patches = patches_l0.clone()
            
            # Get conditions from current state
            cond_list = self.generator_l0.predict(
                patches_l0, mask_l0, [class_emb_cfg]
            )
            
            # Cosine schedule for mask ratio
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter_l0)
            mask_len = int(seq_len_l0 * mask_ratio)
            mask_len = max(1, min(int(mask_l0[:B].sum().item()) - 1, mask_len))
            
            # Determine which tokens to predict this step
            if step >= num_iter_l0 - 1:
                mask_to_pred = mask_l0[:B].bool()
            else:
                # XOR: tokens that are currently masked but won't be next
                mask_next = torch.zeros(B, seq_len_l0, device=device)
                for b in range(B):
                    mask_next[b, orders_l0[b, :mask_len]] = 1.0
                mask_to_pred = mask_l0[:B].bool() ^ mask_next.bool()
            
            mask_l0 = mask_next if step < num_iter_l0 - 1 else mask_l0[:B] * 0
            
            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            
            # Generate patches via L1 for the predicted positions
            # For each position to predict, we need to run L1 recursively
            num_to_pred = int(mask_to_pred[:B].sum().item())
            if num_to_pred > 0:
                # Extract conditions at predicted positions
                cond_for_l1 = []
                for c in cond_list:
                    c_selected = c[mask_to_pred]  # (num_to_pred, embed_dim[0])
                    cond_for_l1.append(c_selected)
                
                # Generate 4×4 patch via L1
                generated_patch = self._generate_l1_patch(
                    cond_for_l1, num_iter_l1, cfg, temperature, filter_threshold,
                    B_eff_override=num_to_pred * (2 if cfg != 1.0 else 1)
                )
                # generated_patch: (num_to_pred, 3, 4, 4)
                
                # Place back
                generated_flat = generated_patch.view(num_to_pred, -1)  # (num_to_pred, 48)
                
                if cfg != 1.0:
                    mask_to_pred_orig, _ = mask_to_pred.chunk(2, dim=0)
                else:
                    mask_to_pred_orig = mask_to_pred
                
                cur_patches[mask_to_pred_orig] = generated_flat.to(cur_patches.dtype)
            
            patches_l0 = cur_patches.clone()
        
        # Unpatchify L0 output to 64×64
        if cfg != 1.0:
            patches_l0 = patches_l0[:B]
        
        images = self.generator_l0.unpatchify(patches_l0)  # (B, 3, 64, 64)
        return images
    
    def _generate_l1_patch(
        self,
        cond_list: List[torch.Tensor],
        num_iter: int,
        cfg: float,
        temperature: float,
        filter_threshold: float,
        B_eff_override: int = None,
    ) -> torch.Tensor:
        """
        Generate a single 4×4 patch given L0 conditions.
        """
        seq_len_l1 = self.generator_l1.seq_len  # 16
        N = cond_list[0].shape[0]
        
        if B_eff_override is not None:
            B_eff = B_eff_override
        elif cfg != 1.0:
            B_eff = N * 2
        else:
            B_eff = N
        
        mask_l1 = torch.ones(B_eff, seq_len_l1, device=cond_list[0].device)
        patches_l1 = torch.zeros(B_eff, seq_len_l1, 3, device=cond_list[0].device)
        orders_l1 = torch.argsort(torch.rand(B_eff, seq_len_l1, device=cond_list[0].device), dim=1).long()
        num_iter = min(seq_len_l1, num_iter)
        
        for step in range(num_iter):
            cur_patches = patches_l1.clone()
            
            if cfg != 1.0:
                patches_l1_in = torch.cat([patches_l1[:N], patches_l1[:N]], dim=0)
                mask_l1_in = torch.cat([mask_l1[:N], mask_l1[:N]], dim=0)
            else:
                patches_l1_in = patches_l1
                mask_l1_in = mask_l1
            
            cond_next = self.generator_l1.predict(patches_l1_in, mask_l1_in, cond_list)
            
            mask_ratio = math.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = max(1, min(int(mask_l1[:N].sum().item()) - 1, int(seq_len_l1 * mask_ratio)))
            
            if step >= num_iter - 1:
                mask_to_pred = mask_l1[:N].bool()
            else:
                mask_next = torch.zeros(N, seq_len_l1, device=cond_list[0].device)
                for b in range(N):
                    mask_next[b, orders_l1[b, :mask_len]] = 1.0
                mask_to_pred = mask_l1[:N].bool() ^ mask_next.bool()
            
            mask_l1 = mask_next if step < num_iter - 1 else mask_l1[:N] * 0
            
            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            
            # Sample pixels via PixelLoss
            num_to_pred = int(mask_to_pred[:N].sum().item())
            if num_to_pred > 0:
                pixel_conds = []
                for c in cond_next:
                    pixel_conds.append(c[mask_to_pred])
                
                sampled_pixels = self.pixel_loss.sample(
                    pixel_conds, temperature, cfg, filter_threshold
                )  # (num_to_pred, 3)
                
                if cfg != 1.0:
                    mask_to_pred_orig, _ = mask_to_pred.chunk(2, dim=0)
                else:
                    mask_to_pred_orig = mask_to_pred
                
                cur_patches[mask_to_pred_orig] = sampled_pixels.to(cur_patches.dtype)
            
            patches_l1 = cur_patches.clone()
        
        # Build output (N, 3, 4, 4) directly — avoid permute entirely
        if cfg != 1.0:
            patches_l1 = patches_l1[:N]
        out = torch.zeros(N, 3, 4, 4, device=patches_l1.device, dtype=patches_l1.dtype)
        for h in range(4):
            for w in range(4):
                out[:, :, h, w] = patches_l1[:N, h * 4 + w, :]
        return out


# ═══════════════════════════════════════════════════════════════════════
# Model factory functions
# ═══════════════════════════════════════════════════════════════════════

def fractalmar_msft_5090(class_num=200, **kwargs):
    """Small model optimized for single RTX 5090 (32GB). ~15M params."""
    return FractalGenMSFT(
        embed_dim_list=(512, 256, 64),
        num_blocks_list=(12, 4, 2),
        num_heads_list=(8, 4, 2),
        generator_type_list=("mar", "mar", "ar"),
        class_num=class_num,
        **kwargs,
    )


def fractalmar_msft_tiny(class_num=200, **kwargs):
    """Tiny model for quick experiments (Colab T4). ~4M params."""
    return FractalGenMSFT(
        embed_dim_list=(256, 128, 32),
        num_blocks_list=(6, 2, 1),
        num_heads_list=(4, 2, 2),
        generator_type_list=("mar", "mar", "ar"),
        class_num=class_num,
        **kwargs,
    )


def fractalmar_msft_medium(class_num=200, **kwargs):
    """Medium model for A100/H100. ~50M params."""
    return FractalGenMSFT(
        embed_dim_list=(768, 384, 128),
        num_blocks_list=(20, 6, 3),
        num_heads_list=(12, 6, 4),
        generator_type_list=("mar", "mar", "ar"),
        class_num=class_num,
        **kwargs,
    )
