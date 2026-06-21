"""
MSFT Cross-Image Training Script for FractalGen.

Three-phase training:
  Phase 1: L0 pretraining on coarse 16×16 images (class-conditional)
  Phase 2: L1+PixelLoss pretraining (L0 frozen, cross-image fine patches)
  Phase 3: Joint fine-tuning with cross-level consistency loss

Usage:
    python train_msft.py --phase 1 --data_root ./data/tiny-imagenet-200
    python train_msft.py --phase 2 --data_root ./data/tiny-imagenet-200 --resume output/phase1_best.pth
    python train_msft.py --phase 3 --data_root ./data/tiny-imagenet-200 --resume output/phase2_best.pth
"""

import os
import sys
import math
import time
import argparse
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

# ── Add project to path ──
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.msft_dataset import MSFT2LevelDataset, msft2_collate_fn, create_dataloaders
from models.fractalgen_msft import (
    FractalGenMSFT,
    fractalmar_msft_5090,
    fractalmar_msft_tiny,
    fractalmar_msft_medium,
)


# ═══════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════

def get_default_config():
    return {
        # Data
        'data_root': './data/tiny-imagenet-200',
        'img_size': 64,
        'k_patches': 8,
        'class_num': 200,
        'pairing_mode': 'same_class',  # same_class, same_image, random
        
        # Model
        'model_size': '5090',  # tiny, 5090, medium
        'consistency_loss': 'contrastive',  # contrastive, mmd, none
        'consistency_weight': 0.1,
        'grad_checkpointing': False,
        
        # Training
        'batch_size': 64,
        'num_workers': 4,
        'accum_steps': 1,         # gradient accumulation steps
        'lr': 5e-4,
        'lr_min': 1e-6,
        'weight_decay': 0.05,
        'epochs': 200,
        'warmup_epochs': 5,
        'grad_clip': 3.0,
        
        # Generation eval
        'eval_every': 20,
        'num_eval_images': 16,
        'num_iter_l0': 64,
        'num_iter_l1': 8,
        'cfg': 1.0,
        'temperature': 1.0,
        
        # Output
        'output_dir': './output',
        'log_dir': './logs',
        'save_every': 20,
        
        # Device
        'device': 'cuda',
        'use_amp': True,
    }


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: L0 Pretraining on Coarse Images
# ═══════════════════════════════════════════════════════════════════════

class L0PretrainWrapper(nn.Module):
    """
    Phase 1: Pretrain only the L0 generator on coarse 16×16 images.
    
    Feeds 16×16 (upsampled to 64×64) through L0 MAR, computes MAR loss.
    L1 and PixelLoss are present but frozen.
    """
    
    def __init__(self, model: FractalGenMSFT):
        super().__init__()
        self.model = model
    
    def forward(self, coarse_16x16: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Simple L0 forward: upsample coarse, run through L0 MAR.
        """
        B = coarse_16x16.shape[0]
        device = coarse_16x16.device
        
        # Class embedding
        class_emb = self.model.class_emb(labels)
        if self.training:
            drop_mask = (torch.rand(B, device=device) < self.model.label_drop_prob).float()
            class_emb = drop_mask.view(-1, 1) * self.model.fake_latent + \
                       (1 - drop_mask.view(-1, 1)) * class_emb
        
        # Upsample to 64×64
        coarse_64 = F.interpolate(coarse_16x16, size=(64, 64), mode='nearest')
        
        # L0 MAR forward → returns masked 4×4 patches
        patches_l0, cond_list_l0, _ = self.model.generator_l0(coarse_64, [class_emb])
        
        # L0 loss via recursive chain: L0 patches → L1 → PixelLoss
        patches_l1, cond_list_l1, _ = self.model.generator_l1(patches_l0, cond_list_l0)
        loss_l0 = self.model.pixel_loss(patches_l1, cond_list_l1)
        
        return loss_l0


def train_phase1(args, config):
    """Phase 1: Pretrain L0 on coarse images."""
    print("\n" + "="*60)
    print("Phase 1: L0 Pretraining on Coarse Images")
    print("="*60)
    
    # ── Data ──
    # Phase 1 only uses coarse images; we can use the MSFT dataset but only take coarse
    train_loader, val_loader = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        k_patches=config['k_patches'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        pairing_mode=config['pairing_mode'],
    )
    
    # ── Model ──
    model = create_model(config)
    wrapper = L0PretrainWrapper(model)
    wrapper.to(config['device'])
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params/1e6:.1f}M")
    
    if args.resume:
        load_checkpoint(model, args.resume, config['device'])
    
    # ── Optimizer ──
    # Only optimize L0 + class_emb + pixel_loss (used for L0 loss)
    optimizer = torch.optim.AdamW(
        wrapper.parameters(), 
        lr=config['lr'],
        betas=(0.9, 0.95),
        weight_decay=config['weight_decay'],
    )
    
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'] - config['warmup_epochs'],
        eta_min=config['lr_min'],
    )
    
    # ── Train ──
    log_writer = SummaryWriter(os.path.join(config['log_dir'], 'phase1'))
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, config['epochs'] + 1):
        wrapper.train()
        epoch_loss = 0.0
        
        # Warmup
        if epoch <= config['warmup_epochs']:
            lr_scale = epoch / config['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = config['lr'] * lr_scale
        
        for batch_idx, batch in enumerate(train_loader):
            coarse = batch['coarse'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            
            with torch.cuda.amp.autocast(enabled=config['use_amp']):
                loss = wrapper(coarse, labels)
            
            loss = loss / config['accum_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['accum_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(wrapper.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config['accum_steps']
            global_step += 1
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{config['epochs']} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        if epoch > config['warmup_epochs']:
            scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\n  Epoch {epoch} | Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        log_writer.add_scalar('loss/train', avg_loss, epoch)
        log_writer.add_scalar('lr', current_lr, epoch)
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss,
                          os.path.join(config['output_dir'], 'phase1_best.pth'))
        
        # Periodic save
        if epoch % config['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss,
                          os.path.join(config['output_dir'], f'phase1_epoch{epoch}.pth'))
    
    # Final save
    save_checkpoint(model, optimizer, config['epochs'], best_loss,
                  os.path.join(config['output_dir'], 'phase1_final.pth'))
    print(f"\nPhase 1 complete. Best loss: {best_loss:.4f}")
    
    log_writer.close()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: L1+PixelLoss Pretraining (L0 Frozen)
# ═══════════════════════════════════════════════════════════════════════

def train_phase2(args, config):
    """Phase 2: Pretrain L1+PixelLoss. L0 frozen, cross-image data."""
    print("\n" + "="*60)
    print("Phase 2: L1 + PixelLoss Pretraining (L0 frozen)")
    print("="*60)
    
    # ── Data ──
    train_loader, val_loader = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        k_patches=config['k_patches'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        pairing_mode=config['pairing_mode'],
    )
    
    # ── Model ──
    model = create_model(config)
    model.to(config['device'])
    
    # Load Phase 1 checkpoint
    if args.resume:
        load_checkpoint(model, args.resume, config['device'])
    else:
        # Try to find phase1 checkpoint
        phase1_path = os.path.join(config['output_dir'], 'phase1_best.pth')
        if os.path.exists(phase1_path):
            load_checkpoint(model, phase1_path, config['device'])
        else:
            print("Warning: No Phase 1 checkpoint found. Training from scratch.")
    
    # Freeze L0
    for name, param in model.named_parameters():
        if name.startswith('generator_l0') or name.startswith('class_emb'):
            param.requires_grad = False
    
    # Verify frozen params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.1f}M / {total/1e6:.1f}M total")
    
    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr'] * 0.2,  # Lower LR for fine-tuning
        betas=(0.9, 0.95),
        weight_decay=config['weight_decay'],
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - config['warmup_epochs'],
        eta_min=config['lr_min'],
    )
    
    # ── Train ──
    log_writer = SummaryWriter(os.path.join(config['log_dir'], 'phase2'))
    best_loss = float('inf')
    global_step = 0
    
    # Disable consistency loss in phase 2 (L0 is frozen, conditions are fixed)
    model.consistency_weight = 0.0
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        # Keep L0 in eval mode (frozen, no dropout)
        model.generator_l0.eval()
        
        epoch_loss = 0.0
        epoch_loss_l0 = 0.0
        epoch_loss_pixel = 0.0
        
        if epoch <= config['warmup_epochs']:
            lr_scale = epoch / config['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = config['lr'] * 0.2 * lr_scale
        
        for batch_idx, batch in enumerate(train_loader):
            coarse = batch['coarse'].to(config['device'])
            fine = batch['fine'].to(config['device'])
            positions = batch['positions'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            
            with torch.cuda.amp.autocast(enabled=config['use_amp']):
                losses = model(coarse, fine, positions, labels)
                loss = losses['loss_total']
            
            loss = loss / config['accum_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['accum_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    config['grad_clip']
                )
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config['accum_steps']
            epoch_loss_l0 += losses['loss_l0'].item()
            epoch_loss_pixel += losses['loss_pixel'].item()
            global_step += 1
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{config['epochs']} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"L0: {losses['loss_l0'].item():.4f} | "
                      f"Pixel: {losses['loss_pixel'].item():.4f} | "
                      f"Total: {loss.item():.4f}")
        
        if epoch > config['warmup_epochs']:
            scheduler.step()
        
        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_l0 = epoch_loss_l0 / n
        avg_pixel = epoch_loss_pixel / n
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n  Epoch {epoch} | L0: {avg_l0:.4f} | Pixel: {avg_pixel:.4f} | "
              f"Total: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        log_writer.add_scalar('loss/total', avg_loss, epoch)
        log_writer.add_scalar('loss/l0', avg_l0, epoch)
        log_writer.add_scalar('loss/pixel', avg_pixel, epoch)
        log_writer.add_scalar('lr', current_lr, epoch)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss,
                          os.path.join(config['output_dir'], 'phase2_best.pth'))
        
        if epoch % config['save_every'] == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss,
                          os.path.join(config['output_dir'], f'phase2_epoch{epoch}.pth'))
        
        # Periodic generation eval
        if epoch % config['eval_every'] == 0:
            generate_and_save(model, config, epoch, log_writer)
    
    save_checkpoint(model, optimizer, config['epochs'], best_loss,
                  os.path.join(config['output_dir'], 'phase2_final.pth'))
    print(f"\nPhase 2 complete. Best loss: {best_loss:.4f}")
    
    log_writer.close()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Joint Fine-tuning with Consistency Loss
# ═══════════════════════════════════════════════════════════════════════

def train_phase3(args, config):
    """Phase 3: Joint fine-tuning. All parameters unfrozen, consistency loss active."""
    print("\n" + "="*60)
    print("Phase 3: Joint Fine-tuning with Cross-Level Consistency")
    print("="*60)
    
    # ── Data ──
    train_loader, val_loader = create_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'] // 2,  # Smaller batch for full model
        k_patches=config['k_patches'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        pairing_mode=config['pairing_mode'],
    )
    
    # ── Model ──
    model = create_model(config)
    model.to(config['device'])
    
    if args.resume:
        load_checkpoint(model, args.resume, config['device'])
    else:
        phase2_path = os.path.join(config['output_dir'], 'phase2_best.pth')
        if os.path.exists(phase2_path):
            load_checkpoint(model, phase2_path, config['device'])
        else:
            print("Warning: No Phase 2 checkpoint found.")
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    # Restore consistency weight
    model.consistency_weight = config['consistency_weight']
    
    # ── Optimizer (lower LR for joint training) ──
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'] * 0.04,  # Very low LR
        betas=(0.9, 0.95),
        weight_decay=config['weight_decay'],
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['epochs'] - config['warmup_epochs'],
        eta_min=config['lr_min'],
    )
    
    # ── Train ──
    log_writer = SummaryWriter(os.path.join(config['log_dir'], 'phase3'))
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0
        epoch_loss_l0 = 0.0
        epoch_loss_pixel = 0.0
        epoch_loss_consist = 0.0
        
        if epoch <= config['warmup_epochs']:
            lr_scale = epoch / config['warmup_epochs']
            for pg in optimizer.param_groups:
                pg['lr'] = config['lr'] * 0.04 * lr_scale
        
        for batch_idx, batch in enumerate(train_loader):
            coarse = batch['coarse'].to(config['device'])
            fine = batch['fine'].to(config['device'])
            positions = batch['positions'].to(config['device'])
            labels = batch['labels'].to(config['device'])
            
            with torch.cuda.amp.autocast(enabled=config['use_amp']):
                losses = model(coarse, fine, positions, labels)
                loss = losses['loss_total']
            
            loss = loss / config['accum_steps']
            loss.backward()
            
            if (batch_idx + 1) % config['accum_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config['accum_steps']
            epoch_loss_l0 += losses['loss_l0'].item()
            epoch_loss_pixel += losses['loss_pixel'].item()
            epoch_loss_consist += losses['loss_consistency'].item()
            global_step += 1
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch}/{config['epochs']} | "
                      f"Batch {batch_idx}/{len(train_loader)} | "
                      f"L0: {losses['loss_l0'].item():.4f} | "
                      f"Pixel: {losses['loss_pixel'].item():.4f} | "
                      f"Consist: {losses['loss_consistency'].item():.4f} | "
                      f"Total: {loss.item():.4f}")
        
        if epoch > config['warmup_epochs']:
            scheduler.step()
        
        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_l0 = epoch_loss_l0 / n
        avg_pixel = epoch_loss_pixel / n
        avg_consist = epoch_loss_consist / n
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n  Epoch {epoch} | L0: {avg_l0:.4f} | Pixel: {avg_pixel:.4f} | "
              f"Consist: {avg_consist:.4f} | Total: {avg_loss:.4f} | LR: {current_lr:.2e}")
        
        log_writer.add_scalar('loss/total', avg_loss, epoch)
        log_writer.add_scalar('loss/l0', avg_l0, epoch)
        log_writer.add_scalar('loss/pixel', avg_pixel, epoch)
        log_writer.add_scalar('loss/consistency', avg_consist, epoch)
        log_writer.add_scalar('lr', current_lr, epoch)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch, avg_loss,
                          os.path.join(config['output_dir'], 'phase3_best.pth'))
        
        if epoch % config['eval_every'] == 0:
            generate_and_save(model, config, epoch, log_writer)
    
    save_checkpoint(model, optimizer, config['epochs'], best_loss,
                  os.path.join(config['output_dir'], 'phase3_final.pth'))
    print(f"\nPhase 3 complete. Best loss: {best_loss:.4f}")
    
    log_writer.close()
    return model


# ═══════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════

def create_model(config: dict) -> FractalGenMSFT:
    """Create model based on config."""
    model_size = config['model_size']
    kwargs = dict(
        class_num=config['class_num'],
        consistency_loss_type=config['consistency_loss'],
        consistency_weight=config['consistency_weight'],
        grad_checkpointing=config['grad_checkpointing'],
    )
    
    if model_size == 'tiny':
        return fractalmar_msft_tiny(**kwargs)
    elif model_size == '5090':
        return fractalmar_msft_5090(**kwargs)
    elif model_size == 'medium':
        return fractalmar_msft_medium(**kwargs)
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def save_checkpoint(model, optimizer, epoch, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"  [✓] Saved checkpoint: {path}")


def load_checkpoint(model, path, device='cuda'):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"  [✓] Loaded checkpoint from {path} (epoch {ckpt.get('epoch', '?')})")


@torch.no_grad()
def generate_and_save(model, config, epoch, log_writer=None):
    """Generate sample images for visual evaluation."""
    model.eval()
    
    num_images = config['num_eval_images']
    device = config['device']
    
    # Generate from random classes
    class_labels = torch.randint(0, config['class_num'], (num_images,)).to(device)
    
    try:
        images = model.generate(
            class_labels=class_labels,
            num_iter_l0=config['num_iter_l0'],
            num_iter_l1=config['num_iter_l1'],
            cfg=config['cfg'],
            temperature=config['temperature'],
        )
    except Exception as e:
        print(f"  [!] Generation failed: {e}")
        return
    
    # Denormalize
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)
    
    # Save as grid image
    from torchvision.utils import make_grid, save_image
    grid = make_grid(images, nrow=4, padding=2)
    
    save_dir = os.path.join(config['output_dir'], 'samples')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'epoch{epoch:04d}.png')
    save_image(grid, save_path)
    print(f"  [✓] Saved samples: {save_path}")
    
    if log_writer:
        log_writer.add_image('samples', grid, epoch)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MSFT Cross-Image Training for FractalGen'
    )
    parser.add_argument('--phase', type=int, required=True, choices=[1, 2, 3],
                       help='Training phase (1=coarse only, 2=L1 frozen L0, 3=joint)')
    parser.add_argument('--data_root', type=str, default='./data/tiny-imagenet-200',
                       help='Path to ImageFolder dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--model_size', type=str, default=None,
                       choices=['tiny', '5090', 'medium'],
                       help='Model size (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda or cpu)')
    parser.add_argument('--pairing_mode', type=str, default=None,
                       choices=['same_class', 'same_image', 'random'],
                       help='Data pairing mode')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='DataLoader workers (overrides config)')
    parser.add_argument('--accum_steps', type=int, default=None,
                       help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Load config and apply CLI overrides
    config = get_default_config()
    for key in ['model_size', 'batch_size', 'epochs', 'lr', 'output_dir', 
                'device', 'pairing_mode', 'data_root', 'num_workers', 'accum_steps']:
        val = getattr(args, key, None)
        if val is not None:
            config[key] = val
    
    # Set device
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        config['device'] = 'cpu'
        config['use_amp'] = False
    
    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['output_dir'] = os.path.join(config['output_dir'], f'msft_phase{args.phase}_{timestamp}')
    config['log_dir'] = os.path.join(config['log_dir'], f'msft_phase{args.phase}_{timestamp}')
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Save config
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Device: {config['device']}")
    print(f"Output: {config['output_dir']}")
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Run training phase
    if args.phase == 1:
        model = train_phase1(args, config)
    elif args.phase == 2:
        model = train_phase2(args, config)
    elif args.phase == 3:
        model = train_phase3(args, config)
    
    print(f"\n{'='*60}")
    print(f"Phase {args.phase} training complete!")
    print(f"Checkpoints saved to: {config['output_dir']}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
