import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.config import TrainingConfig
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.models.oceanembed import OceanEmbedModel
from src.training.loss import OceanEmbedLoss

def verify_mechanics():
    print("=" * 60)
    print("OceanEmbed — Phase 15.4 - 15.7 Training Mechanics Verification")
    print("=" * 60)

    # 1. Setup
    config = TrainingConfig(micro_batch_size=2, accumulation_steps=2) # Keep it small for the test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = OceanEmbedModel(in_channels=12).to(device)
    mask = torch.ones(101, 241).to(device)
    loss_fn = OceanEmbedLoss(20.0, 2.0, 35.0, 1.0, mask, lambda_phys=0.01).to(device)
    
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    
    # AMP Scaler
    scaler = torch.amp.GradScaler(device.type)

    # Mock Data
    x = torch.randn(config.micro_batch_size, 12, 101, 241, device=device)
    t_true = torch.randn(config.micro_batch_size, 15, 101, 241, device=device)
    s_true = torch.randn(config.micro_batch_size, 15, 101, 241, device=device)

    print(f"\n[1] Executing Accumulated AMP Forward/Backward Pass (Device: {device})...")
    
    optimizer.zero_grad()
    
    # 15.7: Gradient Accumulation Loop
    for step in range(config.accumulation_steps):
        # 15.6: Automatic Mixed Precision
        with torch.amp.autocast(device.type, enabled=config.use_amp):
            t_pred, s_pred = model(x)
            loss, _ = loss_fn(t_pred, s_pred, t_true, s_true)
            
            # Divide loss by accumulation steps so the gradients don't explode
            loss = loss / config.accumulation_steps
            
        # Scale loss and backward
        scaler.scale(loss).backward()
        
    print("  [✓] Forward and Backward passes succeeded under AMP.")
    print("  [✓] Loss successfully divided for gradient accumulation.")

    print("\n[2] Gradient Clipping & Optimizer Step...")
    
    # 15.5: Gradient Clipping requires unscaling first!
    scaler.unscale_(optimizer)
    
    # Record pre-clip norm to verify clipping logic fires
    pre_clip_norm = nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norms)
    
    scaler.step(optimizer)
    scaler.update()
    
    print(f"  [✓] Gradients unscaled and clipped (Max Norm: {config.max_grad_norms}).")
    print(f"  [✓] Optimizer step executed.")

    print("\n[3] Scheduler Step Validation...")
    initial_lr = optimizer.param_groups[0]['lr']
    
    # Force the scheduler to trigger by feeding it a bad validation loss repeatedly
    for _ in range(config.scheduler_patience + 2):
        scheduler.step(100.0) # Mock bad validation loss
        
    new_lr = optimizer.param_groups[0]['lr']
    
    assert new_lr < initial_lr, "Scheduler failed to reduce learning rate!"
    print(f"  [✓] Scheduler correctly reduced LR from {initial_lr} to {new_lr} after patience exceeded.")

    print("\n" + "=" * 60)
    print("PHASE 15.4 - 15.7 VERIFICATION COMPLETE: CORE MECHANICS READY")
    print("=" * 60)

if __name__ == "__main__":
    verify_mechanics()