import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.config import TrainingConfig
from src.training.optimizer import build_optimizer
from src.models.oceanembed import OceanEmbedModel
from src.training.loss import OceanEmbedLoss

def verify_config_and_optimizer():
    print("=" * 60)
    print("OceanEmbed — Phase 15.2 & 15.3 Verification")
    print("=" * 60)

    # 1. Verify Configuration
    print("\n[1] Training Configuration...")
    config = TrainingConfig()
    
    assert config.effective_batch_size == 16, "Effective batch size calculation failed"
    assert config.scheduler_patience < config.early_stopping_patience, "Scheduler/Early Stopping collision detected!"
    
    print(f"  [✓] Configuration loaded successfully.")
    print(f"  [✓] Micro-batch: {config.micro_batch_size} | Accumulation: {config.accumulation_steps} | Effective: {config.effective_batch_size}")
    print(f"  [✓] Scheduler Patience ({config.scheduler_patience}) safely precedes Early Stopping ({config.early_stopping_patience}).")

    # 2. Verify Optimizer Parameter Isolation
    print("\n[2] Optimizer Construction...")
    model = OceanEmbedModel(in_channels=12)
    
    # Mock mask and loss
    mask = torch.ones(101, 241)
    loss_fn = OceanEmbedLoss(20.0, 2.0, 35.0, 1.0, mask, lambda_phys=0.01)
    
    # Build optimizer
    optimizer = build_optimizer(model, config)
    
    # Count parameters in optimizer vs model
    opt_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_params = sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)
    
    assert opt_params == model_params, "Optimizer is tracking incorrect number of parameters!"
    assert loss_params == 0, "Loss function contains trainable parameters!"
    
    print(f"  [✓] RAdam Optimizer initialized correctly.")
    print(f"  [✓] Optimizer tracking exact model parameter count: {opt_params:,}")
    print(f"  [✓] Parameter leak prevented (Loss module has 0 trainable weights).")

    print("\n" + "=" * 60)
    print("PHASE 15.2 & 15.3 VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    verify_config_and_optimizer()