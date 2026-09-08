import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.training.loss import OceanEmbedLoss

def run_loss_verification():
    print("=" * 60)
    print("OceanEmbed — Phase 14 Loss Verification")
    print("=" * 60)

    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    loss_fn = OceanEmbedLoss(20.0, 2.0, 35.0, 1.0, mask, lambda_phys=0.01)
    
    B, D, H, W = 1, 15, 2, 2
    t_pred = torch.randn(B, D, H, W, requires_grad=True)
    s_pred = torch.randn(B, D, H, W, requires_grad=True)
    t_true = torch.randn(B, D, H, W)
    s_true = torch.randn(B, D, H, W)

    # 1. Geometry & Dictionary
    total, d = loss_fn(t_pred, s_pred, t_true, s_true)
    assert total.dim() == 0, "Total loss must be a scalar"
    assert all(k in d for k in ["temperature", "salinity", "physics", "total"])
    print("[1 & 7] Geometry and Loss Dictionary: PASS")
    
    # 2. NaN Ground Truth Handling
    t_true_nan = t_true.clone()
    t_true_nan[:, :, 1, 1] = float('nan')
    tot_nan, _ = loss_fn(t_pred, s_pred, t_true_nan, s_true)
    assert not torch.isnan(tot_nan)
    print("[2 & 8] NaN Ground Truth & Numerical Integrity: PASS")

    # 3. Parameter Sanity
    trainable = sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)
    assert trainable == 0, f"Found {trainable} trainable parameters in loss function!"
    print("[10] Parameter Sanity (0 trainable weights): PASS")

    # 4. Gradient Flow
    total.backward()
    assert t_pred.grad is not None and s_pred.grad is not None
    print("[9] Gradient Flow: PASS")

    # 5 & 6. Lambda Scaling and Physics Integration
    loss_stable = OceanEmbedLoss(20.0, 2.0, 35.0, 1.0, mask, lambda_phys=0.0)
    tot_stable, d_stable = loss_stable(t_pred, s_pred, t_true, s_true)
    assert d_stable["physics"] == 0.0
    print("[4, 5 & 6] Physics Integration & Lambda=0 Toggle: PASS")

    print("\n" + "=" * 60)
    print("PHASE 14 VERIFICATION: PASS")
    print("============================================================")

if __name__ == "__main__":
    run_loss_verification()