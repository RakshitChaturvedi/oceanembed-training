import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.oceanembed import OceanEmbedModel
from src.models.cbam import CBAM

def verify_plain_unet():
    print("=" * 60)
    print("OceanEmbed — Phase 16.5 Plain U-Net Verification")
    print("=" * 60)

    # 1. Instantiate WITHOUT CBAM
    model = OceanEmbedModel(in_channels=12, use_cbam=False)
    B, C, H, W = 2, 12, 101, 241
    x = torch.randn(B, C, H, W, requires_grad=True)

    print("\n[1] Structural Integrity (Plain U-Net)...")
    cbam_blocks = [m for m in model.modules() if isinstance(m, CBAM)]
    assert len(cbam_blocks) == 0, f"Ablation failed! Found {len(cbam_blocks)} CBAM blocks."
    print("  [✓] 0 CBAM attention blocks found (Attention cleanly ablated).")

    print("\n[2] Forward Pass & Output Geometry...")
    temp_pred, sal_pred = model(x)
    
    expected_shape = [B, 15, 101, 241]
    assert list(temp_pred.shape) == expected_shape, "Temp shape failed"
    assert list(sal_pred.shape) == expected_shape, "Salinity shape failed"
    print(f"  [✓] Outputs perfectly match expected Phase 8 shapes.")

    print("\n[3] Unified Backpropagation Graph...")
    dummy_loss = temp_pred.mean() + sal_pred.mean()
    dummy_loss.backward()
    
    assert x.grad is not None and x.grad.sum().item() != 0.0, "Gradients failed to reach input!"
    print("  [✓] Gradients successfully propagated without attention gates.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total Trainable Parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("PHASE 16.5 VERIFICATION COMPLETE: PLAIN U-NET READY")
    print("=" * 60)

if __name__ == "__main__":
    verify_plain_unet()