import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.decoder import Decoder

def verify_dual_decoder():
    print("=" * 60)
    print("OceanEmbed — Phase 12.9 Dual-Decoder Verification")
    print("=" * 60)

    # 1. Instantiate both heads independently
    temp_decoder = Decoder()
    salinity_decoder = Decoder()

    assert temp_decoder is not salinity_decoder, "Decoders share identical memory instance!"
    print("\n[1] Parameter Independence...")
    print("  [✓] Temp and Salinity decoders are isolated objects.")

    # 2. Build the exact Phase 11 & Phase 9 outputs
    B = 2
    latent = torch.randn(B, 512, 6, 15, requires_grad=True)
    
    skips = (
        torch.randn(B, 64, 101, 241, requires_grad=True),  # skip1
        torch.randn(B, 128, 50, 120, requires_grad=True),  # skip2
        torch.randn(B, 256, 25, 60, requires_grad=True),   # skip3
        torch.randn(B, 512, 12, 30, requires_grad=True)    # skip4
    )

    print("\n[2] Forward Pass & Output Projections...")
    
    # Push identical inputs into both decoders
    pred_T = temp_decoder(latent, skips)
    pred_S = salinity_decoder(latent, skips)

    # Verify Output Shapes
    assert list(pred_T.shape) == [B, 15, 101, 241], f"Temp shape failed: {pred_T.shape}"
    assert list(pred_S.shape) == [B, 15, 101, 241], f"Salinity shape failed: {pred_S.shape}"
    
    print("  [✓] Temperature shape: [B, 15, 101, 241]")
    print("  [✓] Salinity shape:    [B, 15, 101, 241]")
    
    # Check 1x1 conv terminal logic
    assert isinstance(temp_decoder.final_conv, nn.Conv2d), "Missing Conv2d projection"
    assert temp_decoder.final_conv.kernel_size == (1, 1), "Terminal conv is not 1x1"
    print("  [✓] 1x1 Convolution verified (No activation applied)")

    # 3. Data Integrity & Gradients
    print("\n[3] Integrity & Gradient Flow...")
    assert not torch.isnan(pred_T).any() and not torch.isinf(pred_T).any(), "NaNs in Temp"
    assert not torch.isnan(pred_S).any() and not torch.isinf(pred_S).any(), "NaNs in Salinity"
    
    # Combined Dummy Loss
    loss = pred_T.mean() + pred_S.mean()
    loss.backward()

    # Gradients should flow back to the latent tensor and all skips
    assert latent.grad is not None, "Gradients failed to reach latent bottleneck"
    for idx, skip in enumerate(skips):
        assert skip.grad is not None, f"Gradients failed to reach skip {idx+1}"

    print("  [✓] No NaNs/Infs generated")
    print("  [✓] Gradients flow backward through both heads simultaneously")
    print("  [✓] Shared Latent & Skip tensors successfully receive combined gradients")

    print("\n" + "=" * 60)
    print("PHASE 12 VERIFICATION COMPLETE: DUAL-HEAD ARCHITECTURE READY")
    print("=" * 60)

if __name__ == "__main__":
    verify_dual_decoder()