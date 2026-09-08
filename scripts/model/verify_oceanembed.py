import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.oceanembed import OceanEmbedModel
from src.models.encoder import Encoder
from src.models.bottleneck import Bottleneck
from src.models.decoder import Decoder
from src.models.cbam import CBAM

def verify_full_model():
    print("=" * 60)
    print("OceanEmbed — Phase 15.1 Full Model Verification")
    print("=" * 60)

    model = OceanEmbedModel(in_channels=12)
    B, C, H, W = 2, 12, 101, 241
    x = torch.randn(B, C, H, W, requires_grad=True)

    # ---------------------------------------------------------
    # 1. Structural Assertions
    # ---------------------------------------------------------
    print("\n[1] Structural Integrity...")
    
    encoders = [m for m in model.children() if isinstance(m, Encoder)]
    bottlenecks = [m for m in model.children() if isinstance(m, Bottleneck)]
    decoders = [m for m in model.children() if isinstance(m, Decoder)]
    
    assert len(encoders) == 1, f"Expected 1 Encoder, found {len(encoders)}"
    assert len(bottlenecks) == 1, f"Expected 1 Bottleneck, found {len(bottlenecks)}"
    assert len(decoders) == 2, f"Expected 2 Decoders, found {len(decoders)}"
    print("  [✓] Exactly one Encoder and one Bottleneck identified.")
    print("  [✓] Exactly two Decoder instances (Dual-Head) identified.")

    cbam_blocks = [m for m in model.modules() if isinstance(m, CBAM)]
    assert len(cbam_blocks) == 5, f"Expected 5 CBAM blocks (4 in Encoder, 1 in Bottleneck), found {len(cbam_blocks)}"
    print("  [✓] 5 CBAM attention blocks correctly placed.")

    # ---------------------------------------------------------
    # 2. Forward Pass Geometry
    # ---------------------------------------------------------
    print("\n[2] Forward Pass & Output Geometry...")
    temp_pred, sal_pred = model(x)
    
    expected_shape = [B, 15, 101, 241]
    assert list(temp_pred.shape) == expected_shape, f"Temp shape failed: {temp_pred.shape}"
    assert list(sal_pred.shape) == expected_shape, f"Salinity shape failed: {sal_pred.shape}"
    print(f"  [✓] Input Tensor:  {list(x.shape)}")
    print(f"  [✓] Temp Output:   {list(temp_pred.shape)}")
    print(f"  [✓] Sal Output:    {list(sal_pred.shape)}")

    # ---------------------------------------------------------
    # 3. Unified Backpropagation
    # ---------------------------------------------------------
    print("\n[3] Unified Backpropagation Graph...")
    # Dummy loss to check if gradients flow through the entire graph
    dummy_loss = temp_pred.mean() + sal_pred.mean()
    dummy_loss.backward()
    
    assert x.grad is not None, "Gradients failed to reach input tensor!"
    assert x.grad.sum().item() != 0.0, "Gradients at input are zero!"
    
    # Check if a parameter from the encoder received gradients
    encoder_param = next(model.encoder.parameters())
    assert encoder_param.grad is not None, "Encoder did not receive gradients."
    
    print("  [✓] Gradients successfully propagated from both decoders, through the bottleneck, and down to the encoder.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Total Trainable Parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("PHASE 15.1 VERIFICATION COMPLETE: MODEL GRAPH READY")
    print("=" * 60)

if __name__ == "__main__":
    verify_full_model()