import sys
import os
import torch
import torch.nn as nn

# Ensure src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.encoder import Encoder
from src.models.cbam import CBAM
from src.models.bottleneck import Bottleneck

def verify_bottleneck():
    print("=" * 60)
    print("OceanEmbed — Phase 11 Bottleneck Verification")
    print("=" * 60)

    B, C, H, W = 2, 512, 6, 15
    x = torch.randn(B, C, H, W, requires_grad=True)

    # ---------------------------------------------------------
    # 1. Standalone Architecture & Constraints
    # ---------------------------------------------------------
    print("\n[1] Standalone Architecture Assertions...")
    bottleneck = Bottleneck(channels=C, attention_block=CBAM)

    # Assert no MaxPool2d exists
    pool_layers = [m for m in bottleneck.modules() if isinstance(m, nn.MaxPool2d)]
    assert len(pool_layers) == 0, f"Forbidden MaxPool2d found in Bottleneck: {pool_layers}"
    
    # Assert ELU inplace=True
    elu_layers = [m for m in bottleneck.modules() if isinstance(m, nn.ELU)]
    assert len(elu_layers) == 2, f"Expected 2 ELU layers, found {len(elu_layers)}"
    assert all(m.inplace for m in elu_layers), "Not all ELU layers have inplace=True"

    # Assert Conv2d geometry constraints
    conv_layers = [m for m in bottleneck.modules() if isinstance(m, nn.Conv2d) and m.kernel_size == (3,3)]
    for conv in conv_layers:
        assert conv.padding == (1, 1), "Conv2d missing padding=1"
        assert conv.stride == (1, 1), "Conv2d stride must be 1"
        assert conv.in_channels == 512 and conv.out_channels == 512, "Hidden channel explosion detected"

    print("  [✓] No MaxPool2d explicitly verified")
    print("  [✓] ELU inplace=True explicitly verified")
    print("  [✓] 512 -> 512 channel preservation verified (No 1024 spike)")

    # ---------------------------------------------------------
    # 2. Geometry, Sanity, and Gradients
    # ---------------------------------------------------------
    print("\n[2] Execution & Geometry...")
    out = bottleneck(x)
    
    assert list(out.shape) == [B, C, H, W], f"Output shape mismatch: {out.shape}"
    assert not torch.isnan(out).any() and not torch.isinf(out).any(), "NaN/Inf detected"
    
    loss = out.mean()
    loss.backward()
    
    assert x.grad is not None and x.grad.sum().item() != 0.0, "Gradient flow broken"
    print(f"  [✓] Geometry preserved: {list(out.shape)}")
    print("  [✓] Data integrity & Gradient flow verified")

    total_params = sum(p.numel() for p in bottleneck.parameters())
    trainable_params = sum(p.numel() for p in bottleneck.parameters() if p.requires_grad)
    print(f"  [✓] Total Parameters: {total_params:,}")

    # ---------------------------------------------------------
    # 3. Integration with Phase 9 Encoder
    # ---------------------------------------------------------
    print("\n[3] Phase 9 -> Phase 11 Integration...")
    encoder = Encoder(in_channels=12, attention_block=CBAM)
    
    # Phase 8 Input Tensor
    surface_tensor = torch.randn(2, 12, 101, 241)
    
    # Run Encoder
    enc_latent, skips = encoder(surface_tensor)
    
    # Run Bottleneck
    final_latent = bottleneck(enc_latent)

    # Verify skips are untouched and latent is processed correctly
    assert list(skips[0].shape) == [2, 64, 101, 241], "Skip 1 modified"
    assert list(skips[-1].shape) == [2, 512, 12, 30], "Skip 4 modified"
    assert list(final_latent.shape) == [2, 512, 6, 15], "Final latent shape mismatch"

    print("  [✓] Phase 9 Latent output accepted by Phase 11 Bottleneck")
    print("  [✓] Skip connections bypassed bottleneck successfully")
    print("  [✓] Final representation locked at [B, 512, 6, 15]")

    print("\n" + "=" * 60)
    print("PHASE 11 VERIFICATION COMPLETE: READY FOR PHASE 12 DECODERS")
    print("=" * 60)

if __name__ == "__main__":
    verify_bottleneck()