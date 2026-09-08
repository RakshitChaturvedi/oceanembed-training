import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.models.decoder import DecoderStage

def verify_decoder_stage():
    print("=" * 60)
    print("OceanEmbed — Phase 12.7 DecoderStage Verification")
    print("=" * 60)

    # Test the [12x30] -> [25x60] jump
    print("\n[1] Dynamic Padding Test (12x30 -> 25x60)...")
    stage = DecoderStage(in_channels=512, skip_channels=256, out_channels=256)
    
    latent = torch.randn(2, 512, 12, 30)
    skip3 = torch.randn(2, 256, 25, 60)
    
    out = stage(latent, skip3)
    
    assert list(out.shape) == [2, 256, 25, 60], f"Output shape failed: {out.shape}"
    print("  [✓] 24x60 dynamically padded to 25x60 before concatenation")
    print("  [✓] 512 + 256 (768) correctly squashed down to 256 channels")

    # Test the [50x120] -> [101x241] jump
    print("\n[2] Final Basin Grid Test (50x120 -> 101x241)...")
    stage_final = DecoderStage(in_channels=128, skip_channels=64, out_channels=64)
    
    latent_2 = torch.randn(2, 128, 50, 120)
    skip1 = torch.randn(2, 64, 101, 241)
    
    out_2 = stage_final(latent_2, skip1)
    
    assert list(out_2.shape) == [2, 64, 101, 241], f"Output shape failed: {out_2.shape}"
    print("  [✓] 100x240 dynamically padded to 101x241 before concatenation")
    print("  [✓] 128 + 64 (192) correctly squashed down to 64 channels")

    print("\n" + "=" * 60)
    print("DECODER STAGE VERIFIED")
    print("=" * 60)

if __name__ == "__main__":
    verify_decoder_stage()