import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.physics.stratification import DensityMonotonicityLoss

def verify_physics():
    print("=" * 60)
    print("OceanEmbed — Phase 13 Physics-Informed Constraint")
    print("=" * 60)

    # 1. Mock Phase 5/6 Artifacts
    # Using easy math for verification: mean=20, std=2
    t_mean, t_std = 20.0, 2.0
    s_mean, s_std = 35.0, 1.0
    
    # 2x2 mock mask: 3 ocean pixels, 1 land pixel
    mock_mask = torch.tensor([
        [1.0, 1.0],
        [1.0, 0.0]
    ])

    physics_module = DensityMonotonicityLoss(t_mean, t_std, s_mean, s_std, mock_mask)
    
    # Define shapes: [Batch=1, Depth=15, H=2, W=2]
    B, D, H, W = 1, 15, 2, 2

    # ---------------------------------------------------------
    # Case A: Stable Stratification
    # ---------------------------------------------------------
    print("\n[1] Case A: Stable Profile...")
    # T goes down with depth (normalized: 1 -> -1), S goes up
    # This guarantees density increases with depth
    t_stable = torch.linspace(1.0, -1.0, D).view(1, D, 1, 1).expand(B, D, H, W)
    s_stable = torch.linspace(-1.0, 1.0, D).view(1, D, 1, 1).expand(B, D, H, W)
    
    loss_stable = physics_module(t_stable, s_stable)
    assert loss_stable.item() == 0.0, f"Stable profile yielded non-zero loss: {loss_stable.item()}"
    print("  [✓] Stable profile yielded exactly 0.0 penalty.")

    # ---------------------------------------------------------
    # Case B: Completely Unstable
    # ---------------------------------------------------------
    print("\n[2] Case B: Unstable Profile (Inversion)...")
    # T goes UP with depth, S goes DOWN (physically impossible)
    t_unstable = torch.linspace(-1.0, 1.0, D).view(1, D, 1, 1).expand(B, D, H, W).clone().requires_grad_(True)
    s_unstable = torch.linspace(1.0, -1.0, D).view(1, D, 1, 1).expand(B, D, H, W).clone().requires_grad_(True)
    
    loss_unstable = physics_module(t_unstable, s_unstable)
    assert loss_unstable.item() > 0.0, "Unstable profile failed to trigger penalty!"
    print(f"  [✓] Unstable profile caught. Penalty: {loss_unstable.item():.4f}")

    # ---------------------------------------------------------
    # Gradient Flow Verification
    # ---------------------------------------------------------
    print("\n[3] Differentiability & Autograd Flow...")
    loss_unstable.backward()
    
    assert t_unstable.grad is not None, "Temperature gradient lost!"
    assert s_unstable.grad is not None, "Salinity gradient lost!"
    print("  [✓] Gradients propagated successfully through mask, ReLU, diff, EOS, and Denorm.")
    
    # ---------------------------------------------------------
    # Case C: Land Mask Validation
    # ---------------------------------------------------------
    print("\n[4] Land Masking Validation...")
    # Check the gradient on the land pixel (bottom right: [1, 1])
    # Because the mask is 0 there, it should have NO gradient (exactly 0.0)
    land_grad_sum = t_unstable.grad[0, :, 1, 1].abs().sum().item()
    ocean_grad_sum = t_unstable.grad[0, :, 0, 0].abs().sum().item()
    
    assert land_grad_sum == 0.0, "Gradients are flowing through land pixels!"
    assert ocean_grad_sum != 0.0, "Ocean pixels missing gradients!"
    print("  [✓] Land pixels correctly isolated. No gradients computed over land.")

    print("\n" + "=" * 60)
    print("PHASE 13 VERIFICATION COMPLETE: PHYSICS MODULE READY")
    print("=" * 60)

if __name__ == "__main__":
    verify_physics()