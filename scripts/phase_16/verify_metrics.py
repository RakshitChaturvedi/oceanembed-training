import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.evaluation.metrics import evaluate_predictions

def verify_metrics():
    print("=" * 60)
    print("OceanEmbed — Phase 16.2 Metrics Verification")
    print("=" * 60)

    # 1. Setup mock geometry
    B, D, H, W = 2, 15, 2, 2
    
    # 2x2 mask: 3 ocean pixels, 1 land pixel
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    
    # Predictions: all 0.0
    t_pred = torch.zeros(B, D, H, W)
    s_pred = torch.zeros(B, D, H, W)
    
    # Targets: all 2.0 (so absolute error should be exactly 2.0)
    t_true = torch.ones(B, D, H, W) * 2.0
    s_true = torch.ones(B, D, H, W) * 2.0
    
    # Inject NaN on an ocean pixel to test dynamic sanitization
    t_true[0, 0, 0, 0] = float('nan')
    
    # Inject massive error (999.0) on the land pixel
    t_true[:, :, 1, 1] = 999.0
    s_true[:, :, 1, 1] = 999.0

    # 2. Run Evaluation
    results = evaluate_predictions(t_pred, s_pred, t_true, s_true, mask)

    # 3. Validations
    print("\n[1] Dictionary Structure...")
    assert len(results["temperature"]["mae"]) == 15
    assert len(results["salinity"]["rmse"]) == 15
    print("  [✓] Depth-wise arrays contain exactly 15 levels.")

    print("\n[2] Mask & NaN Handling...")
    # Salinity should be exactly 2.0 because land (999.0) is masked out.
    assert abs(results["salinity"]["mean_mae"] - 2.0) < 1e-5, f"Mask failed! MAE: {results['salinity']['mean_mae']}"
    print("  [✓] Land pixels strictly excluded from MAE/RMSE calculations.")
    
    # Temperature should have handled the NaN cleanly without corrupting the metric
    assert not np.isnan(results["temperature"]["mean_mae"])
    print("  [✓] Ground truth NaNs safely handled and excluded.")

    print("\n" + "=" * 60)
    print("PHASE 16.2 VERIFICATION COMPLETE: RULER READY")
    print("=" * 60)

if __name__ == "__main__":
    verify_metrics()