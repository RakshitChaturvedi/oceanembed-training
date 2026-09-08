import sys
import os
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.evaluation.tabular_dataset import extract_tabular_data, reconstruct_from_tabular

def verify_data():
    print("=" * 60)
    print("Phase 16.5: Tabular Data Verification (Gate 2)")
    print("=" * 60)

    B, C, D, H, W = 2, 12, 15, 2, 2
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]]) # 3 ocean, 1 land
    lat = torch.tensor([[10.0, 10.0], [9.0, 9.0]])
    lon = torch.tensor([[60.0, 61.0], [60.0, 61.0]])
    
    x = torch.ones(B, C, H, W)
    t = torch.ones(B, D, H, W) * 5.0
    s = torch.ones(B, D, H, W) * 35.0

    print("[1] Testing Tabular Extraction...")
    X_tab, Y_tab = extract_tabular_data(x, t, s, mask, lat, lon)
    
    # 2 batches * 3 ocean pixels = 6 valid rows
    assert X_tab.shape == (6, 14), f"X shape wrong: {X_tab.shape}"
    assert Y_tab.shape == (6, 30), f"Y shape wrong: {Y_tab.shape}"
    print("  [✓] Shapes correct: [N, 14] and [N, 30].")

    print("\n[2] Testing Geographic Features...")
    assert X_tab[0, 12].item() == 10.0 and X_tab[0, 13].item() == 60.0, "Lat/Lon mapping failed!"
    print("  [✓] Latitude and Longitude appended correctly.")

    print("\n[3] Testing Reconstruction & Leakage...")
    t_rec, s_rec = reconstruct_from_tabular(Y_tab, mask, B)
    
    assert t_rec.shape == (B, D, H, W), "Reconstruction shape failed!"
    assert t_rec[0, 0, 1, 1].item() == 0.0, "Land pixel leaked!"
    assert t_rec[0, 0, 0, 0].item() == 5.0, "Ocean pixel corrupted!"
    print("  [✓] Spatial grids reconstructed perfectly. Land remains masked.")
    print("=" * 60)

if __name__ == "__main__":
    verify_data()