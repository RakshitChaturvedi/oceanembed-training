import sys
import os
import torch
import numpy as np
import joblib
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.evaluation.tabular_dataset import extract_tabular_data
from src.physics.eos import linear_eos

def calculate_metrics(preds, truths):
    """Calculates RMSE and MAE, safely ignoring NaNs in the truth data."""
    valid_mask = ~np.isnan(truths) & (truths != 0.0)
    preds_valid = preds[valid_mask]
    truths_valid = truths[valid_mask]
    
    if len(truths_valid) == 0: return 0.0, 0.0
        
    mse = np.mean((preds_valid - truths_valid) ** 2)
    mae = np.mean(np.abs(preds_valid - truths_valid))
    return np.sqrt(mse), mae

def calculate_inversion_statistics(t_preds, s_preds, valid_mask_3d, tolerance=0.0):
    if len(t_preds) == 0:
        return {"layer_rate": 0.0, "column_rate": 0.0, "mean_magnitude": 0.0, "median_magnitude": 0.0, "p95_magnitude": 0.0, "max_magnitude": 0.0, "aggregate_severity": 0.0}

    t = torch.as_tensor(t_preds, dtype=torch.float32)
    s = torch.as_tensor(s_preds, dtype=torch.float32)
    mask = torch.as_tensor(valid_mask_3d, dtype=torch.bool)

    density = linear_eos(t, s)
    delta_rho = torch.diff(density, dim=1)
    
    valid_transitions = mask[:, 1:] & mask[:, :-1]
    inversion_mask = (delta_rho < -tolerance) & valid_transitions

    total_pairs = valid_transitions.sum().item()
    inversion_count = inversion_mask.sum().item()
    layer_rate = (inversion_count / total_pairs * 100.0 if total_pairs > 0 else 0.0)

    valid_columns_mask = valid_transitions.any(dim=1)
    total_valid_columns = valid_columns_mask.sum().item()
    inverted_columns = inversion_mask.any(dim=1) & valid_columns_mask
    column_rate = (inverted_columns.sum().item() / total_valid_columns * 100.0 if total_valid_columns > 0 else 0.0)

    inversion_magnitudes = torch.abs(delta_rho[inversion_mask])
    if inversion_magnitudes.numel() == 0:
        mean_magnitude = median_magnitude = p95_magnitude = max_magnitude = 0.0
    else:
        mean_magnitude = inversion_magnitudes.mean().item()
        median_magnitude = inversion_magnitudes.median().item()
        p95_magnitude = torch.quantile(inversion_magnitudes, 0.95).item()
        max_magnitude = inversion_magnitudes.max().item()

    aggregate_severity = (inversion_magnitudes.sum().item() / total_pairs if total_pairs > 0 else 0.0)

    return {
        "layer_rate": layer_rate, "column_rate": column_rate, "mean_magnitude": mean_magnitude,
        "median_magnitude": median_magnitude, "p95_magnitude": p95_magnitude, "max_magnitude": max_magnitude,
        "aggregate_severity": aggregate_severity,
    }

def main(fold=1):
    print("=" * 80)
    print(f"Phase 16.10: DRY RUN - Tabular Baselines Only (Fold {fold})")
    print("=" * 80)

    print("[1] Loading Artifacts & DataLoaders...")
    mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
    ocean_mask = torch.from_numpy(mask_ds.values).float()
    lat_coords = mask_ds.coords[mask_ds.dims[0]].values
    lon_coords = mask_ds.coords[mask_ds.dims[1]].values
    lon_2d, lat_2d = np.meshgrid(lon_coords, lat_coords)
    lat_grid = torch.from_numpy(lat_2d).float()
    lon_grid = torch.from_numpy(lon_2d).float()

    _, _, test_loader = get_dataloaders(fold=fold, batch_size=16)

    print("[2] Loading MLR and XGBoost Models...")
    models = {
        "MLR": joblib.load(f"experiments/phase16/mlr/fold_{fold}/model.joblib"),
        "XGB": joblib.load(f"experiments/phase16/xgboost/fold_{fold}/model.joblib")
    }

    results = {name: {"T_p": [], "S_p": [], "T_t": [], "S_t": []} for name in models.keys()}

    print("[3] Running Inference on Test Set...")
    with torch.no_grad():
        for batch_idx, (x, t_true, s_true) in enumerate(test_loader):
            x_tab, y_tab_true = extract_tabular_data(x, t_true, s_true, ocean_mask, lat_grid, lon_grid)
            x_tab_np, y_tab_np = np.nan_to_num(x_tab.numpy(), nan=0.0), y_tab_true.numpy()
            t_true_tab, s_true_tab = y_tab_np[:, :15], y_tab_np[:, 15:]
            
            for name in ["MLR", "XGB"]:
                pred = models[name].predict(x_tab_np)
                results[name]["T_p"].append(pred[:, :15])
                results[name]["S_p"].append(pred[:, 15:])
                results[name]["T_t"].append(t_true_tab)
                results[name]["S_t"].append(s_true_tab)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

    print("\n[4] Dry Run Leaderboard")
    print(f"{'Model':<15} | {'T RMSE':<8} | {'T MAE':<8} | {'S RMSE':<8} | {'S MAE':<8} | {'Inv Layer %':<11} | {'Inv Column %':<12} | {'Mean Inv':<9} | {'P95 Inv':<9} | {'Agg Severity':<13}")
    print("-" * 125)
    
    ordered_names = ["MLR", "XGB"]
    ground_truth_t = np.vstack(results["MLR"]["T_t"])
    ground_truth_s = np.vstack(results["MLR"]["S_t"])
    gt_valid_mask = ~np.isnan(ground_truth_t)

    true_inv_stats = calculate_inversion_statistics(ground_truth_t, ground_truth_s, valid_mask_3d=gt_valid_mask, tolerance=0.0)
    
    for name in ordered_names:
        t_p = np.vstack(results[name]["T_p"])
        t_t = np.vstack(results[name]["T_t"])
        s_p = np.vstack(results[name]["S_p"])
        s_t = np.vstack(results[name]["S_t"])
        
        valid_mask = ~np.isnan(t_t)

        t_rmse, t_mae = calculate_metrics(t_p[valid_mask], t_t[valid_mask])
        s_rmse, s_mae = calculate_metrics(s_p[valid_mask], s_t[valid_mask])
        
        inv_stats = calculate_inversion_statistics(t_p, s_p, valid_mask_3d=valid_mask, tolerance=0.0)

        print(
            f"{name:<15} | {t_rmse:<8.4f} | {t_mae:<8.4f} | {s_rmse:<8.4f} | {s_mae:<8.4f} | "
            f"{inv_stats['layer_rate']:>7.2f}% | {inv_stats['column_rate']:>7.2f}% | "
            f"{inv_stats['mean_magnitude']:>8.4f} | {inv_stats['p95_magnitude']:>8.4f} | {inv_stats['aggregate_severity']:>9.5f}"
        )

    print("-" * 125)
    print(
        f"{'GROUND TRUTH':<15} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | {'N/A':<8} | "
        f"{true_inv_stats['layer_rate']:>7.2f}% | {true_inv_stats['column_rate']:>7.2f}% | "
        f"{true_inv_stats['mean_magnitude']:>8.4f} | {true_inv_stats['p95_magnitude']:>8.4f} | {true_inv_stats['aggregate_severity']:>9.5f}"
    )
    print("=" * 80)

if __name__ == "__main__":
    main(fold=1)