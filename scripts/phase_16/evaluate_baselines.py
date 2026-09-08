import sys
import os
import torch
import numpy as np
import joblib
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.models.oceanembed import OceanEmbedModel
from src.evaluation.tabular_dataset import extract_tabular_data
from src.physics.eos import linear_eos  # Importing your EOS!

def calculate_metrics(preds, truths):
    """Calculates RMSE, MAE, R^2, and Bias, safely ignoring NaNs in the truth data."""
    valid_mask = ~np.isnan(truths) & (truths != 0.0)
    preds_valid = preds[valid_mask]
    truths_valid = truths[valid_mask]
    
    if len(truths_valid) == 0: return 0.0, 0.0, 0.0, 0.0
        
    # Standard Errors
    mse = np.mean((preds_valid - truths_valid) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(preds_valid - truths_valid))
    
    # Mean Bias (Systematic over/under prediction)
    bias = np.mean(preds_valid - truths_valid)
    
    # R-squared (Variance Explained)
    ss_res = np.sum((truths_valid - preds_valid) ** 2)
    ss_tot = np.sum((truths_valid - np.mean(truths_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return rmse, mae, r2, bias

def calculate_inversion_statistics(
    t_preds,
    s_preds,
    valid_mask_3d,
    tolerance=0.0
):
    """
    Calculate vertical density inversion statistics strictly over valid ocean cells.

    Parameters
    ----------
    t_preds : np.ndarray
        Temperature predictions, shape [N, 15].
    s_preds : np.ndarray
        Salinity predictions, shape [N, 15].
    valid_mask_3d: np.ndarray
        Boolean mask of valid ocean cells, shape [N, 15].
    tolerance : float
        Minimum density decrease required to count as an inversion.
    """

    if len(t_preds) == 0:
        return {
            "layer_rate": 0.0,
            "column_rate": 0.0,
            "mean_magnitude": 0.0,
            "median_magnitude": 0.0,
            "p95_magnitude": 0.0,
            "max_magnitude": 0.0,
            "aggregate_severity": 0.0,
        }

    t = torch.as_tensor(t_preds, dtype=torch.float32)
    s = torch.as_tensor(s_preds, dtype=torch.float32)
    mask = torch.as_tensor(valid_mask_3d, dtype=torch.bool)

    # [N, 15]
    density = linear_eos(t, s)

    # [N, 14]
    delta_rho = torch.diff(density, dim=1)
    
    # ---------------------------------------------------------
    # Apply Valid Transition Masking
    # ---------------------------------------------------------
    # Transition is valid ONLY if both depth z and z+1 are valid ocean
    valid_transitions = mask[:, 1:] & mask[:, :-1]  # [N, 14]
    
    # Inversions happen when delta_rho < -tolerance AND it's a valid transition
    inversion_mask = (delta_rho < -tolerance) & valid_transitions

    # ---------------------------------------------------------
    # 1. Layer inversion rate
    # ---------------------------------------------------------
    total_pairs = valid_transitions.sum().item()
    inversion_count = inversion_mask.sum().item()

    layer_rate = (
        inversion_count / total_pairs * 100.0
        if total_pairs > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # 2. Column inversion rate
    # ---------------------------------------------------------
    # A column is only considered if it has at least one valid transition
    valid_columns_mask = valid_transitions.any(dim=1)
    total_valid_columns = valid_columns_mask.sum().item()
    
    inverted_columns = inversion_mask.any(dim=1) & valid_columns_mask

    column_rate = (
        inverted_columns.sum().item() / total_valid_columns * 100.0
        if total_valid_columns > 0
        else 0.0
    )

    # ---------------------------------------------------------
    # 3. Inversion magnitudes
    # ---------------------------------------------------------
    # Only pull magnitudes for ACTUAL valid inversions
    inversion_magnitudes = torch.abs(delta_rho[inversion_mask])

    if inversion_magnitudes.numel() == 0:
        mean_magnitude = 0.0
        median_magnitude = 0.0
        p95_magnitude = 0.0
        max_magnitude = 0.0
    else:
        mean_magnitude = inversion_magnitudes.mean().item()
        median_magnitude = inversion_magnitudes.median().item()
        p95_magnitude = torch.quantile(inversion_magnitudes, 0.95).item()
        max_magnitude = inversion_magnitudes.max().item()

    # ---------------------------------------------------------
    # 4. Aggregate inversion severity
    # ---------------------------------------------------------
    aggregate_severity = (
        inversion_magnitudes.sum().item() / total_pairs
        if total_pairs > 0
        else 0.0
    )

    return {
        "layer_rate": layer_rate,
        "column_rate": column_rate,
        "mean_magnitude": mean_magnitude,
        "median_magnitude": median_magnitude,
        "p95_magnitude": p95_magnitude,
        "max_magnitude": max_magnitude,
        "aggregate_severity": aggregate_severity,
    }

def load_unet(experiment_name, use_cbam, fold, device):
    """Helper to load U-Net variants."""
    model = OceanEmbedModel(in_channels=12, use_cbam=use_cbam).to(device)
    checkpoint = torch.load(f"experiments/phase16/{experiment_name}/fold_{fold}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def main(fold=1):
    print("=" * 80)
    print(f"Phase 16.10: Final 10-Model Ablation Evaluation (Fold {fold})")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Artifacts
    print("[1] Loading Artifacts & DataLoaders...")
    mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
    ocean_mask = torch.from_numpy(mask_ds.values).float()
    lat_coords = mask_ds.coords[mask_ds.dims[0]].values
    lon_coords = mask_ds.coords[mask_ds.dims[1]].values
    lon_2d, lat_2d = np.meshgrid(lon_coords, lat_coords)
    lat_grid = torch.from_numpy(lat_2d).float()
    lon_grid = torch.from_numpy(lon_2d).float()
    mask_bool = (ocean_mask == 1).bool()

    _, _, test_loader = get_dataloaders(fold=fold, batch_size=16)

    # 2. Load Models
    print("[2] Loading All 10 Trained Models...")
    
    # Define all spatial models to iterate through cleanly
    spatial_models = {
        "UNet": ("plain_unet", False),
        "UNet+Phys": ("plain_unet_physics", False),
        "CBAM": ("cbam_unet", True),
        "OceanEmbed": ("oceanembed", True),
        "OE_L0.1": ("oceanembed_L0.1", True),
        "OE_L1.0": ("oceanembed_L1.0", True),
        "OE_L10.0": ("oceanembed_L10.0", True),
        "OE_L100.0": ("oceanembed_L100.0", True)
    }

    models = {
        "MLR": joblib.load(f"experiments/phase16/mlr/fold_{fold}/model.joblib"),
        "XGB": joblib.load(f"experiments/phase16/xgboost/fold_{fold}/model.joblib")
    }
    
    for name, (exp_name, use_cbam) in spatial_models.items():
        models[name] = load_unet(exp_name, use_cbam, fold, device)

    # Tracking dictionaries
    results = {name: {"T_p": [], "S_p": [], "T_t": [], "S_t": []} for name in models.keys()}

    print("[3] Running Inference on Test Set (365 Days)...")
    with torch.no_grad():
        for batch_idx, (x, t_true, s_true) in enumerate(test_loader):
            
            # Tabular Data Extraction
            x_tab, y_tab_true = extract_tabular_data(x, t_true, s_true, ocean_mask, lat_grid, lon_grid)
            x_tab_np, y_tab_np = np.nan_to_num(x_tab.numpy(), nan=0.0), y_tab_true.numpy()
            t_true_tab, s_true_tab = y_tab_np[:, :15], y_tab_np[:, 15:]
            
            # MLR & XGBoost Predictions
            for name in ["MLR", "XGB"]:
                pred = models[name].predict(x_tab_np)
                results[name]["T_p"].append(pred[:, :15])
                results[name]["S_p"].append(pred[:, 15:])
                results[name]["T_t"].append(t_true_tab)
                results[name]["S_t"].append(s_true_tab)

            # Spatial CNN Predictions
            x_dev = x.to(device)
            tt_un = t_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy()
            ts_un = s_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy()

            for name in spatial_models.keys():
                t_pred, s_pred = models[name](x_dev)
                pt_un = t_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy()
                ps_un = s_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy()
                
                results[name]["T_p"].append(pt_un); results[name]["S_p"].append(ps_un)
                results[name]["T_t"].append(tt_un); results[name]["S_t"].append(ts_un)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches...")

    print("\n[4] Final Ablation Leaderboard")
    print(
        f"{'Model':<15} | "
        f"{'T RMSE':<7} | {'T MAE':<7} | {'T R²':<6} | {'T Bias':<7} | "
        f"{'S RMSE':<7} | {'S MAE':<7} | {'S R²':<6} | {'S Bias':<7} | "
        f"{'Inv Lyr %':<9} | {'Inv Col %':<9} | {'Agg Sev':<8}"
    )

    print("-" * 125)
    
    ordered_names = ["MLR", "XGB"] + list(spatial_models.keys())
    ground_truth_t = np.vstack(results["MLR"]["T_t"])
    ground_truth_s = np.vstack(results["MLR"]["S_t"])

    # Dynamically generate the 3D valid mask from Ground Truth NaNs!
    gt_valid_mask = ~np.isnan(ground_truth_t)

    true_inv_stats = calculate_inversion_statistics(
        ground_truth_t,
        ground_truth_s,
        valid_mask_3d=gt_valid_mask,
        tolerance=0.0
    )
    
    for name in ordered_names:
        t_p = np.vstack(results[name]["T_p"])
        t_t = np.vstack(results[name]["T_t"])
        s_p = np.vstack(results[name]["S_p"])
        s_t = np.vstack(results[name]["S_t"])
        
        valid_mask = ~np.isnan(t_t)

        # Ensure RMSE/MAE are only calculated on VALID ocean cells
        t_rmse, t_mae, t_r2, t_bias = calculate_metrics(t_p[valid_mask], t_t[valid_mask])
        s_rmse, s_mae, s_r2, s_bias = calculate_metrics(s_p[valid_mask], s_t[valid_mask])
        
        inv_stats = calculate_inversion_statistics(
            t_p, s_p, valid_mask_3d=valid_mask, tolerance=0.0
        )

        print(
            f"{name:<15} | "
            f"{t_rmse:<7.4f} | {t_mae:<7.4f} | {t_r2:<6.3f} | {t_bias:>7.4f} | "
            f"{s_rmse:<7.4f} | {s_mae:<7.4f} | {s_r2:<6.3f} | {s_bias:>7.4f} | "
            f"{inv_stats['layer_rate']:>8.2f}% | {inv_stats['column_rate']:>8.2f}% | {inv_stats['aggregate_severity']:>7.5f}"
        )

    print("-" * 125)
    print(
        f"{'GROUND TRUTH':<15} | "
        f"{'N/A':<7} | {'N/A':<7} | {'N/A':<6} | {'N/A':<7} | "
        f"{'N/A':<7} | {'N/A':<7} | {'N/A':<6} | {'N/A':<7} | "
        f"{true_inv_stats['layer_rate']:>8.2f}% | {true_inv_stats['column_rate']:>8.2f}% | {true_inv_stats['aggregate_severity']:>7.5f}"
    )
    print("=" * 125)

if __name__ == "__main__":
    main(fold=1)