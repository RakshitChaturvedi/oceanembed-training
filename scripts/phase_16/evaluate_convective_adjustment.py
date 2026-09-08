import sys
import os
import torch
import numpy as np
import joblib
import xarray as xr
import gc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.evaluation.tabular_dataset import extract_tabular_data
from src.models.oceanembed import OceanEmbedModel
from src.physics.eos import linear_eos

def calculate_metrics(preds, truths):
    valid_mask = ~np.isnan(truths) & (truths != 0.0)
    preds_valid = preds[valid_mask]
    truths_valid = truths[valid_mask]
    
    if len(truths_valid) == 0: return 0.0, 0.0, 0.0, 0.0
        
    mse = np.mean((preds_valid - truths_valid) ** 2)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((truths_valid - preds_valid) ** 2)
    ss_tot = np.sum((truths_valid - np.mean(truths_valid)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return rmse, r2

def calculate_inversion_statistics(t_preds, s_preds, valid_mask_3d, tolerance=0.0):
    if len(t_preds) == 0: return {"layer_rate": 0.0, "column_rate": 0.0, "aggregate_severity": 0.0}

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
    aggregate_severity = (inversion_magnitudes.sum().item() / total_pairs if total_pairs > 0 else 0.0)

    return {"layer_rate": layer_rate, "column_rate": column_rate, "aggregate_severity": aggregate_severity}

def apply_convective_adjustment(t_preds, s_preds, valid_mask_3d):
    """
    Physically mixes unstable layers iteratively and logs the intervention intensity.
    Returns: (t_adj, s_adj, passes_run, fraction_touched_percent)
    """
    t_adj = torch.tensor(t_preds, dtype=torch.float32)
    s_adj = torch.tensor(s_preds, dtype=torch.float32)
    mask = torch.tensor(valid_mask_3d, dtype=torch.bool)
    
    # Track which cells get modified
    touched = torch.zeros_like(mask, dtype=torch.bool)
    passes_run = 0
    
    dz = torch.tensor([5, 5, 10, 10, 20, 25, 25, 25, 25, 50, 100, 200, 200, 300, 300], dtype=torch.float32)
    
    for p in range(15):
        density = linear_eos(t_adj, s_adj)
        delta_rho = density[:, 1:] - density[:, :-1]
        inv_mask = (delta_rho < -1e-6) & mask[:, 1:] & mask[:, :-1]
        
        if not inv_mask.any(): 
            break
            
        passes_run += 1
            
        for parity in [0, 1]:
            for z in range(parity, 14, 2):
                rho_z = linear_eos(t_adj[:, z], s_adj[:, z])
                rho_z1 = linear_eos(t_adj[:, z+1], s_adj[:, z+1])
                
                is_inv = (rho_z > rho_z1 + 1e-6) & mask[:, z] & mask[:, z+1]
                if not is_inv.any(): continue
                    
                w1, w2 = dz[z], dz[z+1]
                w_tot = w1 + w2
                
                t_mix = (t_adj[:, z]*w1 + t_adj[:, z+1]*w2) / w_tot
                s_mix = (s_adj[:, z]*w1 + s_adj[:, z+1]*w2) / w_tot
                
                t_adj[:, z] = torch.where(is_inv, t_mix, t_adj[:, z])
                t_adj[:, z+1] = torch.where(is_inv, t_mix, t_adj[:, z+1])
                s_adj[:, z] = torch.where(is_inv, s_mix, s_adj[:, z])
                s_adj[:, z+1] = torch.where(is_inv, s_mix, s_adj[:, z+1])
                
                # Mark these cells as mathematically altered
                touched[:, z] |= is_inv
                touched[:, z+1] |= is_inv
                
    # Calculate what fraction of the VALID ocean was overwritten
    total_valid = mask.sum().item()
    total_touched = (touched & mask).sum().item()
    frac_touched = (total_touched / total_valid * 100.0) if total_valid > 0 else 0.0
                
    return t_adj.numpy(), s_adj.numpy(), passes_run, frac_touched
def load_unet(experiment_name, use_cbam, fold, device):
    model = OceanEmbedModel(in_channels=12, use_cbam=use_cbam).to(device)
    checkpoint = torch.load(f"experiments/phase16/{experiment_name}/fold_{fold}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def main(fold=1):
    print("=" * 100)
    print("Phase 16.11: Memory-Optimized Post-Processing Evaluation")
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("[1] Loading Geographic Artifacts...")
    mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
    ocean_mask = torch.from_numpy(mask_ds.values).float()
    lat_coords = mask_ds.coords[mask_ds.dims[0]].values
    lon_coords = mask_ds.coords[mask_ds.dims[1]].values
    lon_2d, lat_2d = np.meshgrid(lon_coords, lat_coords)
    lat_grid = torch.from_numpy(lat_2d).float()
    lon_grid = torch.from_numpy(lon_2d).float()
    mask_bool = (ocean_mask == 1).bool()

    _, _, test_loader = get_dataloaders(fold=fold, batch_size=16)

    model_configs = {
        "MLR": "tabular",
        "XGB": "tabular",
        "UNet": ("plain_unet", False),
        "UNet+Phys": ("plain_unet_physics", False),
        "CBAM": ("cbam_unet", True),
        "oceanembed": ("oceanembed", True), # Added your missing 0.01 run here!
        "OE_L0.1": ("oceanembed_L0.1", True),
        "OE_L1.0": ("oceanembed_L1.0", True),
        "OE_L10.0": ("oceanembed_L10.0", True),
        "OE_L100.0": ("oceanembed_L100.0", True)
    }

    raw_results = {}
    adj_results = {}
    gt_inv_stats = None  # To store the Ground Truth stats

    print("[2] Evaluating Models Sequentially (Memory Safe Mode)...")
    
    for name, config in model_configs.items():
        print(f"\n  Evaluating {name}...")
        
        # 1. Load Model dynamically
        if config == "tabular":
            if name == "MLR": model = joblib.load(f"experiments/phase16/mlr/fold_{fold}/model.joblib")
            if name == "XGB": model = joblib.load(f"experiments/phase16/xgboost/fold_{fold}/model.joblib")
        else:
            model = load_unet(config[0], config[1], fold, device)

        # 2. Accumulate predictions for this single model
        T_p, S_p, T_t, S_t = [], [], [], []
        
        with torch.no_grad():
            for x, t_true, s_true in test_loader:
                if config == "tabular":
                    x_tab, y_tab = extract_tabular_data(x, t_true, s_true, ocean_mask, lat_grid, lon_grid)
                    x_tab_np, y_tab_np = np.nan_to_num(x_tab.numpy(), nan=0.0), y_tab.numpy()
                    pred = model.predict(x_tab_np)
                    T_p.append(pred[:, :15]); S_p.append(pred[:, 15:])
                    T_t.append(y_tab_np[:, :15]); S_t.append(y_tab_np[:, 15:])
                else:
                    tt_un = t_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy()
                    ts_un = s_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy()
                    t_pred, s_pred = model(x.to(device))
                    T_p.append(t_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy())
                    S_p.append(s_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy())
                    T_t.append(tt_un); S_t.append(ts_un)

        # Combine arrays
        t_p_arr, s_p_arr = np.vstack(T_p), np.vstack(S_p)
        t_t_arr, s_t_arr = np.vstack(T_t), np.vstack(S_t)
        gt_valid_mask = ~np.isnan(t_t_arr)

        # Calculate Ground Truth stats once during the first loop
        if gt_inv_stats is None:
            gt_inv_stats = calculate_inversion_statistics(t_t_arr, s_t_arr, gt_valid_mask)

        # 3. Calculate Raw Metrics
        t_rmse, t_r2 = calculate_metrics(t_p_arr[gt_valid_mask], t_t_arr[gt_valid_mask])
        s_rmse, s_r2 = calculate_metrics(s_p_arr[gt_valid_mask], s_t_arr[gt_valid_mask])
        raw_inv = calculate_inversion_statistics(t_p_arr, s_p_arr, gt_valid_mask)
        raw_results[name] = (t_rmse, t_r2, s_rmse, s_r2, raw_inv)

        # 4. Apply Adjustment & Calculate Adjusted Metrics
        print(f"    -> Applying Convective Adjustment...")
        t_adj_arr, s_adj_arr, passes, frac = apply_convective_adjustment(t_p_arr, s_p_arr, gt_valid_mask)
        
        t_rmse_a, t_r2_a = calculate_metrics(t_adj_arr[gt_valid_mask], t_t_arr[gt_valid_mask])
        s_rmse_a, s_r2_a = calculate_metrics(s_adj_arr[gt_valid_mask], s_t_arr[gt_valid_mask])
        adj_inv = calculate_inversion_statistics(t_adj_arr, s_adj_arr, gt_valid_mask)
        
        adj_results[name] = (t_rmse_a, t_r2_a, s_rmse_a, s_r2_a, adj_inv, passes, frac)

        # 5. Clear arrays
        del model, T_p, S_p, T_t, S_t, t_p_arr, s_p_arr, t_t_arr, s_t_arr, t_adj_arr, s_adj_arr, gt_valid_mask
        gc.collect()

    print("\n[3] Generating Final Comparison Tables...")
    
    def print_leaderboard(title, results_dict, is_adjusted=False):
        print(f"\n{title}")
        if is_adjusted:
            print(f"{'Model':<12} | {'T RMSE':<7} | {'T R²':<6} | {'S RMSE':<7} | {'S R²':<6} | {'Inv Lyr %':<9} | {'Inv Col %':<9} | {'Agg Sev':<8} | {'% Touched':<9}")
            print("-" * 110)
        else:
            print(f"{'Model':<12} | {'T RMSE':<7} | {'T R²':<6} | {'S RMSE':<7} | {'S R²':<6} | {'Inv Lyr %':<9} | {'Inv Col %':<9} | {'Agg Sev':<8}")
            print("-" * 95)
            
        for name, stats in results_dict.items():
            if is_adjusted:
                t_rmse, t_r2, s_rmse, s_r2, inv, passes, frac = stats
                print(f"{name:<12} | {t_rmse:<7.4f} | {t_r2:<6.3f} | {s_rmse:<7.4f} | {s_r2:<6.3f} | {inv['layer_rate']:>8.2f}% | {inv['column_rate']:>8.2f}% | {inv['aggregate_severity']:>7.5f} | {frac:>8.2f}%")
            else:
                t_rmse, t_r2, s_rmse, s_r2, inv = stats
                print(f"{name:<12} | {t_rmse:<7.4f} | {t_r2:<6.3f} | {s_rmse:<7.4f} | {s_r2:<6.3f} | {inv['layer_rate']:>8.2f}% | {inv['column_rate']:>8.2f}% | {inv['aggregate_severity']:>7.5f}")
        
        # Print Ground Truth
        if is_adjusted:
            print("-" * 110)
            print(f"{'GROUND TRUTH':<12} | {'N/A':<7} | {'N/A':<6} | {'N/A':<7} | {'N/A':<6} | {gt_inv_stats['layer_rate']:>8.2f}% | {gt_inv_stats['column_rate']:>8.2f}% | {gt_inv_stats['aggregate_severity']:>7.5f} | {'N/A':<9}")
            print("=" * 110)
        else:
            print("-" * 95)
            print(f"{'GROUND TRUTH':<12} | {'N/A':<7} | {'N/A':<6} | {'N/A':<7} | {'N/A':<6} | {gt_inv_stats['layer_rate']:>8.2f}% | {gt_inv_stats['column_rate']:>8.2f}% | {gt_inv_stats['aggregate_severity']:>7.5f}")
            print("=" * 95)

    print_leaderboard("TABLE A: RAW PREDICTIONS (Before Adjustment)", raw_results, is_adjusted=False)
    print_leaderboard("TABLE B: POST-PROCESSED (With Convective Adjustment)", adj_results, is_adjusted=True)

if __name__ == "__main__":
    main(fold=1)