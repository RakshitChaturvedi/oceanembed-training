import numpy as np
import torch
import joblib
import sys, os
import xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.models.oceanembed import OceanEmbedModel

def load_unet(exp_name, use_cbam, fold, device):
    model = OceanEmbedModel(in_channels=12, use_cbam=use_cbam).to(device)
    checkpoint = torch.load(f"experiments/phase16/{exp_name}/fold_{fold}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = get_dataloaders(fold=1, batch_size=16)
    
    models = {
        "OE_L0.1": ("oceanembed_L0.1", True),
        "OE_L1.0": ("oceanembed_L1.0", True),
        "OE_L10.0": ("oceanembed_L10.0", True),
        "OE_L100.0": ("oceanembed_L100.0", True)
    }
    
    loaded_models = {name: load_unet(exp, cbam, 1, device) for name, (exp, cbam) in models.items()}
    
    mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
    mask_bool = (torch.from_numpy(mask_ds.values) == 1).bool()

    # Accumulate predictions
    all_preds = {name: {"T": [], "S": []} for name in models.keys()}
    all_truths = {"T": [], "S": []}

    print("Extracting vertical profiles...")
    with torch.no_grad():
        for x, t_true, s_true in test_loader:
            x_dev = x.to(device)
            tt_un = t_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy()
            ts_un = s_true.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).numpy()
            all_truths["T"].append(tt_un)
            all_truths["S"].append(ts_un)
            
            for name, model in loaded_models.items():
                t_pred, s_pred = model(x_dev)
                pt = t_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy()
                ps = s_pred.permute(0, 2, 3, 1)[:, mask_bool, :].reshape(-1, 15).cpu().numpy()
                all_preds[name]["T"].append(pt)
                all_preds[name]["S"].append(ps)
            break # Just one batch is enough to prove the statistical collapse

    print("\nMean Vertical Standard Deviation (Variance across 15 depth layers)")
    print("-" * 65)
    print(f"{'Model':<15} | {'Temp Std (°C)':<15} | {'Salinity Std (PSU)':<15}")
    print("-" * 65)
    
    # Truth
    t_t = np.vstack(all_truths["T"]); s_t = np.vstack(all_truths["S"])
    
    # Filter out empty padded rows (where surface is exactly 0.0)
    valid_mask = (t_t[:, 0] != 0.0) & ~np.isnan(t_t[:, 0])
    valid_t = t_t[valid_mask]
    valid_s = s_t[valid_mask]
    
    # Use nanstd and nanmean to safely ignore any internal NaNs
    gt_t_std = np.nanmean(np.nanstd(valid_t, axis=1))
    gt_s_std = np.nanmean(np.nanstd(valid_s, axis=1))
    
    print(f"{'Ground Truth':<15} | {gt_t_std:<15.4f} | {gt_s_std:<15.4f}")
    # Models
    for name in models.keys():
        t_p = np.vstack(all_preds[name]["T"])
        s_p = np.vstack(all_preds[name]["S"])
        
        # Only measure variance where we actually have water columns
        valid_mask = np.vstack(all_truths["T"])[:, 0] != 0
        t_valid = t_p[valid_mask]
        s_valid = s_p[valid_mask]
        
        t_std = np.mean(np.std(t_valid, axis=1))
        s_std = np.mean(np.std(s_valid, axis=1))
        
        print(f"{name:<15} | {t_std:<15.4f} | {s_std:<15.4f}")
    print("\nAcross-Sample Spatial Standard Deviation (Salinity)")
    print("-" * 65)
    print(f"{'Model':<15} | {'Surface (Depth 0) Std':<25} | {'Deep (Depth 10) Std':<25}")
    print("-" * 65)
    
    # Ground Truth Spatial Variance
    gt_s_spatial_0 = np.nanstd(valid_s[:, 0])
    gt_s_spatial_10 = np.nanstd(valid_s[:, 10])
    print(f"{'Ground Truth':<15} | {gt_s_spatial_0:<25.4f} | {gt_s_spatial_10:<25.4f}")
    
    # Model Spatial Variance
    for name in models.keys():
        s_p = np.vstack(all_preds[name]["S"])
        s_valid = s_p[valid_mask]
        
        mod_spatial_0 = np.std(s_valid[:, 0])
        mod_spatial_10 = np.std(s_valid[:, 10])
        print(f"{name:<15} | {mod_spatial_0:<25.4f} | {mod_spatial_10:<25.4f}")
    print("-" * 65)
    
    sal = xr.open_dataset("data/processed/phase7/salinity_targets_raw.nc")["salinity"]
    mask_bool_np = mask_ds.values == 1  # your 2D ocean mask

    for d_idx, depth_val in enumerate(sal["depth"].values):
        layer = sal.isel(time=0, depth=d_idx).values
        ocean_pixels = layer[mask_bool_np]
        nan_frac = np.isnan(ocean_pixels).mean()
        print(f"depth={depth_val:>6.1f}m  NaN fraction among 'ocean' pixels: {nan_frac:.2%}")

if __name__ == "__main__":
    main()