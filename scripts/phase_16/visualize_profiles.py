import sys
import os
import torch
import numpy as np
import joblib
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders
from src.evaluation.tabular_dataset import extract_tabular_data
from src.models.oceanembed import OceanEmbedModel

def load_unet(experiment_name, use_cbam, fold, device):
    model = OceanEmbedModel(in_channels=12, use_cbam=use_cbam).to(device)
    checkpoint = torch.load(f"experiments/phase16/{experiment_name}/fold_{fold}/best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

def main(fold=1):
    print("=" * 60)
    print("Phase 16: Visualizing Vertical Profiles")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Artifacts
    print("[1] Loading Artifacts...")
    mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
    ocean_mask = torch.from_numpy(mask_ds.values).float()
    lat_coords = mask_ds.coords[mask_ds.dims[0]].values
    lon_coords = mask_ds.coords[mask_ds.dims[1]].values
    lon_2d, lat_2d = np.meshgrid(lon_coords, lat_coords)
    lat_grid = torch.from_numpy(lat_2d).float()
    lon_grid = torch.from_numpy(lon_2d).float()
    
    # Standard 15 depth layers used in GLORYS
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    _, _, test_loader = get_dataloaders(fold=fold, batch_size=1)

    print("[2] Loading Models...")
    models = {
        "MLR": joblib.load(f"experiments/phase16/mlr/fold_{fold}/model.joblib"),
        "XGB": joblib.load(f"experiments/phase16/xgboost/fold_{fold}/model.joblib"),
        "UNet": load_unet("plain_unet", False, fold, device),
        "UNet+Phys": load_unet("plain_unet_physics", False, fold, device)
    }

    print("[3] Extracting a single test sample...")
    with torch.no_grad():
        # Just grab the first batch
        x, t_true, s_true = next(iter(test_loader))
        
        # Extract tabular for baselines
        x_tab, y_tab = extract_tabular_data(x, t_true, s_true, ocean_mask, lat_grid, lon_grid)
        x_tab_np, y_tab_np = np.nan_to_num(x_tab.numpy(), nan=0.0), y_tab.numpy()
        
        # Run CNNs
        x_dev = x.to(device)
        t_unet, s_unet = models["UNet"](x_dev)
        t_phys, s_phys = models["UNet+Phys"](x_dev)

    # 4. Find a valid ocean pixel (deep enough to have no NaNs)
    # y_tab_np has shape [valid_pixels, 30]. We want a row with NO NaNs at all.
    valid_rows = np.where(~np.isnan(y_tab_np).any(axis=1))[0]
    
    if len(valid_rows) == 0:
        print("Could not find a fully deep water column in this batch. Run again!")
        return
        
    # Pick the first valid deep-ocean pixel
    target_idx = valid_rows[0]
    
    # Extract Ground Truth
    gt_t = y_tab_np[target_idx, :15]
    gt_s = y_tab_np[target_idx, 15:]
    
    # Extract Tabular Predictions
    mlr_pred = models["MLR"].predict(x_tab_np[target_idx].reshape(1, -1))[0]
    xgb_pred = models["XGB"].predict(x_tab_np[target_idx].reshape(1, -1))[0]
    
    # To get the CNN predictions for this exact pixel, we use the 2D mask indices
    # extract_tabular_data flattens using ocean_mask == 1. 
    valid_y, valid_x = torch.where(ocean_mask == 1)
    pixel_y, pixel_x = valid_y[target_idx], valid_x[target_idx]
    
    unet_t = t_unet[0, :, pixel_y, pixel_x].cpu().numpy()
    unet_s = s_unet[0, :, pixel_y, pixel_x].cpu().numpy()
    phys_t = t_phys[0, :, pixel_y, pixel_x].cpu().numpy()
    phys_s = s_phys[0, :, pixel_y, pixel_x].cpu().numpy()

    # 5. Plotting
    print("[4] Generating Visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    
    colors = {"MLR": "gray", "XGB": "orange", "UNet": "blue", "UNet+Phys": "purple"}
    styles = {"MLR": ":", "XGB": "--", "UNet": "-", "UNet+Phys": "-."}

    # Temperature Plot
    ax1.plot(gt_t, depths, label='Ground Truth', color='black', linewidth=3, zorder=5)
    ax1.plot(mlr_pred[:15], depths, label='MLR', color=colors["MLR"], linestyle=styles["MLR"], linewidth=2)
    ax1.plot(xgb_pred[:15], depths, label='XGBoost', color=colors["XGB"], linestyle=styles["XGB"], linewidth=2)
    ax1.plot(unet_t, depths, label='UNet', color=colors["UNet"], linestyle=styles["UNet"], linewidth=2)
    ax1.plot(phys_t, depths, label='UNet+Phys', color=colors["UNet+Phys"], linestyle=styles["UNet+Phys"], linewidth=2)
    
    ax1.set_title("Temperature Profile", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Temperature (°C)", fontsize=12)
    ax1.set_ylabel("Depth (m)", fontsize=12)
    ax1.invert_yaxis() # Oceanography standard: surface at top
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Salinity Plot
    ax2.plot(gt_s, depths, label='Ground Truth', color='black', linewidth=3, zorder=5)
    ax2.plot(mlr_pred[15:], depths, label='MLR', color=colors["MLR"], linestyle=styles["MLR"], linewidth=2)
    ax2.plot(xgb_pred[15:], depths, label='XGBoost', color=colors["XGB"], linestyle=styles["XGB"], linewidth=2)
    ax2.plot(unet_s, depths, label='UNet', color=colors["UNet"], linestyle=styles["UNet"], linewidth=2)
    ax2.plot(phys_s, depths, label='UNet+Phys', color=colors["UNet+Phys"], linestyle=styles["UNet+Phys"], linewidth=2)
    
    ax2.set_title("Salinity Profile", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Salinity (PSU)", fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.suptitle(f"Model Comparison: Vertical Profile at Lat {lat_coords[pixel_y]:.2f}, Lon {lon_coords[pixel_x]:.2f}", fontsize=16)
    plt.tight_layout()
    
    save_path = "profile_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [✓] Visualization saved to: {save_path}")

if __name__ == "__main__":
    main()