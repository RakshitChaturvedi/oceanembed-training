import sys, os, torch, numpy as np, time, xarray as xr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.evaluation.tabular_dataset import extract_tabular_data
from src.baselines.mlr import MLRBaseline
from src.baselines.xgboost import XGBoostBaseline
from src.data.loader import get_dataloaders

def build_tabular_matrices(dataloader, ocean_mask, lat_grid, lon_grid):
    # loops dataloader and flattens all batches into np arrs
    X_list, Y_list = [], []

    for batch_idx, (x, t_true, s_true) in enumerate(dataloader):
        X_tab, Y_tab = extract_tabular_data(x, t_true, s_true, ocean_mask, lat_grid, lon_grid)
        X_list.append(X_tab.cpu().numpy())
        Y_list.append(Y_tab.cpu().numpy())

        if (batch_idx + 1)%50 == 0:
            print(f"Processed {batch_idx+1}/{len(dataloader)} batches")
    return np.vstack(X_list), np.vstack(Y_list)

def train_baselines(fold: int=1):
    print("="*60)
    print(f"Phase 16.9: Training MLR & XGBoost (Fold {fold})")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load Geographic Artifacts (Phase 5)
    # ---------------------------------------------------------
    print("[1] Loading Phase 5 Geographic Artifacts and Generating Grids...")
    
    try:
        # Load the 2D mask
        mask_ds = xr.open_dataarray("data/processed/phase5/ocean_mask.nc")
        ocean_mask = torch.from_numpy(mask_ds.values).float()
        
        # Extract the 1D coordinate arrays (assuming they are named 'lat' and 'lon' or similar)
        # We use .coords to dynamically fetch whatever the lat/lon dimension names are
        lat_coords = mask_ds.coords[mask_ds.dims[0]].values # e.g., 101 values
        lon_coords = mask_ds.coords[mask_ds.dims[1]].values # e.g., 241 values
        
        # Create 2D grids mapping exactly to the [101, 241] shape
        lon_2d, lat_2d = np.meshgrid(lon_coords, lat_coords)
        
        lat_grid = torch.from_numpy(lat_2d).float()
        lon_grid = torch.from_numpy(lon_2d).float()
        
        print(f"  [✓] Mask Shape: {ocean_mask.shape}")
        print(f"  [✓] Lat/Lon Grids Generated: {lat_grid.shape}")
        
    except Exception as e:
        print(f"FATAL ERROR: Could not load ocean_mask.nc or extract coordinates: {e}")
        print("Please check your Phase 5 mask path.")
        sys.exit(1)

    # load dataloaders
    print(f"\n[2] initializing phase 8 dataloaders for fold {fold}")
    train_loader, val_loader, test_loader = get_dataloaders(fold=fold, batch_size=32)

    # extracting tabular matrices
    print("\n[3] Flattening spatial dataset into tabular matrices...")
    X_train, Y_train = build_tabular_matrices(train_loader, ocean_mask, lat_grid, lon_grid)

    print("  [!] Sanitizing bathymetry NaNs for Scikit-Learn/XGBoost...")
    # 1. Fill NaNs in inputs (spatial context from land is fine as 0.0)
    X_train = np.nan_to_num(X_train, nan=0.0)
    
    # 2. Drop any rows where the targets (Y) contain NaNs (seafloor cutoffs)
    valid_train_rows = ~np.isnan(Y_train).any(axis=1)
    X_train, Y_train = X_train[valid_train_rows], Y_train[valid_train_rows]

    print(f"  [✓] Train Tabular Extraction Complete (Dropped invalid rows).")
    print(f"      Inputs (X):  {X_train.shape}  |  Targets (Y): {Y_train.shape}")

    mlr_dir = f"experiments/phase16/mlr/fold_{fold}"

    mlr_dir = f"experiments/phase16/mlr/fold_{fold}"
    xgb_dir = f"experiments/phase16/xgboost/fold_{fold}"
    os.makedirs(mlr_dir, exist_ok=True)
    os.makedirs(xgb_dir, exist_ok=True)

    # train mlr
    print("\n[4] Training MLR Baseline...")
    mlr = MLRBaseline()
    
    start_time = time.time()
    mlr.fit(X_train, Y_train)
    mlr_time = time.time() - start_time
    
    mlr_path = os.path.join(mlr_dir, "model.joblib")
    mlr.save(mlr_path)
    print(f"  [✓] MLR saved to {mlr_path} (Took {mlr_time:.2f} seconds)")

    # train xgboost
    print("\n[5] Training XGBoost Baseline (MultiOutput GPU)...")
    xgb = XGBoostBaseline(use_gpu=True)
    
    start_time = time.time()
    xgb.fit(X_train, Y_train)
    xgb_time = time.time() - start_time
    
    xgb_path = os.path.join(xgb_dir, "model.joblib")
    xgb.save(xgb_path)
    print(f"  [✓] XGBoost saved to {xgb_path} (Took {xgb_time/60:.2f} minutes)")

    print("\n" + "=" * 60)
    print(f"PHASE 16.9 COMPLETE: FOLD {fold} BASELINES TRAINED")
    print("=" * 60)

if __name__ == "__main__":
    train_baselines(fold=1)
