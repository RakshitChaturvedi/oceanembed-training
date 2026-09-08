import json
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from src.data.dataset import OceanDataset

def get_dataloaders(
    fold: int, 
    split_config_path: str = "data/processed/phase8/split_config.json",
    input_nc: str = "data/processed/phase6/input_tensor_normalized.nc",
    target_t_nc: str = "data/processed/phase7/temperature_targets_masked.nc",
    target_s_nc: str = "data/processed/phase7/salinity_targets_masked.nc",
    batch_size: int = 16
):
    with open(split_config_path, 'r') as f:
        config = json.load(f)
        
    fold_data = next((f for f in config["folds"] if f["fold_id"] == fold), None)
    if fold_data is None: raise ValueError(f"Fold {fold} not found.")

    def expand_indices(idx_data):
        if isinstance(idx_data[0], list):
            indices = []
            for rng in idx_data: indices.extend(list(range(rng[0], rng[1])))
            return indices
        else:
            return list(range(idx_data[0], idx_data[1]))

    train_indices = expand_indices(fold_data["train_indices"])
    val_indices = expand_indices(fold_data["validation_indices"])
    test_indices = expand_indices(fold_data["test_indices"])

    print(f"Fold {fold} Setup | Train: {len(train_indices)} days | Val: {len(val_indices)} days | Test: {len(test_indices)} days")

    # ---------------------------------------------------------
    # MEMORY OPTIMIZATION: Load ONCE as float32 (~5.7 GB total)
    # ---------------------------------------------------------
    print("  -> Loading NetCDF files into memory (float32)...")
    inputs_arr = xr.open_dataarray(input_nc).astype(np.float32).values
    targets_t_arr = xr.open_dataarray(target_t_nc).astype(np.float32).values
    targets_s_arr = xr.open_dataarray(target_s_nc).astype(np.float32).values

    # Pass the shared memory references to the datasets
    train_dataset = OceanDataset(inputs_arr, targets_t_arr, targets_s_arr, train_indices)
    val_dataset = OceanDataset(inputs_arr, targets_t_arr, targets_s_arr, val_indices)
    test_dataset = OceanDataset(inputs_arr, targets_t_arr, targets_s_arr, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader