import xarray as xr
import numpy as np
import os

def main():
    print("=" * 60)
    print("Phase 16 Hotfix: Cross-Variable Bathymetry Check")
    print("=" * 60)
    
    temp_path = "data/processed/phase7/temperature_targets_raw.nc"
    sal_path = "data/processed/phase7/salinity_targets_raw.nc"
    
    if not os.path.exists(temp_path) or not os.path.exists(sal_path):
        print(f"Error: Missing one of the target files.")
        return

    print("Loading Temperature and Salinity targets...")
    temp_ds = xr.open_dataset(temp_path)["temperature"]
    sal_ds = xr.open_dataset(sal_path)["salinity"]
    
    # Check spatial alignment at time = 0
    nan_temp_t0 = np.isnan(temp_ds.isel(time=0).values)
    nan_sal_t0 = np.isnan(sal_ds.isel(time=0).values)
    
    is_identical = np.array_equal(nan_temp_t0, nan_sal_t0)
    print(f"\nTemp/Salinity NaN patterns identical: {is_identical}")
    
    if not is_identical:
        diff_count = np.sum(nan_temp_t0 != nan_sal_t0)
        total_pixels = nan_temp_t0.size
        print(f"WARNING: Found {diff_count} mismatched pixels out of {total_pixels}!")
    else:
        print("SUCCESS: 3D Mask is completely safe to use for both variables.")

if __name__ == "__main__":
    main()