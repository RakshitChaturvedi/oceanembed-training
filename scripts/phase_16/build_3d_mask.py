import xarray as xr
import numpy as np
import os

def main():
    print("=" * 60)
    print("Phase 16 Hotfix: Building 3D Ocean Bathymetry Mask")
    print("=" * 60)

    # 1. Load raw targets to check the real NaN layout
    print("[1] Loading raw salinity targets to verify bathymetry...")
    raw_path = "data/processed/phase7/salinity_targets_raw.nc"
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Could not find {raw_path}")
        
    ds = xr.open_dataset(raw_path)
    sal = ds["salinity"]  # Dimensions: [time, lat, lon, depth]

    # 2. Verify static bathymetry across time
    print("[2] Checking if seafloor layout is static across time...")
    nan_t0 = np.isnan(sal.isel(time=0).values)       # [lat, lon, depth]
    nan_tend = np.isnan(sal.isel(time=-1).values)    # [lat, lon, depth]
    
    is_static = np.array_equal(nan_t0, nan_tend)
    print(f"    Static across time: {is_static}")
    
    if not is_static:
        print("    ERROR: Bathymetry changes over time! This dataset has moving land/seafloor.")
        return

    # 3. Generate the 3D Mask
    print("\n[3] Generating 3D depth-resolved mask...")
    # Load the original 2D mask [lat, lon]
    mask_2d = xr.open_dataarray("data/processed/phase5/ocean_mask.nc").values
    
    # Expand 2D mask to [lat, lon, 1] so it broadcasts with [lat, lon, depth]
    mask_2d_expanded = np.expand_dims(mask_2d, axis=-1)
    
    # 3D mask is True where (2D mask is True) AND (Raw data is NOT NaN)
    mask_3d_latlondepth = np.logical_and(mask_2d_expanded, ~nan_t0)

    # Transpose to [depth, lat, lon] for PyTorch channel alignment (15, 101, 241)
    mask_3d = mask_3d_latlondepth.transpose(2, 0, 1)

    # 4. Save the new artifact
    print("\n[4] Saving artifact...")
    ds_3d = xr.DataArray(
        mask_3d.astype(int),
        coords={
            "depth": sal.coords["depth"],
            "latitude": sal.coords["latitude"],
            "longitude": sal.coords["longitude"]
        },
        dims=["depth", "latitude", "longitude"]
    )
    
    out_path = "data/processed/phase5/ocean_mask_3d.nc"
    ds_3d.to_netcdf(out_path)
    print(f"    Saved successfully to: {out_path}")
    print(f"    Shape: {mask_3d.shape} (Depth, Lat, Lon)")
    
    # 5. Quick Verification Print
    print("\n[5] Depth-wise Valid Pixel Retention (vs Surface 2D Mask):")
    total_2d_ocean = mask_2d.sum()
    for d in range(mask_3d.shape[0]):
        retained = mask_3d[d].sum()
        pct = (retained / total_2d_ocean) * 100
        print(f"    Depth {d:2d} ({sal.coords['depth'].values[d]:6.1f}m): {pct:5.2f}% valid ocean")

if __name__ == "__main__":
    main()