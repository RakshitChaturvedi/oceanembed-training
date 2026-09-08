import sys
import os
import numpy as np
import xarray as xr
import torch
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.data.loader import get_dataloaders

# =============================================================================
# Config
# =============================================================================

UNCERTAINTY_DIR = "experiments/phase16/oe_production_ens"
OCEAN_MASK_PATH = "data/processed/phase5/ocean_mask.nc"
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
TOP_N_LOCATIONS = 25

# Which variable(s) drive the recommendation.
# "combined" averages normalized T and S uncertainty; use "temperature" or
# "salinity" alone if you want a single-variable ranking instead.
UNCERTAINTY_SOURCE = "combined"


def load_flat_uncertainty():
    t = np.load(os.path.join(UNCERTAINTY_DIR, "uncertainty_t.npy"))  # [N, 15]
    s = np.load(os.path.join(UNCERTAINTY_DIR, "uncertainty_s.npy"))  # [N, 15]
    return t, s


def get_spatial_context(fold=1):
    """
    Recovers everything needed to map flat [N, 15] rows back to (day, lat, lon):
      - the 2D ocean mask and its boolean version
      - lat/lon coordinate arrays
      - number of test days (so we know how to reshape N -> [days, pixels])
    """
    mask_ds = xr.open_dataarray(OCEAN_MASK_PATH)
    mask_2d = mask_ds.values                      # [H, W]
    mask_bool = (mask_2d == 1)
    lat_coords = mask_ds.coords[mask_ds.dims[0]].values
    lon_coords = mask_ds.coords[mask_ds.dims[1]].values

    _, _, test_loader = get_dataloaders(fold=fold, batch_size=16)
    num_test_days = len(test_loader.dataset)

    return mask_2d, mask_bool, lat_coords, lon_coords, num_test_days


def unflatten_to_grid(flat_arr, mask_bool, num_days):
    """
    flat_arr: [N, 15] where N = num_days * num_ocean_pixels, ordered
              (day, pixel-within-mask) to match how the eval script built it
              (shuffle=False test_loader, vstacked in loader order).

    Returns: [num_days, 15, H, W] with NaN at land cells.
    """
    num_ocean_pixels = int(mask_bool.sum())
    expected_n = num_days * num_ocean_pixels

    if flat_arr.shape[0] != expected_n:
        raise ValueError(
            f"Shape mismatch: got N={flat_arr.shape[0]}, expected "
            f"num_days({num_days}) * num_ocean_pixels({num_ocean_pixels}) = {expected_n}. "
            f"Check that this uncertainty array was built from the same fold/mask "
            f"you're loading context for."
        )

    n_depth = flat_arr.shape[1]
    H, W = mask_bool.shape

    reshaped = flat_arr.reshape(num_days, num_ocean_pixels, n_depth)  # [days, pixels, depth]

    grid = np.full((num_days, n_depth, H, W), np.nan, dtype=np.float32)
    for d in range(num_days):
        for k in range(n_depth):
            layer = np.full((H, W), np.nan, dtype=np.float32)
            layer[mask_bool] = reshaped[d, :, k]
            grid[d, k] = layer

    return grid  # [days, depth, H, W]


def build_ranking(t_uncert_grid, s_uncert_grid, lat_coords, lon_coords, mask_bool):
    """
    Collapses [days, depth, H, W] -> a single per-pixel score by:
      1. averaging over time (persistent uncertainty, not a one-day blip)
      2. taking the max over depth (a location is worth flagging if ANY
         depth is poorly constrained, not only if all of them are)
    Then normalizes T and S scores independently before combining, since
    they're in different physical units and shouldn't be averaged raw.
    """
    t_time_avg = np.nanmean(t_uncert_grid, axis=0)   # [depth, H, W]
    s_time_avg = np.nanmean(s_uncert_grid, axis=0)   # [depth, H, W]

    t_score_2d = np.nanmax(t_time_avg, axis=0)       # [H, W]
    s_score_2d = np.nanmax(s_time_avg, axis=0)       # [H, W]

    def normalize(arr):
        valid = arr[~np.isnan(arr)]
        lo, hi = valid.min(), valid.max()
        return (arr - lo) / (hi - lo + 1e-12)

    t_norm = normalize(t_score_2d)
    s_norm = normalize(s_score_2d)

    if UNCERTAINTY_SOURCE == "temperature":
        combined = t_norm
    elif UNCERTAINTY_SOURCE == "salinity":
        combined = s_norm
    else:
        combined = (t_norm + s_norm) / 2.0

    rows = []
    H, W = combined.shape
    for i in range(H):
        for j in range(W):
            if not mask_bool[i, j]:
                continue
            rows.append({
                "latitude": float(lat_coords[i]),
                "longitude": float(lon_coords[j]),
                "temp_uncertainty_norm": float(t_norm[i, j]),
                "salinity_uncertainty_norm": float(s_norm[i, j]),
                "combined_score": float(combined[i, j]),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    return df


def main():
    print("=" * 80)
    print("ARGO Float Deployment Recommendation (Uncertainty-Guided)")
    print("=" * 80)

    print("\n[1] Loading flat ensemble uncertainty arrays...")
    t_flat, s_flat = load_flat_uncertainty()
    print(f"    T uncertainty shape: {t_flat.shape}")
    print(f"    S uncertainty shape: {s_flat.shape}")

    print("\n[2] Recovering spatial context (mask, coords, fold length)...")
    mask_2d, mask_bool, lat_coords, lon_coords, num_test_days = get_spatial_context(fold=1)
    print(f"    Ocean pixels: {int(mask_bool.sum())}  |  Test days: {num_test_days}")

    print("\n[3] Unflattening to [days, depth, H, W] grids...")
    t_grid = unflatten_to_grid(t_flat, mask_bool, num_test_days)
    s_grid = unflatten_to_grid(s_flat, mask_bool, num_test_days)

    print("\n[4] Building ranked deployment recommendations...")
    ranking = build_ranking(t_grid, s_grid, lat_coords, lon_coords, mask_bool)

    print(f"\n[5] Top {TOP_N_LOCATIONS} recommended ARGO deployment locations:")
    print("-" * 80)
    top = ranking.head(TOP_N_LOCATIONS)
    print(top.to_string(index=False))

    out_path = "experiments/phase16/oe_production_ens/argo_recommendations.csv"
    ranking.to_csv(out_path, index=False)
    print(f"\nFull ranked list saved to: {out_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()