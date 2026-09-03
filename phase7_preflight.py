import xarray as xr
import numpy as np
from pathlib import Path

GLORYS = Path("data/processed/phase2/glorys.nc")
INPUT = Path("data/processed/phase6/normalized_tensor.nc")
MASK = Path("data/processed/phase5/ocean_mask.nc")

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700]

print("=" * 90)
print("OceanEmbed — Phase 7 GLORYS Preflight")
print("=" * 90)

# ------------------------------------------------------------------
# GLORYS
# ------------------------------------------------------------------
print("\n[1] GLORYS FILE")
print("-" * 90)

ds = xr.open_dataset(GLORYS)

print("Path:", GLORYS)
print("Dimensions:", dict(ds.sizes))
print("Variables:", list(ds.data_vars))
print("Coordinates:", list(ds.coords))

# ------------------------------------------------------------------
# Depth
# ------------------------------------------------------------------
print("\n[2] DEPTH")
print("-" * 90)

depth_name = next(
    (x for x in ["depth", "deptht"] if x in ds.coords or x in ds.dims),
    None
)

if depth_name is None:
    print("ERROR: No 'depth' or 'deptht' coordinate found")
else:
    depths = ds[depth_name].values
    print("Coordinate:", depth_name)
    print("Number of levels:", len(depths))
    print("Available depths:")
    print(depths)

    print("\nRequested depth mapping:")
    for d in TARGET_DEPTHS:
        actual = depths[np.argmin(np.abs(depths - d))]
        print(f"  requested {d:>4} m -> actual {actual:.3f} m")

    print("\n1000 m:")
    actual = depths[np.argmin(np.abs(depths - 1000))]
    print(f"  nearest available = {actual:.3f} m")

# ------------------------------------------------------------------
# Time
# ------------------------------------------------------------------
print("\n[3] TIME")
print("-" * 90)

if "time" in ds:
    t = ds.time
    print("Count:", t.size)
    print("First:", t.values[0])
    print("Last :", t.values[-1])
else:
    print("ERROR: no time coordinate")

# ------------------------------------------------------------------
# Spatial grid
# ------------------------------------------------------------------
print("\n[4] HORIZONTAL GRID")
print("-" * 90)

for name in ["latitude", "longitude"]:
    if name in ds:
        c = ds[name]
        print(
            f"{name}: count={c.size}, "
            f"first={c.values[0]}, "
            f"last={c.values[-1]}"
        )

        if c.size > 1:
            print("  spacing:", float(c.values[1] - c.values[0]))
    else:
        print(f"ERROR: missing {name}")

# ------------------------------------------------------------------
# Phase 6 comparison
# ------------------------------------------------------------------
print("\n[5] PHASE 6 INPUT GRID")
print("-" * 90)

if INPUT.exists():
    inp = xr.open_dataset(INPUT)

    print("Path:", INPUT)
    print("Dimensions:", dict(inp.sizes))
    print("Variables:", list(inp.data_vars))
    print("Coordinates:", list(inp.coords))

    for name in ["time", "latitude", "longitude"]:
        if name in inp:
            c = inp[name]
            print(
                f"{name}: count={c.size}, "
                f"first={c.values[0]}, "
                f"last={c.values[-1]}"
            )
else:
    print("WARNING: Phase 6 normalized tensor not found:")
    print(INPUT)

# ------------------------------------------------------------------
# Mask
# ------------------------------------------------------------------
print("\n[6] OCEAN MASK")
print("-" * 90)

if MASK.exists():
    mask_ds = xr.open_dataset(MASK)

    print("Path:", MASK)
    print("Dimensions:", dict(mask_ds.sizes))
    print("Variables:", list(mask_ds.data_vars))

    mask_var = list(mask_ds.data_vars)[0]
    mask = mask_ds[mask_var]

    values = np.asarray(mask.values)

    print("Mask variable:", mask_var)
    print("Shape:", values.shape)
    print("Unique values:", np.unique(values))
    print("Ocean cells:", np.sum(values == 1))
    print("Land cells :", np.sum(values == 0))
else:
    print("WARNING: ocean mask not found:")
    print(MASK)

# ------------------------------------------------------------------
# Chunking / encoding
# ------------------------------------------------------------------
print("\n[7] GLORYS CHUNKING / ENCODING")
print("-" * 90)

for var in ds.data_vars:
    v = ds[var]
    print(f"\n{var}")
    print("  dims  :", v.dims)
    print("  shape :", v.shape)
    print("  chunks:", v.chunks)
    print("  dtype :", v.dtype)

# ------------------------------------------------------------------
# T/S sanity
# ------------------------------------------------------------------
print("\n[8] T/S VARIABLES")
print("-" * 90)

for candidate in ["thetao", "temperature", "temp"]:
    if candidate in ds:
        v = ds[candidate]
        print("Temperature variable:", candidate)
        print("  dims:", v.dims)
        print("  dtype:", v.dtype)

for candidate in ["so", "salinity", "salt"]:
    if candidate in ds:
        v = ds[candidate]
        print("Salinity variable:", candidate)
        print("  dims:", v.dims)
        print("  dtype:", v.dtype)

print("\n" + "=" * 90)
print("PREFLIGHT COMPLETE")
print("=" * 90)

ds.close()
