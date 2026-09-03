from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe


INPUT = Path("data/raw/glorys/glorys_deep_1062.nc")
OUTPUT = Path("data/processed/phase2/glorys_1062.nc")
WEIGHTS = Path("data/processed/phase2/weights/glorys_bilinear_weights.nc")


# Exact Phase 2 target grid
latitude = np.arange(5.0, 30.0 + 0.25 / 2, 0.25)
longitude = np.arange(45.0, 105.0 + 0.25 / 2, 0.25)

target_grid = xr.Dataset(
    coords={
        "latitude": latitude,
        "longitude": longitude,
    }
)

print("=" * 90)
print("OceanEmbed — Phase 2 Deep GLORYS Regrid")
print("=" * 90)

print("\nINPUT")
print(f"  {INPUT}")

ds = xr.open_dataset(INPUT)

print("  dimensions:", dict(ds.sizes))
print("  variables :", list(ds.data_vars))
print("  depth     :", ds.depth.values)

# Match the variable names used by processed Phase 2 GLORYS
ds = ds.rename({
    "thetao": "temperature",
    "so": "salinity",
})

print("\nTARGET GRID")
print(f"  latitude : {latitude.size}")
print(f"  longitude: {longitude.size}")
print(f"  shape    : ({latitude.size}, {longitude.size})")
print("  resolution: 0.25 degrees")

print("\nWEIGHTS")
print(f"  {WEIGHTS}")

if not WEIGHTS.exists():
    raise FileNotFoundError(f"Existing GLORYS weights not found: {WEIGHTS}")

# IMPORTANT:
# Reuse the exact Phase 2 GLORYS bilinear weights.
regridder = xe.Regridder(
    ds,
    target_grid,
    method="bilinear",
    filename=str(WEIGHTS),
    reuse_weights=True,
)

print("\nREGRIDDING")
print("  method: bilinear")
print("  using existing GLORYS weights")

result = regridder(ds, keep_attrs=True)

result.attrs.update(ds.attrs)
result.attrs["oceanembed_regrid_method"] = "bilinear"
result.attrs["oceanembed_grid_resolution"] = "0.25 degrees"

# Preserve the single depth coordinate.
result = result.assign_coords(depth=ds.depth.values)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("\nWRITING")
print(f"  {OUTPUT}")

result.to_netcdf(
    OUTPUT,
    format="NETCDF4",
)

ds.close()
result.close()

print("\nVERIFYING")

with xr.open_dataset(OUTPUT) as check:
    print("  dimensions:", dict(check.sizes))
    print("  variables :", list(check.data_vars))
    print("  depth     :", check.depth.values)
    print("  temperature:", check.temperature.shape)
    print("  salinity   :", check.salinity.shape)

    assert check.sizes["time"] == 1419
    assert check.sizes["depth"] == 1
    assert check.sizes["latitude"] == 101
    assert check.sizes["longitude"] == 241

print("\n" + "=" * 90)
print("PASS — GLORYS 1062 m REGRID COMPLETE")
print("=" * 90)
