from pathlib import Path

import numpy as np
import xarray as xr


path = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training/"
    "data/processed/phase2/sst.nc"
)

backup_path = path.with_name("sst_before_zero_fix.nc")
temp_path = path.with_name("sst_fixed_tmp.nc")


print(f"Opening: {path}")

ds = xr.open_dataset(path)

sst = ds["sst"]

zero_count = int((sst == 0.0).sum().values)

print(f"Zero values found: {zero_count:,}")

if zero_count == 0:
    print("No zero values found. Nothing to change.")
    ds.close()
    raise SystemExit(0)

# ------------------------------------------------------------------
# Backup original
# ------------------------------------------------------------------

print(f"Creating backup: {backup_path}")

ds.to_netcdf(
    backup_path,
    mode="w",
)

# ------------------------------------------------------------------
# Replace physically invalid 0 K values with NaN
# ------------------------------------------------------------------

print("Replacing 0 K values with NaN...")

ds["sst"] = sst.where(sst != 0.0)

# ------------------------------------------------------------------
# Record what we did
# ------------------------------------------------------------------

ds.attrs["oceanembed_sst_zero_fix"] = (
    "Replaced physically invalid 0 K fill/mask values with NaN"
)

ds.attrs["oceanembed_sst_zero_values_replaced"] = zero_count

# ------------------------------------------------------------------
# Write corrected dataset
# ------------------------------------------------------------------

print(f"Writing corrected dataset: {temp_path}")

ds.to_netcdf(
    temp_path,
    mode="w",
)

ds.close()

# ------------------------------------------------------------------
# Atomically replace original
# ------------------------------------------------------------------

print("Replacing original sst.nc...")

temp_path.replace(path)

print("Done.")