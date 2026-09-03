from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset


# =============================================================================
# Paths
# =============================================================================

TEMPERATURE_INPUT = Path(
    "data/processed/phase7/temperature_targets_raw.nc"
)

SALINITY_INPUT = Path(
    "data/processed/phase7/salinity_targets_raw.nc"
)

OCEAN_MASK = Path(
    "data/processed/phase5/ocean_mask.nc"
)

OUTPUT_DIR = Path(
    "data/processed/phase7"
)

TEMPERATURE_OUTPUT = (
    OUTPUT_DIR / "temperature_targets_masked.nc"
)

SALINITY_OUTPUT = (
    OUTPUT_DIR / "salinity_targets_masked.nc"
)


# =============================================================================
# Expected contract
# =============================================================================

EXPECTED_TIME = 1419
EXPECTED_LATITUDE = 101
EXPECTED_LONGITUDE = 241
EXPECTED_DEPTH = 15

TARGET_DEPTHS = np.array(
    [
        0,
        5,
        10,
        20,
        30,
        50,
        75,
        100,
        125,
        150,
        200,
        300,
        500,
        700,
        1000,
    ],
    dtype=np.float32,
)


# =============================================================================
# Validation
# =============================================================================

passed = 0
failed = 0


def check(
    name: str,
    condition: bool,
    detail: str = "",
) -> None:
    global passed, failed

    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        failed += 1


# =============================================================================
# Header
# =============================================================================

print("=" * 90)
print("OceanEmbed — Phase 7 Target Masking")
print("=" * 90)

print(
    """
Purpose:
  Apply the canonical Phase 5 ocean mask to the Phase 7
  temperature and salinity target tensors.

Rules:
  - No interpolation.
  - No normalization.
  - No gap filling.
  - No modification of ocean values.
  - Land cells are forced to NaN.
  - Existing ocean NaNs remain unchanged.
  - Processing is performed in time chunks.
  - Full temperature/salinity tensors are never materialized.
"""
)


# =============================================================================
# [1] Required artifacts
# =============================================================================

print("\n" + "=" * 90)
print("[1] REQUIRED ARTIFACTS")
print("=" * 90)

check(
    "temperature raw artifact exists",
    TEMPERATURE_INPUT.exists(),
)

check(
    "salinity raw artifact exists",
    SALINITY_INPUT.exists(),
)

check(
    "ocean mask exists",
    OCEAN_MASK.exists(),
)

if failed:
    raise SystemExit(1)


# =============================================================================
# [2] Open datasets lazily
# =============================================================================

print("\n" + "=" * 90)
print("[2] OPEN INPUT ARTIFACTS")
print("=" * 90)

temperature = xr.open_dataset(TEMPERATURE_INPUT)
salinity = xr.open_dataset(SALINITY_INPUT)
mask_ds = xr.open_dataset(OCEAN_MASK)

print("\nTemperature:")
print(f"  dimensions: {dict(temperature.sizes)}")
print(f"  variables : {list(temperature.data_vars)}")

print("\nSalinity:")
print(f"  dimensions: {dict(salinity.sizes)}")
print(f"  variables : {list(salinity.data_vars)}")

print("\nMask:")
print(f"  dimensions: {dict(mask_ds.sizes)}")
print(f"  variables : {list(mask_ds.data_vars)}")


# =============================================================================
# [3] Input validation
# =============================================================================

print("\n" + "=" * 90)
print("[3] INPUT VALIDATION")
print("=" * 90)

check(
    "temperature variable exists",
    "temperature" in temperature,
)

check(
    "salinity variable exists",
    "salinity" in salinity,
)

check(
    "ocean_mask variable exists",
    "ocean_mask" in mask_ds,
)

if failed:
    temperature.close()
    salinity.close()
    mask_ds.close()
    raise SystemExit(1)


expected_shape = (
    EXPECTED_TIME,
    EXPECTED_LATITUDE,
    EXPECTED_LONGITUDE,
    EXPECTED_DEPTH,
)

check(
    "temperature shape",
    temperature["temperature"].shape == expected_shape,
    (
        f"expected={expected_shape}, "
        f"actual={temperature['temperature'].shape}"
    ),
)

check(
    "salinity shape",
    salinity["salinity"].shape == expected_shape,
    (
        f"expected={expected_shape}, "
        f"actual={salinity['salinity'].shape}"
    ),
)

check(
    "mask shape",
    mask_ds["ocean_mask"].shape
    == (EXPECTED_LATITUDE, EXPECTED_LONGITUDE),
    (
        f"expected={(EXPECTED_LATITUDE, EXPECTED_LONGITUDE)}, "
        f"actual={mask_ds['ocean_mask'].shape}"
    ),
)


# =============================================================================
# [4] Coordinate validation
# =============================================================================

print("\n" + "=" * 90)
print("[4] COORDINATE VALIDATION")
print("=" * 90)

temp = temperature["temperature"]
sal = salinity["salinity"]
ocean_mask = mask_ds["ocean_mask"]

check(
    "temperature dimensions",
    temp.dims == (
        "time",
        "latitude",
        "longitude",
        "depth",
    ),
)

check(
    "salinity dimensions",
    sal.dims == (
        "time",
        "latitude",
        "longitude",
        "depth",
    ),
)

check(
    "temperature depth coordinate",
    np.array_equal(
        temp["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "salinity depth coordinate",
    np.array_equal(
        sal["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "temperature/salinity time coordinates identical",
    np.array_equal(
        temp["time"].values,
        sal["time"].values,
    ),
)

check(
    "temperature/salinity latitude coordinates identical",
    np.array_equal(
        temp["latitude"].values,
        sal["latitude"].values,
    ),
)

check(
    "temperature/salinity longitude coordinates identical",
    np.array_equal(
        temp["longitude"].values,
        sal["longitude"].values,
    ),
)

check(
    "temperature ↔ mask latitude coordinates identical",
    np.array_equal(
        temp["latitude"].values,
        ocean_mask["latitude"].values,
    ),
)

check(
    "temperature ↔ mask longitude coordinates identical",
    np.array_equal(
        temp["longitude"].values,
        ocean_mask["longitude"].values,
    ),
)

check(
    "salinity ↔ mask latitude coordinates identical",
    np.array_equal(
        sal["latitude"].values,
        ocean_mask["latitude"].values,
    ),
)

check(
    "salinity ↔ mask longitude coordinates identical",
    np.array_equal(
        sal["longitude"].values,
        ocean_mask["longitude"].values,
    ),
)


# =============================================================================
# [5] Mask validation
# =============================================================================

print("\n" + "=" * 90)
print("[5] OCEAN MASK VALIDATION")
print("=" * 90)

mask_values = ocean_mask.values

unique_mask_values = np.unique(mask_values)

print(f"Unique values: {unique_mask_values}")

check(
    "mask is boolean",
    mask_values.dtype == bool,
    f"dtype={mask_values.dtype}",
)

check(
    "mask contains no NaN",
    not np.any(np.isnan(mask_values.astype(np.float32))),
)

ocean_cells = int(mask_values.sum())
land_cells = int((~mask_values).sum())

print(f"Ocean cells: {ocean_cells}")
print(f"Land cells : {land_cells}")

check(
    "mask ocean cell count",
    ocean_cells == 12490,
    f"actual={ocean_cells}",
)

check(
    "mask land cell count",
    land_cells == 11851,
    f"actual={land_cells}",
)


# =============================================================================
# [6] Prepare outputs
# =============================================================================

print("\n" + "=" * 90)
print("[6] OUTPUT PREPARATION")
print("=" * 90)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

if TEMPERATURE_OUTPUT.exists():
    print(f"Removing existing {TEMPERATURE_OUTPUT}")
    TEMPERATURE_OUTPUT.unlink()

if SALINITY_OUTPUT.exists():
    print(f"Removing existing {SALINITY_OUTPUT}")
    SALINITY_OUTPUT.unlink()


# =============================================================================
# [7] Time-chunked masking
# =============================================================================

print("\n" + "=" * 90)
print("[7] TIME-CHUNKED MASKING")
print("=" * 90)

TIME_CHUNK = 16

print(f"Time chunk size: {TIME_CHUNK} days")
print("Applying ocean mask...")

# -------------------------------------------------------------------------
# Output paths
# -------------------------------------------------------------------------

TEMPERATURE_OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

# Remove previous outputs so this run always starts cleanly.
if TEMPERATURE_OUTPUT.exists():
    print(f"  Removing existing {TEMPERATURE_OUTPUT}")
    TEMPERATURE_OUTPUT.unlink()

if SALINITY_OUTPUT.exists():
    print(f"  Removing existing {SALINITY_OUTPUT}")
    SALINITY_OUTPUT.unlink()


# -------------------------------------------------------------------------
# Prepare the 2-D ocean mask
# -------------------------------------------------------------------------
#
# ocean_mask dimensions:
#   latitude × longitude
#
# The target data dimensions are:
#   time × latitude × longitude × depth
#
# We do NOT create a materialized 4-D mask.
# The mask is selected/broadcast lazily for each time chunk.
#

mask_2d = ocean_mask


# -------------------------------------------------------------------------
# Preserve metadata
# -------------------------------------------------------------------------

temperature_attrs = dict(temperature.attrs)

salinity_attrs = dict(salinity.attrs)

temperature_attrs.update(
    {
        "oceanembed_phase": "7",
        "oceanembed_artifact": "masked temperature targets",
        "oceanembed_mask": str(OCEAN_MASK),
        "oceanembed_mask_rule": (
            "land cells forced to NaN; ocean values preserved"
        ),
    }
)

salinity_attrs.update(
    {
        "oceanembed_phase": "7",
        "oceanembed_artifact": "masked salinity targets",
        "oceanembed_mask": str(OCEAN_MASK),
        "oceanembed_mask_rule": (
            "land cells forced to NaN; ocean values preserved"
        ),
    }
)


# -------------------------------------------------------------------------
# Coordinate encoding
# -------------------------------------------------------------------------
#
# xarray has decoded the source time coordinate into datetime64.
# netCDF4 requires a numeric time variable, so encode it back into
# CF-compliant numeric time.
#

time_source = temperature["time"]

time_units = time_source.encoding.get(
    "units",
    time_source.attrs.get(
        "units",
        "days since 1950-01-01 00:00:00",
    ),
)

time_calendar = time_source.encoding.get(
    "calendar",
    time_source.attrs.get(
        "calendar",
        "standard",
    ),
)

encoded_time_result = xr.coding.times.encode_cf_datetime(
    time_source.values,
    units=time_units,
    calendar=time_calendar,
)

encoded_time = np.asarray(
    encoded_time_result[0],
    dtype=np.float64,
)


latitude_values = np.asarray(
    temperature["latitude"].values,
    dtype=np.float32,
)

longitude_values = np.asarray(
    temperature["longitude"].values,
    dtype=np.float32,
)

depth_values = np.asarray(
    TARGET_DEPTHS,
    dtype=np.float32,
)


# -------------------------------------------------------------------------
# Create output files
# -------------------------------------------------------------------------
#
# The time dimension is explicitly unlimited.
# We write each chunk directly into the correct slice.
#

temperature_nc = Dataset(
    TEMPERATURE_OUTPUT,
    "w",
    format="NETCDF4",
)

salinity_nc = Dataset(
    SALINITY_OUTPUT,
    "w",
    format="NETCDF4",
)


# -------------------------------------------------------------------------
# Create dimensions
# -------------------------------------------------------------------------

for nc in (temperature_nc, salinity_nc):

    nc.createDimension("time", None)
    nc.createDimension("latitude", EXPECTED_LATITUDE)
    nc.createDimension("longitude", EXPECTED_LONGITUDE)
    nc.createDimension("depth", EXPECTED_DEPTH)


# -------------------------------------------------------------------------
# Create coordinate variables
# -------------------------------------------------------------------------

for nc in (temperature_nc, salinity_nc):

    time_var = nc.createVariable(
        "time",
        "f8",
        ("time",),
    )

    latitude_var = nc.createVariable(
        "latitude",
        "f4",
        ("latitude",),
    )

    longitude_var = nc.createVariable(
        "longitude",
        "f4",
        ("longitude",),
    )

    depth_var = nc.createVariable(
        "depth",
        "f4",
        ("depth",),
    )

    time_var[:] = encoded_time
    latitude_var[:] = latitude_values
    longitude_var[:] = longitude_values
    depth_var[:] = depth_values

    time_var.units = time_units
    time_var.calendar = time_calendar

    latitude_var.units = (
        temperature["latitude"].attrs.get("units", "degrees_north")
    )

    longitude_var.units = (
        temperature["longitude"].attrs.get("units", "degrees_east")
    )

    depth_var.units = "m"


# -------------------------------------------------------------------------
# Create data variables
# -------------------------------------------------------------------------

temperature_var = temperature_nc.createVariable(
    "temperature",
    "f4",
    (
        "time",
        "latitude",
        "longitude",
        "depth",
    ),
    zlib=True,
    complevel=4,
    chunksizes=(
        TIME_CHUNK,
        EXPECTED_LATITUDE,
        EXPECTED_LONGITUDE,
        EXPECTED_DEPTH,
    ),
    fill_value=np.float32(np.nan),
)

salinity_var = salinity_nc.createVariable(
    "salinity",
    "f4",
    (
        "time",
        "latitude",
        "longitude",
        "depth",
    ),
    zlib=True,
    complevel=4,
    chunksizes=(
        TIME_CHUNK,
        EXPECTED_LATITUDE,
        EXPECTED_LONGITUDE,
        EXPECTED_DEPTH,
    ),
    fill_value=np.float32(np.nan),
)


# -------------------------------------------------------------------------
# Dataset metadata
# -------------------------------------------------------------------------

for key, value in temperature_attrs.items():
    temperature_nc.setncattr(key, value)

for key, value in salinity_attrs.items():
    salinity_nc.setncattr(key, value)


# -------------------------------------------------------------------------
# Time-chunked masking
# -------------------------------------------------------------------------

for start in range(
    0,
    EXPECTED_TIME,
    TIME_CHUNK,
):

    end = min(
        start + TIME_CHUNK,
        EXPECTED_TIME,
    )

    print(
        f"  Processing time [{start}:{end}]"
    )
    # ---------------------------------------------------------------------
    # Select ONLY this time chunk and the actual data variables.
    # ---------------------------------------------------------------------

    temp_chunk = (
        temperature["temperature"]
        .isel(time=slice(start, end))
    )

    sal_chunk = (
        salinity["salinity"]
        .isel(time=slice(start, end))
    )

    # ---------------------------------------------------------------------
    # Apply the 2-D ocean mask.
    #
    # True  -> preserve ocean value
    # False -> NaN
    #
    # Broadcasting remains lazy.
    # ---------------------------------------------------------------------

    temp_chunk = (
        temp_chunk
        .where(mask_2d)
        .astype(np.float32)
    )

    sal_chunk = (
        sal_chunk
        .where(mask_2d)
        .astype(np.float32)
    )

    print("  temp_chunk type :", type(temp_chunk))
    print("  sal_chunk type  :", type(sal_chunk))
    print("  temp_chunk dims :", temp_chunk.dims)
    print("  temp_chunk shape:", temp_chunk.shape)

    # ---------------------------------------------------------------------
    # Materialize ONLY this 16-day chunk.
    # ---------------------------------------------------------------------

    temp_data = temp_chunk.to_numpy().astype(
        np.float32,
        copy=False,
    )

    sal_data = sal_chunk.to_numpy().astype(
        np.float32,
        copy=False,
    )

    # ---------------------------------------------------------------------
    # Write directly into the corresponding time slice.
    # ---------------------------------------------------------------------

    temperature_var[start:end, :, :, :] = temp_data
    salinity_var[start:end, :, :, :] = sal_data

    del temp_chunk
    del sal_chunk
    del temp_data
    del sal_data

# -------------------------------------------------------------------------
# Flush and close
# -------------------------------------------------------------------------

temperature_nc.sync()
salinity_nc.sync()

temperature_nc.close()
salinity_nc.close()

print("  Time-chunked masking complete.")
# =============================================================================
# [8] Output verification
# =============================================================================

print("\n" + "=" * 90)
print("[8] OUTPUT VERIFICATION")
print("=" * 90)

temperature_out = xr.open_dataset(
    TEMPERATURE_OUTPUT
)

salinity_out = xr.open_dataset(
    SALINITY_OUTPUT
)

print("\nTemperature output:")
print(
    f"  dimensions: "
    f"{dict(temperature_out.sizes)}"
)

print(
    f"  variables : "
    f"{list(temperature_out.data_vars)}"
)

print("\nSalinity output:")
print(
    f"  dimensions: "
    f"{dict(salinity_out.sizes)}"
)

print(
    f"  variables : "
    f"{list(salinity_out.data_vars)}"
)

check(
    "temperature output shape",
    temperature_out["temperature"].shape
    == expected_shape,
)

check(
    "salinity output shape",
    salinity_out["salinity"].shape
    == expected_shape,
)

check(
    "temperature output depth coordinate",
    np.array_equal(
        temperature_out["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "salinity output depth coordinate",
    np.array_equal(
        salinity_out["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "temperature output time coordinate",
    np.array_equal(
        temperature_out["time"].values,
        temp["time"].values,
    ),
)

check(
    "salinity output time coordinate",
    np.array_equal(
        salinity_out["time"].values,
        sal["time"].values,
    ),
)

check(
    "temperature output latitude coordinate",
    np.array_equal(
        temperature_out["latitude"].values,
        temp["latitude"].values,
    ),
)

check(
    "temperature output longitude coordinate",
    np.array_equal(
        temperature_out["longitude"].values,
        temp["longitude"].values,
    ),
)

check(
    "salinity output latitude coordinate",
    np.array_equal(
        salinity_out["latitude"].values,
        sal["latitude"].values,
    ),
)

check(
    "salinity output longitude coordinate",
    np.array_equal(
        salinity_out["longitude"].values,
        sal["longitude"].values,
    ),
)


# =============================================================================
# [9] Mask correctness spot-check
# =============================================================================

print("\n" + "=" * 90)
print("[9] MASK CORRECTNESS CHECK")
print("=" * 90)

# Materialize ONLY the 2-D output slices needed for validation.
#
# We inspect the first timestep and first depth. This is intentionally
# small and does not load the full output tensors.

temp_sample = (
    temperature_out["temperature"]
    .isel(time=0, depth=0)
    .values
)

sal_sample = (
    salinity_out["salinity"]
    .isel(time=0, depth=0)
    .values
)

raw_temp_sample = (
    temp
    .isel(time=0, depth=0)
    .values
)

raw_sal_sample = (
    sal
    .isel(time=0, depth=0)
    .values
)

# -------------------------------------------------------------------------
# Land cells must be NaN.
# -------------------------------------------------------------------------

land_temp = temp_sample[~mask_values]
land_sal = sal_sample[~mask_values]

check(
    "temperature land cells are NaN",
    np.all(np.isnan(land_temp)),
    f"non_nan={np.count_nonzero(~np.isnan(land_temp))}",
)

check(
    "salinity land cells are NaN",
    np.all(np.isnan(land_sal)),
    f"non_nan={np.count_nonzero(~np.isnan(land_sal))}",
)


# -------------------------------------------------------------------------
# Ocean cells must preserve the raw values.
#
# Only compare cells that were finite in the raw input.
# -------------------------------------------------------------------------

ocean_temp_raw = raw_temp_sample[mask_values]
ocean_temp_masked = temp_sample[mask_values]

ocean_sal_raw = raw_sal_sample[mask_values]
ocean_sal_masked = sal_sample[mask_values]

temp_finite = np.isfinite(ocean_temp_raw)
sal_finite = np.isfinite(ocean_sal_raw)

check(
    "temperature ocean finite values preserved",
    np.allclose(
        ocean_temp_masked[temp_finite],
        ocean_temp_raw[temp_finite],
        equal_nan=False,
    ),
)

check(
    "salinity ocean finite values preserved",
    np.allclose(
        ocean_sal_masked[sal_finite],
        ocean_sal_raw[sal_finite],
        equal_nan=False,
    ),
)


# =============================================================================
# [10] NaN behavior
# =============================================================================

print("\n" + "=" * 90)
print("[10] NaN BEHAVIOR")
print("=" * 90)

# Count the first timestep/depth only.
#
# These checks establish that masking does not manufacture finite values.

raw_temp_nan = int(np.isnan(raw_temp_sample).sum())
masked_temp_nan = int(np.isnan(temp_sample).sum())

raw_sal_nan = int(np.isnan(raw_sal_sample).sum())
masked_sal_nan = int(np.isnan(sal_sample).sum())

print(
    f"Temperature NaN count:"
    f" raw={raw_temp_nan},"
    f" masked={masked_temp_nan}"
)

print(
    f"Salinity NaN count:"
    f" raw={raw_sal_nan},"
    f" masked={masked_sal_nan}"
)

check(
    "temperature masked NaNs >= raw NaNs",
    masked_temp_nan >= raw_temp_nan,
)

check(
    "salinity masked NaNs >= raw NaNs",
    masked_sal_nan >= raw_sal_nan,
)


# =============================================================================
# [11] Metadata
# =============================================================================

print("\n" + "=" * 90)
print("[11] METADATA")
print("=" * 90)

check(
    "temperature mask metadata",
    temperature_out.attrs.get("oceanembed_mask")
    == str(OCEAN_MASK),
)

check(
    "salinity mask metadata",
    salinity_out.attrs.get("oceanembed_mask")
    == str(OCEAN_MASK),
)

check(
    "temperature mask rule metadata",
    "land cells forced to NaN"
    in temperature_out.attrs.get(
        "oceanembed_mask_rule",
        "",
    ),
)

check(
    "salinity mask rule metadata",
    "land cells forced to NaN"
    in salinity_out.attrs.get(
        "oceanembed_mask_rule",
        "",
    ),
)


# =============================================================================
# Cleanup
# =============================================================================

temperature_out.close()
salinity_out.close()

temperature.close()
salinity.close()
mask_ds.close()


# =============================================================================
# Final result
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 7 — PART 4 RESULT")
print("=" * 90)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:
    print(
        """
PART 4 TARGET MASKING: PASS

The canonical ocean mask was applied successfully.
Land cells are NaN and ocean values are preserved.
"""
    )
else:
    print(
        """
PART 4 TARGET MASKING: FAIL

Do not proceed to the next Phase 7 stage.
"""
    )

print("=" * 90)

raise SystemExit(
    0 if failed == 0 else 1
)