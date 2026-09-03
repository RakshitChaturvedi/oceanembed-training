from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset, date2num


# =============================================================================
# Paths
# =============================================================================

MAIN_GLORYS = Path("data/processed/phase2/glorys.nc")
DEEP_GLORYS = Path("data/processed/phase2/glorys_1062.nc")
DEPTH_MAPPING = Path("data/processed/phase7/depth_mapping.json")

OUTPUT_DIR = Path("data/processed/phase7")

TEMPERATURE_OUTPUT = OUTPUT_DIR / "temperature_targets_raw.nc"
SALINITY_OUTPUT = OUTPUT_DIR / "salinity_targets_raw.nc"


# =============================================================================
# Model target depths
# =============================================================================

TARGET_DEPTHS = [
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
]


# =============================================================================
# Extraction configuration
# =============================================================================

# Number of daily timesteps processed in one computation.
#
# 16 days × 15 depths × 101 × 241 × 2 variables × float64
# is comfortably smaller than processing the entire 1419-day tensor.
TIME_CHUNK = 16


# =============================================================================
# Validation helpers
# =============================================================================

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
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
print("OceanEmbed — Phase 7 Target Extraction")
print("=" * 90)

print(
    """
Purpose:
  Extract Temperature and Salinity at the 15 model target depths.

Memory policy:
  - GLORYS is never loaded as a complete tensor.
  - No .load() is used on the source datasets.
  - No full-array .values is used on temperature/salinity.
  - Extraction is performed in time chunks.
  - 1000 m is sourced from the dedicated 1062.44 m GLORYS artifact.
"""
)


# =============================================================================
# [1] Required artifacts
# =============================================================================

print("\n" + "=" * 90)
print("[1] REQUIRED ARTIFACTS")
print("=" * 90)

check("MAIN GLORYS exists", MAIN_GLORYS.exists())
check("DEEP GLORYS exists", DEEP_GLORYS.exists())
check("DEPTH MAPPING exists", DEPTH_MAPPING.exists())

if failed:
    raise SystemExit(1)


# =============================================================================
# [2] Load depth mapping
# =============================================================================

print("\n" + "=" * 90)
print("[2] DEPTH MAPPING")
print("=" * 90)

with DEPTH_MAPPING.open("r", encoding="utf-8") as f:
    mapping_artifact = json.load(f)

check(
    "mapping artifact is dictionary",
    isinstance(mapping_artifact, dict),
)

if not isinstance(mapping_artifact, dict):
    raise RuntimeError("Invalid depth mapping artifact.")

mapping_entries = mapping_artifact.get("mapping")

check(
    "mapping entries are a list",
    isinstance(mapping_entries, list),
)

if not isinstance(mapping_entries, list):
    raise RuntimeError(
        "depth_mapping.json['mapping'] must be a list."
    )

check(
    "mapping contains 15 entries",
    len(mapping_entries) == 15,
    f"actual={len(mapping_entries)}",
)

check(
    "mapping targets match model contract",
    [
        entry["target_depth_m"]
        for entry in mapping_entries
    ] == TARGET_DEPTHS,
)

print("\nResolved mapping:")

for entry in mapping_entries:
    print(
        f"  target={entry['target_depth_m']:>4} m"
        f"  source={entry['source_depth_m']:>12.6f} m"
        f"  type={entry['source_type']}"
    )


# =============================================================================
# [3] Open source datasets
# =============================================================================

print("\n" + "=" * 90)
print("[3] OPEN SOURCE DATASETS")
print("=" * 90)

main = xr.open_dataset(MAIN_GLORYS)
deep = xr.open_dataset(DEEP_GLORYS)

print("MAIN")
print(f"  dimensions: {dict(main.sizes)}")

print("\nDEEP")
print(f"  dimensions: {dict(deep.sizes)}")

check(
    "MAIN temperature exists",
    "temperature" in main,
)

check(
    "MAIN salinity exists",
    "salinity" in main,
)

check(
    "DEEP temperature exists",
    "temperature" in deep,
)

check(
    "DEEP salinity exists",
    "salinity" in deep,
)


# =============================================================================
# [4] Source coordinate validation
# =============================================================================

print("\n" + "=" * 90)
print("[4] SOURCE COORDINATES")
print("=" * 90)

check(
    "MAIN time count",
    main.sizes["time"] == 1419,
)

check(
    "MAIN latitude count",
    main.sizes["latitude"] == 101,
)

check(
    "MAIN longitude count",
    main.sizes["longitude"] == 241,
)

check(
    "DEEP time count",
    deep.sizes["time"] == 1419,
)

check(
    "DEEP latitude count",
    deep.sizes["latitude"] == 101,
)

check(
    "DEEP longitude count",
    deep.sizes["longitude"] == 241,
)

check(
    "MAIN and DEEP time coordinates identical",
    np.array_equal(
        main["time"].values,
        deep["time"].values,
    ),
)

check(
    "MAIN and DEEP latitude coordinates identical",
    np.array_equal(
        main["latitude"].values,
        deep["latitude"].values,
    ),
)

check(
    "MAIN and DEEP longitude coordinates identical",
    np.array_equal(
        main["longitude"].values,
        deep["longitude"].values,
    ),
)


# =============================================================================
# [5] Construct target datasets
# =============================================================================

print("\n" + "=" * 90)
print("[5] TARGET CONSTRUCTION")
print("=" * 90)

target_depth_coord = xr.DataArray(
    TARGET_DEPTHS,
    dims="depth",
    name="depth",
)

temperature_layers = []
salinity_layers = []

source_depths = []
source_types = []
source_files = []


for entry in mapping_entries:

    target_depth = entry["target_depth_m"]
    source_depth = entry["source_depth_m"]
    source_type = entry["source_type"]

    print(
        f"\nTarget depth: {target_depth} m"
    )

    # -------------------------------------------------------------------------
    # Main GLORYS levels
    # -------------------------------------------------------------------------

    if source_type == "main_glorys":

        selected_t = (
            main["temperature"]
            .sel(
                depth=source_depth,
                method="nearest",
            )
            .drop_vars("depth")
        )

        selected_s = (
            main["salinity"]
            .sel(
                depth=source_depth,
                method="nearest",
            )
            .drop_vars("depth")
        )

        actual_depth_t = float(
            main["depth"]
            .sel(
                depth=source_depth,
                method="nearest",
            )
            .values
        )

        actual_depth_s = actual_depth_t

        source_file = str(MAIN_GLORYS)

    # -------------------------------------------------------------------------
    # Dedicated deep GLORYS level
    # -------------------------------------------------------------------------

    elif source_type == "deep_glorys":

        selected_t = (
            deep["temperature"]
            .isel(depth=0)
            .drop_vars("depth")
        )

        selected_s = (
            deep["salinity"]
            .isel(depth=0)
            .drop_vars("depth")
        )

        actual_depth_t = float(
            deep["depth"].values[0]
        )

        actual_depth_s = actual_depth_t

        source_file = str(DEEP_GLORYS)

    else:

        raise RuntimeError(
            f"Unknown source_type: {source_type}"
        )

    # -------------------------------------------------------------------------
    # Validate actual source depth
    # -------------------------------------------------------------------------

    check(
        f"{target_depth} m source depth",
        np.isclose(
            actual_depth_t,
            source_depth,
            atol=1e-4,
        ),
        (
            f"mapping={source_depth}, "
            f"actual={actual_depth_t}"
        ),
    )

    # -------------------------------------------------------------------------
    # Add model target depth dimension
    # -------------------------------------------------------------------------

    selected_t = selected_t.expand_dims(
        depth=[target_depth]
    )

    selected_s = selected_s.expand_dims(
        depth=[target_depth]
    )

    temperature_layers.append(selected_t)
    salinity_layers.append(selected_s)

    source_depths.append(actual_depth_t)
    source_types.append(source_type)
    source_files.append(source_file)

    print(
        f"  source depth : {actual_depth_t:.6f} m"
    )

    print(
        f"  source type  : {source_type}"
    )

    print(
        "  lazy layer constructed"
    )


# =============================================================================
# [6] Concatenate
# =============================================================================

print("\n" + "=" * 90)
print("[6] CONCATENATE")
print("=" * 90)

temperature_targets = xr.concat(
    temperature_layers,
    dim="depth",
)

salinity_targets = xr.concat(
    salinity_layers,
    dim="depth",
)

temperature_targets = temperature_targets.transpose(
    "time",
    "latitude",
    "longitude",
    "depth",
)

salinity_targets = salinity_targets.transpose(
    "time",
    "latitude",
    "longitude",
    "depth",
)

expected_shape = (
    1419,
    101,
    241,
    15,
)

print(
    f"Temperature shape: {temperature_targets.shape}"
)

print(
    f"Salinity shape   : {salinity_targets.shape}"
)

check(
    "temperature target shape",
    temperature_targets.shape == expected_shape,
)

check(
    "salinity target shape",
    salinity_targets.shape == expected_shape,
)

check(
    "temperature target depth coordinate",
    np.array_equal(
        temperature_targets["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "salinity target depth coordinate",
    np.array_equal(
        salinity_targets["depth"].values,
        TARGET_DEPTHS,
    ),
)


# =============================================================================
# [7] Metadata
# =============================================================================

print("\n" + "=" * 90)
print("[7] METADATA")
print("=" * 90)

common_attrs = {
    "oceanembed_phase": "7",
    "oceanembed_artifact": "GLORYS target preparation",
    "oceanembed_target_depths_m": json.dumps(
        TARGET_DEPTHS
    ),
    "oceanembed_source_depths_m": json.dumps(
        source_depths
    ),
    "oceanembed_source_types": json.dumps(
        source_types
    ),
    "oceanembed_source_files": json.dumps(
        source_files
    ),
    "oceanembed_depth_mapping": str(
        DEPTH_MAPPING
    ),
}

temperature_targets.attrs.update(common_attrs)
temperature_targets.attrs["target_variable"] = "temperature"

salinity_targets.attrs.update(common_attrs)
salinity_targets.attrs["target_variable"] = "salinity"


# =============================================================================
# [8] Remove old outputs
# =============================================================================

print("\n" + "=" * 90)
print("[8] OUTPUT PREPARATION")
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
# [9] TIME-CHUNKED EXTRACTION
# =============================================================================
print("\n" + "=" * 90)
print("[9] TIME-CHUNKED EXTRACTION")
print("=" * 90)

print(f"Time chunk size: {TIME_CHUNK} days")
print("Writing temperature and salinity targets...")

# -------------------------------------------------------------------------
# NetCDF4 direct indexed writing
#
# xarray mode="a" does NOT append new records along an unlimited dimension.
# It attempts to reconcile the incoming dataset's dimensions with the
# existing file. Therefore we create the final files with an unlimited
# time dimension and write each chunk explicitly into [start:end].
#
# Only the current TIME_CHUNK is materialized in memory.
# -------------------------------------------------------------------------
def create_output_file(path, variable_name):
    nc = Dataset(path, "w", format="NETCDF4")

    # =========================================================================
    # Dimensions
    # =========================================================================

    nc.createDimension("time", None)
    nc.createDimension("latitude", 101)
    nc.createDimension("longitude", 241)
    nc.createDimension("depth", 15)

    # =========================================================================
    # Time coordinate
    # =========================================================================

    time_source = main["time"]

    time_units = (
        time_source.encoding.get("units")
        or time_source.attrs.get(
            "units",
            "days since 1950-01-01 00:00:00"
        )
    )

    calendar = (
        time_source.encoding.get("calendar")
        or time_source.attrs.get(
            "calendar",
            "standard"
        )
    )

    time_var = nc.createVariable(
        "time",
        "f8",
        ("time",),
    )

    # main["time"].values is already datetime64.
    # Convert it to Python datetime objects before date2num.
    time_values = [
        value.astype("datetime64[us]").astype(object)
        for value in time_source.values
    ]

    time_numeric = date2num(
        time_values,
        units=time_units,
        calendar=calendar,
    )

    time_var[:] = np.asarray(
        time_numeric,
        dtype=np.float64,
    )

    # =========================================================================
    # Latitude / longitude / depth
    # =========================================================================

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

    latitude_var[:] = main["latitude"].values.astype(np.float32)
    longitude_var[:] = main["longitude"].values.astype(np.float32)
    depth_var[:] = np.asarray(
        TARGET_DEPTHS,
        dtype=np.float32,
    )

    # =========================================================================
    # Coordinate attributes
    # =========================================================================

    time_var.units = time_units
    time_var.calendar = calendar

    latitude_var.units = "degrees_north"
    longitude_var.units = "degrees_east"
    depth_var.units = "m"

    # =========================================================================
    # Data variable
    # =========================================================================

    data_var = nc.createVariable(
        variable_name,
        "f4",
        ("time", "latitude", "longitude", "depth"),
        zlib=True,
        complevel=4,
        chunksizes=(TIME_CHUNK, 101, 241, 15),
        fill_value=np.float32(np.nan),
    )

    # =========================================================================
    # Metadata
    # =========================================================================

    for key, value in common_attrs.items():
        nc.setncattr(key, value)

    nc.setncattr(
        "target_variable",
        variable_name,
    )

    return nc, data_var


temperature_nc, temperature_var = create_output_file(
    TEMPERATURE_OUTPUT,
    "temperature",
)

salinity_nc, salinity_var = create_output_file(
    SALINITY_OUTPUT,
    "salinity",
)

try:

    for start in range(
        0,
        expected_shape[0],
        TIME_CHUNK,
    ):

        end = min(
            start + TIME_CHUNK,
            expected_shape[0],
        )

        print(
            f"  Processing time [{start}:{end}]"
        )

        # -------------------------------------------------------------
        # Select only the current time chunk.
        # -------------------------------------------------------------
        t_chunk = temperature_targets.isel(
            time=slice(start, end)
        )

        s_chunk = salinity_targets.isel(
            time=slice(start, end)
        )

        # -------------------------------------------------------------
        # Materialize ONLY this chunk.
        # -------------------------------------------------------------
        t_data = t_chunk.values.astype(
            np.float32,
            copy=False,
        )

        s_data = s_chunk.values.astype(
            np.float32,
            copy=False,
        )

        # -------------------------------------------------------------
        # Explicit indexed write.
        #
        # This is the critical difference from xarray mode="a":
        # [start:end] tells NetCDF exactly where these records belong.
        # -------------------------------------------------------------
        temperature_var[start:end, :, :, :] = t_data
        salinity_var[start:end, :, :, :] = s_data

        # Flush data to disk periodically.
        temperature_nc.sync()
        salinity_nc.sync()

        # Release chunk memory.
        del t_chunk
        del s_chunk
        del t_data
        del s_data

finally:

    temperature_nc.close()
    salinity_nc.close()

print("  Time-chunked extraction complete.")
# =============================================================================
# [10] Output verification
# =============================================================================

print("\n" + "=" * 90)
print("[10] OUTPUT VERIFICATION")
print("=" * 90)

temperature_check = xr.open_dataset(
    TEMPERATURE_OUTPUT
)

salinity_check = xr.open_dataset(
    SALINITY_OUTPUT
)

print("\nTemperature output:")
print(
    f"  dimensions: {dict(temperature_check.sizes)}"
)

print(
    f"  variables : {list(temperature_check.data_vars)}"
)

print("\nSalinity output:")
print(
    f"  dimensions: {dict(salinity_check.sizes)}"
)

print(
    f"  variables : {list(salinity_check.data_vars)}"
)

check(
    "temperature output shape",
    temperature_check["temperature"].shape
    == expected_shape,
    (
        f"expected={expected_shape}, "
        f"actual={temperature_check['temperature'].shape}"
    ),
)

check(
    "salinity output shape",
    salinity_check["salinity"].shape
    == expected_shape,
    (
        f"expected={expected_shape}, "
        f"actual={salinity_check['salinity'].shape}"
    ),
)

check(
    "temperature output depths",
    np.array_equal(
        temperature_check["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "salinity output depths",
    np.array_equal(
        salinity_check["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "temperature output time",
    np.array_equal(
        temperature_check["time"].values,
        main["time"].values,
    ),
)

check(
    "salinity output time",
    np.array_equal(
        salinity_check["time"].values,
        main["time"].values,
    ),
)

check(
    "temperature output latitude",
    np.array_equal(
        temperature_check["latitude"].values,
        main["latitude"].values,
    ),
)

check(
    "temperature output longitude",
    np.array_equal(
        temperature_check["longitude"].values,
        main["longitude"].values,
    ),
)

check(
    "salinity output latitude",
    np.array_equal(
        salinity_check["latitude"].values,
        main["latitude"].values,
    ),
)

check(
    "salinity output longitude",
    np.array_equal(
        salinity_check["longitude"].values,
        main["longitude"].values,
    ),
)


# =============================================================================
# Cleanup
# =============================================================================

temperature_check.close()
salinity_check.close()

main.close()
deep.close()


# =============================================================================
# Final result
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 7 — PART 3 RESULT")
print("=" * 90)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:

    print(
        "\nPART 3 TARGET EXTRACTION: PASS\n"
        "Temperature and salinity targets were extracted "
        "for all 15 model depths."
    )

else:

    print(
        "\nPART 3 TARGET EXTRACTION: FAIL\n"
        "Do not proceed to target masking."
    )

print("=" * 90)

raise SystemExit(
    0 if failed == 0 else 1
)