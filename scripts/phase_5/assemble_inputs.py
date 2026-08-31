#!/usr/bin/env python3

"""
OceanEmbed — Phase 5 Input Tensor Assembly

Assembles the validated Phase-5 / Phase-2 / Phase-4 fields into the
12-channel model input tensor.

READ ONLY INPUTS
----------------
Phase 5:
    sst_filled.nc
    sss_filled.nc
    ssh_filled.nc
    current_u_filled.nc
    current_v_filled.nc
    ocean_mask.nc
    missingness_masks.nc

Phase 2:
    wind.nc

Phase 4:
    wind_stress_curl.nc

Output:
    data/processed/phase5/input_tensor.nc

Channel order:
    0  SST
    1  SSS
    2  SSHA
    3  wind_U
    4  wind_V
    5  current_U
    6  current_V
    7  wind_stress_curl
    8  latitude
    9  longitude
   10  sin(day_of_year)
   11  cos(day_of_year)

No normalization is performed here.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import xarray as xr


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

PHASE2_DIR = ROOT / "data" / "processed" / "phase2"
PHASE4_DIR = ROOT / "data" / "processed" / "phase4"
PHASE5_DIR = ROOT / "data" / "processed" / "phase5"

OUTPUT_FILE = PHASE5_DIR / "input_tensor.nc"


# =============================================================================
# EXPECTED GRID
# =============================================================================

EXPECTED_TIME = 1419
EXPECTED_LAT = 101
EXPECTED_LON = 241

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0
EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0
EXPECTED_RESOLUTION = 0.25


# =============================================================================
# CHANNEL SCHEMA
# =============================================================================

CHANNELS = [
    "SST",
    "SSS",
    "SSHA",
    "wind_U",
    "wind_V",
    "current_U",
    "current_V",
    "wind_stress_curl",
    "latitude",
    "longitude",
    "sin_day_of_year",
    "cos_day_of_year",
]


# =============================================================================
# FILE / VARIABLE CONFIGURATION
# =============================================================================

FIELD_CONFIG = {
    "SST": {
        "file": PHASE5_DIR / "sst_filled.nc",
        "variable": "sst",
    },
    "SSS": {
        "file": PHASE5_DIR / "sss_filled.nc",
        "variable": "sss",
    },
    "SSHA": {
        "file": PHASE5_DIR / "ssh_filled.nc",
        "variable": "ssh",
    },
    "wind_U": {
        "file": PHASE2_DIR / "wind.nc",
        "variable": "wind_u",
    },
    "wind_V": {
        "file": PHASE2_DIR / "wind.nc",
        "variable": "wind_v",
    },
    "current_U": {
        "file": PHASE5_DIR / "current_u_filled.nc",
        "variable": "current_u",
    },
    "current_V": {
        "file": PHASE5_DIR / "current_v_filled.nc",
        "variable": "current_v",
    },
    "wind_stress_curl": {
        "file": PHASE5_DIR / "wind_stress_curl_filled.nc",
        "variable": "wind_stress_curl",
    },
}


# =============================================================================
# LOGGING
# =============================================================================

def header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def fail(message: str) -> None:
    print()
    print(f"FATAL ERROR: {message}")
    sys.exit(1)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_dimensions(da: xr.DataArray, name: str) -> None:
    expected_dims = ("time", "latitude", "longitude")

    if da.dims != expected_dims:
        fail(
            f"{name}: unexpected dimensions.\n"
            f"  expected: {expected_dims}\n"
            f"  found:    {da.dims}"
        )

    expected_shape = (
        EXPECTED_TIME,
        EXPECTED_LAT,
        EXPECTED_LON,
    )

    if da.shape != expected_shape:
        fail(
            f"{name}: unexpected shape.\n"
            f"  expected: {expected_shape}\n"
            f"  found:    {da.shape}"
        )


def validate_grid(da: xr.DataArray, name: str) -> None:
    lat = da["latitude"].values
    lon = da["longitude"].values

    if not np.all(np.isfinite(lat)):
        fail(f"{name}: latitude contains non-finite values.")

    if not np.all(np.isfinite(lon)):
        fail(f"{name}: longitude contains non-finite values.")

    if not np.isclose(lat[0], EXPECTED_LAT_MIN):
        fail(
            f"{name}: latitude minimum mismatch: "
            f"{lat[0]} != {EXPECTED_LAT_MIN}"
        )

    if not np.isclose(lat[-1], EXPECTED_LAT_MAX):
        fail(
            f"{name}: latitude maximum mismatch: "
            f"{lat[-1]} != {EXPECTED_LAT_MAX}"
        )

    if not np.isclose(lon[0], EXPECTED_LON_MIN):
        fail(
            f"{name}: longitude minimum mismatch: "
            f"{lon[0]} != {EXPECTED_LON_MIN}"
        )

    if not np.isclose(lon[-1], EXPECTED_LON_MAX):
        fail(
            f"{name}: longitude maximum mismatch: "
            f"{lon[-1]} != {EXPECTED_LON_MAX}"
        )

    lat_diff = np.diff(lat)
    lon_diff = np.diff(lon)

    if not np.allclose(lat_diff, EXPECTED_RESOLUTION):
        fail(f"{name}: latitude spacing is not 0.25 degrees.")

    if not np.allclose(lon_diff, EXPECTED_RESOLUTION):
        fail(f"{name}: longitude spacing is not 0.25 degrees.")


def validate_coordinates(
    reference: xr.DataArray,
    candidate: xr.DataArray,
    name: str,
) -> None:

    for coord in ("time", "latitude", "longitude"):
        if coord not in candidate.coords:
            fail(f"{name}: missing coordinate '{coord}'.")

        if coord not in reference.coords:
            fail(f"Reference dataset missing coordinate '{coord}'.")

        ref = reference[coord].values
        got = candidate[coord].values

        if ref.shape != got.shape:
            fail(
                f"{name}: coordinate '{coord}' shape mismatch.\n"
                f"  reference: {ref.shape}\n"
                f"  candidate: {got.shape}"
            )

        if np.issubdtype(ref.dtype, np.datetime64):
            equal = np.array_equal(ref, got)
        else:
            equal = np.allclose(ref, got, equal_nan=True)

        if not equal:
            fail(
                f"{name}: coordinate '{coord}' does not exactly "
                f"match the reference grid."
            )


def validate_values(da: xr.DataArray, name: str) -> None:
    print(f"  Checking values: {name}")

    values = da.values

    nan_count = int(np.isnan(values).sum())
    inf_count = int(np.isinf(values).sum())

    if nan_count:
        fail(
            f"{name}: tensor contains {nan_count:,} NaN values."
        )

    if inf_count:
        fail(
            f"{name}: tensor contains {inf_count:,} infinite values."
        )

    print("    NaNs : 0")
    print("    Infs : 0")


# =============================================================================
# LOAD FIELD
# =============================================================================

def load_field(
    name: str,
    file_path: Path,
    variable: str,
) -> xr.DataArray:

    print(f"\n{name}")
    print(f"  File     : {file_path}")
    print(f"  Variable : {variable}")

    if not file_path.exists():
        fail(f"Input file does not exist: {file_path}")

    ds = xr.open_dataset(file_path)

    if variable not in ds:
        ds.close()
        fail(
            f"Variable '{variable}' not found in {file_path}"
        )

    da = ds[variable].load()

    # Detach from the file before closing it.
    da = da.copy()

    ds.close()

    expected_dims = ("time", "latitude", "longitude")
    if set(da.dims) != set(expected_dims):
        fail(
        f"{name}: unexpected dimensions.\n"
        f"  expected dimensions: {expected_dims}\n"
        f"  found:              {da.dims}"
        )

    da = da.transpose(*expected_dims)
    validate_dimensions(da, name)
    validate_grid(da, name)

    print(f"  Shape    : {da.shape}")
    print(f"  Units    : {da.attrs.get('units', 'unknown')}")

    return da


# =============================================================================
# TEMPORAL FEATURES
# =============================================================================

def build_temporal_features(
    time: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:

    print("\nBuilding temporal features...")

    if not np.issubdtype(time.dtype, np.datetime64):
        fail(
            "Time coordinate is not datetime64; cannot construct "
            "day-of-year features."
        )

    day_of_year = time.dt.dayofyear

    # Use 365.25 so the encoding remains continuous across leap years.
    angle = 2.0 * np.pi * (day_of_year - 1) / 365.25

    sin_time = xr.DataArray(
        np.sin(angle.values).astype(np.float32),
        dims=("time",),
        coords={"time": time},
        name="sin_day_of_year",
    )

    cos_time = xr.DataArray(
        np.cos(angle.values).astype(np.float32),
        dims=("time",),
        coords={"time": time},
        name="cos_day_of_year",
    )

    print("  sin_day_of_year : OK")
    print("  cos_day_of_year : OK")

    return sin_time, cos_time


# =============================================================================
# SPATIAL FEATURES
# =============================================================================

def build_spatial_features(
    reference: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray]:

    print("\nBuilding spatial coordinate features...")

    lat = reference["latitude"]
    lon = reference["longitude"]

    latitude = xr.DataArray(
        lat.values.astype(np.float32),
        dims=("latitude",),
        coords={"latitude": lat},
        name="latitude",
    )

    longitude = xr.DataArray(
        lon.values.astype(np.float32),
        dims=("longitude",),
        coords={"longitude": lon},
        name="longitude",
    )

    print("  latitude  : OK")
    print("  longitude : OK")

    return latitude, longitude


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    header("OceanEmbed — Phase 5 Input Tensor Assembly")

    print("READ ONLY INPUTS")
    print("No source dataset will be modified.")
    print("No normalization is performed.")
    print()

    # -------------------------------------------------------------------------
    # Load first field as reference
    # -------------------------------------------------------------------------

    header("LOADING REFERENCE FIELD")

    first_name = "SST"

    cfg = FIELD_CONFIG[first_name]

    reference = load_field(
        first_name,
        cfg["file"],
        cfg["variable"],
    )

    # -------------------------------------------------------------------------
    # Load remaining fields
    # -------------------------------------------------------------------------

    header("LOADING MODEL FIELDS")

    fields: dict[str, xr.DataArray] = {
        first_name: reference,
    }

    for name, cfg in FIELD_CONFIG.items():

        if name == first_name:
            continue

        da = load_field(
            name,
            cfg["file"],
            cfg["variable"],
        )

        validate_coordinates(
            reference,
            da,
            name,
        )

        print("  Coordinates: MATCH")

        fields[name] = da

    # -------------------------------------------------------------------------
    # Validate all source values
    # -------------------------------------------------------------------------

    header("VALIDATING SOURCE VALUES")

    for name in CHANNELS[:8]:
        validate_values(
            fields[name],
            name,
        )

    # -------------------------------------------------------------------------
    # Build coordinate/time channels
    # -------------------------------------------------------------------------

    header("BUILDING AUXILIARY CHANNELS")

    latitude, longitude = build_spatial_features(reference)

    sin_time, cos_time = build_temporal_features(
        reference["time"]
    )

    # -------------------------------------------------------------------------
    # Broadcast auxiliary channels
    # -------------------------------------------------------------------------

    print("\nBroadcasting auxiliary features to full grid...")

    lat_channel = (
        latitude
        .broadcast_like(reference)
        .transpose("time", "latitude", "longitude")
        .astype(np.float32)
    )

    lon_channel = (
        longitude
        .broadcast_like(reference)
        .transpose("time", "latitude", "longitude")
        .astype(np.float32)
    )

    sin_channel = (
        sin_time
        .broadcast_like(reference)
        .transpose("time", "latitude", "longitude")
        .astype(np.float32)
    )

    cos_channel = (
        cos_time
        .broadcast_like(reference)
        .transpose("time", "latitude", "longitude")
        .astype(np.float32)
    )

    print("  latitude           : OK")
    print("  longitude          : OK")
    print("  sin_day_of_year    : OK")
    print("  cos_day_of_year    : OK")

    # -------------------------------------------------------------------------
    # Assemble channel list
    # -------------------------------------------------------------------------

    header("ASSEMBLING 12-CHANNEL INPUT")

    channel_arrays = [
        fields["SST"],
        fields["SSS"],
        fields["SSHA"],
        fields["wind_U"],
        fields["wind_V"],
        fields["current_U"],
        fields["current_V"],
        fields["wind_stress_curl"],
        lat_channel,
        lon_channel,
        sin_channel,
        cos_channel,
    ]

    for index, (name, da) in enumerate(
        zip(CHANNELS, channel_arrays)
    ):
        print(f"  [{index:02d}] {name}")

        validate_dimensions(
            da,
            name,
        )

        validate_coordinates(
            reference,
            da,
            name,
        )

    # -------------------------------------------------------------------------
    # Stack
    # -------------------------------------------------------------------------

    print("\nStacking channels...")

    tensor = xr.concat(
        channel_arrays,
        dim=xr.IndexVariable(
            "channel",
            np.arange(len(CHANNELS)),
        ),
    )

    tensor = tensor.transpose(
        "time",
        "latitude",
        "longitude",
        "channel",
    )

    tensor = tensor.astype(np.float32)

    tensor.name = "input_tensor"

    tensor.attrs.update(
        {
            "long_name": "OceanEmbed model input tensor",
            "oceanembed_phase": "phase5",
            "oceanembed_schema_version": "phase5-input-v1",
            "oceanembed_channel_order": ",".join(CHANNELS),
            "oceanembed_tensor_shape": (
                f"{EXPECTED_TIME},{EXPECTED_LAT},"
                f"{EXPECTED_LON},{len(CHANNELS)}"
            ),
            "oceanembed_normalized": "false",
        }
    )

    # -------------------------------------------------------------------------
    # Validate tensor
    # -------------------------------------------------------------------------

    header("FINAL TENSOR VALIDATION")

    expected_shape = (
        EXPECTED_TIME,
        EXPECTED_LAT,
        EXPECTED_LON,
        len(CHANNELS),
    )

    print(f"Expected shape : {expected_shape}")
    print(f"Actual shape   : {tensor.shape}")

    if tensor.shape != expected_shape:
        fail(
            f"Final tensor shape mismatch: "
            f"{tensor.shape} != {expected_shape}"
        )

    validate_values(
        tensor,
        "input_tensor",
    )

    if list(tensor["channel"].values) != list(
        range(len(CHANNELS))
    ):
        fail("Channel coordinate is invalid.")

    print("\nChannel mapping:")
    for i, name in enumerate(CHANNELS):
        print(f"  {i:02d} -> {name}")

    # -------------------------------------------------------------------------
    # Validate masks
    # -------------------------------------------------------------------------

    header("VALIDATING PHASE 5 MASKS")

    mask_file = PHASE5_DIR / "ocean_mask.nc"

    if not mask_file.exists():
        fail(f"Missing ocean mask: {mask_file}")

    mask_ds = xr.open_dataset(mask_file)

    print(f"Mask file: {mask_file}")
    print(f"Variables : {list(mask_ds.data_vars)}")

    if "ocean_mask" not in mask_ds:
        mask_ds.close()
        fail(
            "ocean_mask.nc does not contain variable 'ocean_mask'."
        )

    ocean_mask = mask_ds["ocean_mask"].load()

    mask_ds.close()

    if ocean_mask.dims != ("latitude", "longitude"):
        fail(
            f"Unexpected ocean mask dimensions: "
            f"{ocean_mask.dims}"
        )

    if ocean_mask.shape != (
        EXPECTED_LAT,
        EXPECTED_LON,
    ):
        fail(
            f"Unexpected ocean mask shape: "
            f"{ocean_mask.shape}"
        )

    print(
        f"Permanent unavailable cells: "
        f"{int((ocean_mask == 0).sum()):,}"
    )

    print("Ocean mask: VALID")

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    header("WRITING INPUT TENSOR")

    PHASE5_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoding = {
        "input_tensor": {
            "dtype": "float32",
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": (
                1,
                20,
                20,
                len(CHANNELS),
            ),
        }
    }

    tensor.to_netcdf(
        OUTPUT_FILE,
        encoding=encoding,
    )

    print(f"Wrote: {OUTPUT_FILE}")

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    header("PHASE 5 INPUT TENSOR ASSEMBLY COMPLETE")

    print("Output:")
    print(f"  {OUTPUT_FILE}")
    print()
    print(f"Shape:")
    print(f"  {tensor.shape}")
    print()
    print("Channels:")
    for i, name in enumerate(CHANNELS):
        print(f"  [{i:02d}] {name}")

    print()
    print("Validation:")
    print("  Grid                 : PASS")
    print("  Coordinates          : PASS")
    print("  Channel count        : PASS")
    print("  Shape                : PASS")
    print("  NaNs                 : 0")
    print("  Infs                 : 0")
    print("  Normalization        : NOT APPLIED")
    print("  Source files modified: NO")


if __name__ == "__main__":
    main()