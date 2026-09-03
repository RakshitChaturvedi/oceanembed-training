from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# Paths
# =============================================================================

TEMPERATURE_RAW = Path(
    "data/processed/phase7/temperature_targets_raw.nc"
)

SALINITY_RAW = Path(
    "data/processed/phase7/salinity_targets_raw.nc"
)

TEMPERATURE_OUTPUT = Path(
    "data/processed/phase7/temperature_targets_masked.nc"
)

SALINITY_OUTPUT = Path(
    "data/processed/phase7/salinity_targets_masked.nc"
)

OCEAN_MASK = Path(
    "data/processed/phase5/ocean_mask.nc"
)

DEPTH_MAPPING = Path(
    "data/processed/phase7/depth_mapping.json"
)


# =============================================================================
# Contract
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

TIME_CHUNK = 16

TEMPERATURE_MIN = -5.0
TEMPERATURE_MAX = 45.0

SALINITY_MIN = 0.0
SALINITY_MAX = 45.0

RTOL = 1e-5
ATOL = 1e-5


# =============================================================================
# Counters
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
print("OceanEmbed — Phase 7 Target Validation")
print("=" * 90)

print(
    """
Purpose:
  Independently validate the final masked GLORYS temperature
  and salinity training targets.

Rules:
  - No source artifact is modified.
  - No complete target tensor is loaded.
  - Validation is performed lazily and in time chunks.
  - Land cells must remain NaN.
  - Ocean target values must remain unchanged by masking.
"""
)


# =============================================================================
# [1] Required artifacts
# =============================================================================

print("\n" + "=" * 90)
print("[1] REQUIRED ARTIFACTS")
print("=" * 90)

required = {
    "temperature raw": TEMPERATURE_RAW,
    "salinity raw": SALINITY_RAW,
    "temperature masked": TEMPERATURE_OUTPUT,
    "salinity masked": SALINITY_OUTPUT,
    "ocean mask": OCEAN_MASK,
    "depth mapping": DEPTH_MAPPING,
}

for name, path in required.items():
    check(
        f"{name} exists",
        path.exists(),
        str(path),
    )

if failed:
    raise SystemExit(1)


# =============================================================================
# [2] Open datasets lazily
# =============================================================================

print("\n" + "=" * 90)
print("[2] OPEN TARGET DATASETS")
print("=" * 90)

temp_raw_ds = xr.open_dataset(TEMPERATURE_RAW)
sal_raw_ds = xr.open_dataset(SALINITY_RAW)

temp_ds = xr.open_dataset(TEMPERATURE_OUTPUT)
sal_ds = xr.open_dataset(SALINITY_OUTPUT)

mask_ds = xr.open_dataset(OCEAN_MASK)

print("Temperature masked:")
print(f"  dimensions: {dict(temp_ds.sizes)}")
print(f"  variables : {list(temp_ds.data_vars)}")

print("\nSalinity masked:")
print(f"  dimensions: {dict(sal_ds.sizes)}")
print(f"  variables : {list(sal_ds.data_vars)}")


# =============================================================================
# [3] Variable structure
# =============================================================================

print("\n" + "=" * 90)
print("[3] VARIABLE STRUCTURE")
print("=" * 90)

check(
    "temperature variable exists",
    "temperature" in temp_ds,
)

check(
    "salinity variable exists",
    "salinity" in sal_ds,
)

check(
    "raw temperature variable exists",
    "temperature" in temp_raw_ds,
)

check(
    "raw salinity variable exists",
    "salinity" in sal_raw_ds,
)

check(
    "ocean mask variable exists",
    "ocean_mask" in mask_ds,
)


# =============================================================================
# [4] Dimensions
# =============================================================================

print("\n" + "=" * 90)
print("[4] DIMENSIONS")
print("=" * 90)

expected_sizes = {
    "time": EXPECTED_TIME,
    "latitude": EXPECTED_LATITUDE,
    "longitude": EXPECTED_LONGITUDE,
    "depth": EXPECTED_DEPTH,
}

for dim, expected in expected_sizes.items():

    check(
        f"temperature {dim} dimension",
        temp_ds.sizes.get(dim) == expected,
        f"expected={expected}, actual={temp_ds.sizes.get(dim)}",
    )

    check(
        f"salinity {dim} dimension",
        sal_ds.sizes.get(dim) == expected,
        f"expected={expected}, actual={sal_ds.sizes.get(dim)}",
    )


# =============================================================================
# [5] Dimension order
# =============================================================================

print("\n" + "=" * 90)
print("[5] DIMENSION ORDER")
print("=" * 90)

expected_dims = (
    "time",
    "latitude",
    "longitude",
    "depth",
)

check(
    "temperature dimension order",
    temp_ds["temperature"].dims == expected_dims,
    f"actual={temp_ds['temperature'].dims}",
)

check(
    "salinity dimension order",
    sal_ds["salinity"].dims == expected_dims,
    f"actual={sal_ds['salinity'].dims}",
)


# =============================================================================
# [6] Coordinate alignment
# =============================================================================

print("\n" + "=" * 90)
print("[6] COORDINATE ALIGNMENT")
print("=" * 90)

for coord in (
    "time",
    "latitude",
    "longitude",
    "depth",
):

    check(
        f"temperature ↔ salinity {coord}",
        np.array_equal(
            temp_ds[coord].values,
            sal_ds[coord].values,
        ),
    )


# =============================================================================
# [7] Depth contract
# =============================================================================

print("\n" + "=" * 90)
print("[7] DEPTH CONTRACT")
print("=" * 90)

check(
    "temperature target depths",
    np.array_equal(
        temp_ds["depth"].values,
        TARGET_DEPTHS,
    ),
)

check(
    "salinity target depths",
    np.array_equal(
        sal_ds["depth"].values,
        TARGET_DEPTHS,
    ),
)

print("\nTarget depths:")

for depth in TARGET_DEPTHS:
    print(f"  {float(depth):8.1f} m")


# =============================================================================
# [8] Ocean mask alignment
# =============================================================================

print("\n" + "=" * 90)
print("[8] OCEAN MASK ALIGNMENT")
print("=" * 90)

mask = mask_ds["ocean_mask"]

check(
    "mask latitude count",
    mask.sizes.get("latitude") == EXPECTED_LATITUDE,
)

check(
    "mask longitude count",
    mask.sizes.get("longitude") == EXPECTED_LONGITUDE,
)

check(
    "mask latitude alignment",
    np.array_equal(
        mask["latitude"].values,
        temp_ds["latitude"].values,
    ),
)

check(
    "mask longitude alignment",
    np.array_equal(
        mask["longitude"].values,
        temp_ds["longitude"].values,
    ),
)

check(
    "mask is boolean",
    mask.dtype == bool,
)


# =============================================================================
# [9] 1000 m mapping
# =============================================================================

print("\n" + "=" * 90)
print("[9] 1000 m DEEP-SOURCE VALIDATION")
print("=" * 90)

with DEPTH_MAPPING.open("r", encoding="utf-8") as f:
    mapping_artifact = json.load(f)

mapping_entries = mapping_artifact["mapping"]

deep_entries = [
    entry
    for entry in mapping_entries
    if entry["target_depth_m"] == 1000
]

check(
    "1000 m mapping entry exists",
    len(deep_entries) == 1,
)

if deep_entries:

    entry = deep_entries[0]

    check(
        "1000 m uses deep_glorys",
        entry["source_type"] == "deep_glorys",
        f"actual={entry['source_type']}",
    )

    check(
        "1000 m source file",
        entry["source_file"].endswith(
            "glorys_1062.nc"
        ),
        f"actual={entry['source_file']}",
    )

    check(
        "1000 m source depth",
        np.isclose(
            entry["source_depth_m"],
            1062.43994140625,
            atol=1e-3,
        ),
        f"actual={entry['source_depth_m']}",
    )


# =============================================================================
# [10] Time-chunked numerical validation
# =============================================================================

print("\n" + "=" * 90)
print("[10] TIME-CHUNKED NUMERICAL VALIDATION")
print("=" * 90)

temp = temp_ds["temperature"]
sal = sal_ds["salinity"]

temp_raw = temp_raw_ds["temperature"]
sal_raw = sal_raw_ds["salinity"]

temperature_inf = 0
salinity_inf = 0

temperature_land_finite = 0
salinity_land_finite = 0

temperature_ocean_changed = 0
salinity_ocean_changed = 0

temperature_ocean_finite = 0
salinity_ocean_finite = 0

temperature_land_nan = 0
salinity_land_nan = 0

temperature_physical_fail = 0
salinity_physical_fail = 0


for start in range(0, EXPECTED_TIME, TIME_CHUNK):

    end = min(
        start + TIME_CHUNK,
        EXPECTED_TIME,
    )

    print(
        f"  Processing time [{start}:{end}]"
    )

    t = temp.isel(
        time=slice(start, end)
    ).values

    s = sal.isel(
        time=slice(start, end)
    ).values

    tr = temp_raw.isel(
        time=slice(start, end)
    ).values

    sr = sal_raw.isel(
        time=slice(start, end)
    ).values

    mask_chunk = np.broadcast_to(
        mask.values,
        (end - start, EXPECTED_LATITUDE, EXPECTED_LONGITUDE),
    )

    mask_chunk = mask_chunk[:, :, :, None]

    # -------------------------------------------------------------------------
    # Infinities
    # -------------------------------------------------------------------------

    temperature_inf += np.count_nonzero(
        np.isinf(t)
    )

    salinity_inf += np.count_nonzero(
        np.isinf(s)
    )

    # -------------------------------------------------------------------------
    # Land integrity
    # -------------------------------------------------------------------------

    land = ~mask_chunk

    temperature_land_finite += np.count_nonzero(
        np.isfinite(t) & land
    )

    salinity_land_finite += np.count_nonzero(
        np.isfinite(s) & land
    )

    temperature_land_nan += np.count_nonzero(
        np.isnan(t) & land
    )

    salinity_land_nan += np.count_nonzero(
        np.isnan(s) & land
    )

    # -------------------------------------------------------------------------
    # Ocean values
    # -------------------------------------------------------------------------

    ocean = mask_chunk

    temperature_ocean_finite += np.count_nonzero(
        np.isfinite(t) & ocean
    )

    salinity_ocean_finite += np.count_nonzero(
        np.isfinite(s) & ocean
    )

    # -------------------------------------------------------------------------
    # Raw → masked preservation
    #
    # Only compare cells where:
    #   raw is finite
    #   ocean mask is true
    # -------------------------------------------------------------------------

    temp_compare = (
        np.isfinite(tr)
        & np.isfinite(t)
        & ocean
    )

    sal_compare = (
        np.isfinite(sr)
        & np.isfinite(s)
        & ocean
    )

    temperature_ocean_changed += np.count_nonzero(
        temp_compare
        & ~np.isclose(
            t,
            tr,
            rtol=RTOL,
            atol=ATOL,
        )
    )

    salinity_ocean_changed += np.count_nonzero(
        sal_compare
        & ~np.isclose(
            s,
            sr,
            rtol=RTOL,
            atol=ATOL,
        )
    )

    # -------------------------------------------------------------------------
    # Broad physical sanity
    # -------------------------------------------------------------------------

    temp_finite = np.isfinite(t)

    sal_finite = np.isfinite(s)

    temperature_physical_fail += np.count_nonzero(
        temp_finite
        & (
            (t < TEMPERATURE_MIN)
            | (t > TEMPERATURE_MAX)
        )
    )

    salinity_physical_fail += np.count_nonzero(
        sal_finite
        & (
            (s < SALINITY_MIN)
            | (s > SALINITY_MAX)
        )
    )


# =============================================================================
# [11] Numerical results
# =============================================================================

print("\n" + "=" * 90)
print("[11] NUMERICAL RESULTS")
print("=" * 90)

print(
    f"Temperature infinities : {temperature_inf}"
)

print(
    f"Salinity infinities    : {salinity_inf}"
)

print(
    f"Temperature finite land cells : "
    f"{temperature_land_finite}"
)

print(
    f"Salinity finite land cells    : "
    f"{salinity_land_finite}"
)

print(
    f"Temperature ocean values changed : "
    f"{temperature_ocean_changed}"
)

print(
    f"Salinity ocean values changed    : "
    f"{salinity_ocean_changed}"
)

print(
    f"Temperature physical failures : "
    f"{temperature_physical_fail}"
)

print(
    f"Salinity physical failures    : "
    f"{salinity_physical_fail}"
)

check(
    "temperature contains no infinities",
    temperature_inf == 0,
)

check(
    "salinity contains no infinities",
    salinity_inf == 0,
)

check(
    "temperature land cells are NaN",
    temperature_land_finite == 0,
)

check(
    "salinity land cells are NaN",
    salinity_land_finite == 0,
)

check(
    "temperature ocean values preserved",
    temperature_ocean_changed == 0,
)

check(
    "salinity ocean values preserved",
    salinity_ocean_changed == 0,
)

check(
    "temperature broad physical sanity",
    temperature_physical_fail == 0,
)

check(
    "salinity broad physical sanity",
    salinity_physical_fail == 0,
)


# =============================================================================
# [12] Per-depth statistics
# =============================================================================

print("\n" + "=" * 90)
print("[12] PER-DEPTH STATISTICS")
print("=" * 90)

print(
    "\nTEMPERATURE"
)

print(
    f"{'DEPTH':>8} "
    f"{'FINITE':>12} "
    f"{'NAN':>12} "
    f"{'MIN':>12} "
    f"{'MAX':>12} "
    f"{'MEAN':>12}"
)

for depth_index, depth in enumerate(TARGET_DEPTHS):

    finite_count = 0
    nan_count = 0
    minimum = np.inf
    maximum = -np.inf
    total = 0.0

    for start in range(
        0,
        EXPECTED_TIME,
        TIME_CHUNK,
    ):

        end = min(
            start + TIME_CHUNK,
            EXPECTED_TIME,
        )

        chunk = temp.isel(
            time=slice(start, end),
            depth=depth_index,
        ).values

        finite = np.isfinite(chunk)

        values = chunk[finite]

        finite_count += values.size
        nan_count += np.count_nonzero(
            np.isnan(chunk)
        )

        if values.size:
            minimum = min(
                minimum,
                float(values.min()),
            )

            maximum = max(
                maximum,
                float(values.max()),
            )

            total += float(
                values.sum(dtype=np.float64)
            )

    mean = (
        total / finite_count
        if finite_count
        else np.nan
    )

    print(
        f"{float(depth):8.1f} "
        f"{finite_count:12d} "
        f"{nan_count:12d} "
        f"{minimum:12.5f} "
        f"{maximum:12.5f} "
        f"{mean:12.5f}"
    )


print(
    "\nSALINITY"
)

print(
    f"{'DEPTH':>8} "
    f"{'FINITE':>12} "
    f"{'NAN':>12} "
    f"{'MIN':>12} "
    f"{'MAX':>12} "
    f"{'MEAN':>12}"
)

for depth_index, depth in enumerate(TARGET_DEPTHS):

    finite_count = 0
    nan_count = 0
    minimum = np.inf
    maximum = -np.inf
    total = 0.0

    for start in range(
        0,
        EXPECTED_TIME,
        TIME_CHUNK,
    ):

        end = min(
            start + TIME_CHUNK,
            EXPECTED_TIME,
        )

        chunk = sal.isel(
            time=slice(start, end),
            depth=depth_index,
        ).values

        finite = np.isfinite(chunk)

        values = chunk[finite]

        finite_count += values.size
        nan_count += np.count_nonzero(
            np.isnan(chunk)
        )

        if values.size:
            minimum = min(
                minimum,
                float(values.min()),
            )

            maximum = max(
                maximum,
                float(values.max()),
            )

            total += float(
                values.sum(dtype=np.float64)
            )

    mean = (
        total / finite_count
        if finite_count
        else np.nan
    )

    print(
        f"{float(depth):8.1f} "
        f"{finite_count:12d} "
        f"{nan_count:12d} "
        f"{minimum:12.5f} "
        f"{maximum:12.5f} "
        f"{mean:12.5f}"
    )


# =============================================================================
# Cleanup
# =============================================================================

temp_raw_ds.close()
sal_raw_ds.close()

temp_ds.close()
sal_ds.close()

mask_ds.close()


# =============================================================================
# Final result
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 7 — PART 5 RESULT")
print("=" * 90)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:

    print(
        """
PART 5 TARGET VALIDATION: PASS

The final masked GLORYS temperature and salinity targets
satisfy the structural, coordinate, masking, numerical,
physical-sanity, and source-preservation checks.
"""
    )

else:

    print(
        """
PART 5 TARGET VALIDATION: FAIL

Do not proceed to train/validation/test splitting.
"""
    )

print("=" * 90)

raise SystemExit(
    0 if failed == 0 else 1
)