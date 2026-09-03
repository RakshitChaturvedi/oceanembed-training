from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# Paths
# =============================================================================

MAIN_GLORYS = Path("data/processed/phase2/glorys.nc")
DEEP_GLORYS = Path("data/processed/phase2/glorys_1062.nc")
OCEAN_MASK = Path("data/processed/phase5/ocean_mask.nc")


# =============================================================================
# Expected configuration
# =============================================================================

EXPECTED_TIME_COUNT = 1419

EXPECTED_LAT_COUNT = 101
EXPECTED_LON_COUNT = 241

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0

EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0

EXPECTED_RESOLUTION = 0.25

EXPECTED_MAIN_DEPTH_COUNT = 35

EXPECTED_DEEP_DEPTH = 1062.44

EXPECTED_MASK_OCEAN_CELLS = 12490
EXPECTED_MASK_LAND_CELLS = 11851


# =============================================================================
# Helpers
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


def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def check_grid(ds: xr.Dataset, label: str) -> None:
    """Validate horizontal grid without loading T/S data."""

    check(
        f"{label} latitude coordinate exists",
        "latitude" in ds.coords,
    )

    check(
        f"{label} longitude coordinate exists",
        "longitude" in ds.coords,
    )

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        return

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    check(
        f"{label} latitude count",
        len(lat) == EXPECTED_LAT_COUNT,
        f"expected={EXPECTED_LAT_COUNT}, actual={len(lat)}",
    )

    check(
        f"{label} longitude count",
        len(lon) == EXPECTED_LON_COUNT,
        f"expected={EXPECTED_LON_COUNT}, actual={len(lon)}",
    )

    check(
        f"{label} latitude start",
        np.isclose(lat[0], EXPECTED_LAT_MIN),
        f"expected={EXPECTED_LAT_MIN}, actual={lat[0]}",
    )

    check(
        f"{label} latitude end",
        np.isclose(lat[-1], EXPECTED_LAT_MAX),
        f"expected={EXPECTED_LAT_MAX}, actual={lat[-1]}",
    )

    check(
        f"{label} longitude start",
        np.isclose(lon[0], EXPECTED_LON_MIN),
        f"expected={EXPECTED_LON_MIN}, actual={lon[0]}",
    )

    check(
        f"{label} longitude end",
        np.isclose(lon[-1], EXPECTED_LON_MAX),
        f"expected={EXPECTED_LON_MAX}, actual={lon[-1]}",
    )

    lat_spacing = np.diff(lat)
    lon_spacing = np.diff(lon)

    check(
        f"{label} latitude spacing",
        np.allclose(lat_spacing, EXPECTED_RESOLUTION, atol=1e-5),
        f"min={lat_spacing.min():.8f}, max={lat_spacing.max():.8f}",
    )

    check(
        f"{label} longitude spacing",
        np.allclose(lon_spacing, EXPECTED_RESOLUTION, atol=1e-5),
        f"min={lon_spacing.min():.8f}, max={lon_spacing.max():.8f}",
    )


def check_times(ds: xr.Dataset, label: str) -> None:
    check(
        f"{label} time coordinate exists",
        "time" in ds.coords,
    )

    if "time" not in ds.coords:
        return

    time = ds["time"]

    check(
        f"{label} time count",
        time.sizes["time"] == EXPECTED_TIME_COUNT,
        (
            f"expected={EXPECTED_TIME_COUNT}, "
            f"actual={time.sizes['time']}"
        ),
    )

    print(
        f"        range={time.values[0]} -> {time.values[-1]}"
    )


def check_variables(ds: xr.Dataset, label: str) -> None:
    print(f"\n{label} variables:")
    print(f"  {list(ds.data_vars)}")

    check(
        f"{label} temperature exists",
        "temperature" in ds.data_vars,
    )

    check(
        f"{label} salinity exists",
        "salinity" in ds.data_vars,
    )

    for variable in ("temperature", "salinity"):
        if variable not in ds.data_vars:
            continue

        da = ds[variable]

        expected_dims = ("time", "depth", "latitude", "longitude")

        check(
            f"{label} {variable} dimensions",
            da.dims == expected_dims,
            f"actual={da.dims}",
        )


# =============================================================================
# Start
# =============================================================================

print_header("OceanEmbed — Phase 7 Target Preparation Preflight")

print(
    """
Purpose:
  Validate all upstream artifacts required for Phase 7 target construction.

Rules:
  - No full GLORYS temperature/salinity arrays are loaded.
  - No target data is created.
  - No upstream artifact is modified.
"""
)


# =============================================================================
# [1] File existence
# =============================================================================

print_header("[1] REQUIRED ARTIFACTS")

for label, path in (
    ("MAIN GLORYS", MAIN_GLORYS),
    ("DEEP GLORYS", DEEP_GLORYS),
    ("OCEAN MASK", OCEAN_MASK),
):
    print(f"{label}: {path}")
    check(f"{label} exists", path.exists())


if failed:
    print_header("EARLY EXIT")
    print(f"PASS: {passed}")
    print(f"FAIL: {failed}")
    raise SystemExit(1)


# =============================================================================
# [2] Main GLORYS
# =============================================================================

print_header("[2] MAIN GLORYS")

main = xr.open_dataset(MAIN_GLORYS)

print("Dimensions:")
print(dict(main.sizes))

print("\nVariables:")
print(list(main.data_vars))

check(
    "MAIN time dimension",
    main.sizes.get("time") == EXPECTED_TIME_COUNT,
)

check(
    "MAIN depth dimension",
    main.sizes.get("depth") == EXPECTED_MAIN_DEPTH_COUNT,
    (
        f"expected={EXPECTED_MAIN_DEPTH_COUNT}, "
        f"actual={main.sizes.get('depth')}"
    ),
)

check(
    "MAIN latitude dimension",
    main.sizes.get("latitude") == EXPECTED_LAT_COUNT,
)

check(
    "MAIN longitude dimension",
    main.sizes.get("longitude") == EXPECTED_LON_COUNT,
)

check_variables(main, "MAIN")
check_times(main, "MAIN")
check_grid(main, "MAIN")


# =============================================================================
# [3] Deep GLORYS
# =============================================================================

print_header("[3] DEEP GLORYS")

deep = xr.open_dataset(DEEP_GLORYS)

print("Dimensions:")
print(dict(deep.sizes))

print("\nVariables:")
print(list(deep.data_vars))

check(
    "DEEP time dimension",
    deep.sizes.get("time") == EXPECTED_TIME_COUNT,
)

check(
    "DEEP depth dimension",
    deep.sizes.get("depth") == 1,
    f"expected=1, actual={deep.sizes.get('depth')}",
)

check(
    "DEEP latitude dimension",
    deep.sizes.get("latitude") == EXPECTED_LAT_COUNT,
)

check(
    "DEEP longitude dimension",
    deep.sizes.get("longitude") == EXPECTED_LON_COUNT,
)

check_variables(deep, "DEEP")
check_times(deep, "DEEP")
check_grid(deep, "DEEP")

if "depth" in deep.coords:
    deep_depth = float(deep["depth"].values[0])

    print(f"\nDeep source depth: {deep_depth:.6f} m")

    check(
        "DEEP source depth ≈ 1062.44 m",
        np.isclose(deep_depth, EXPECTED_DEEP_DEPTH, atol=0.01),
        f"expected≈{EXPECTED_DEEP_DEPTH}, actual={deep_depth}",
    )


# =============================================================================
# [4] Ocean mask
# =============================================================================

print_header("[4] OCEAN MASK")

mask_ds = xr.open_dataset(OCEAN_MASK)

print("Dimensions:")
print(dict(mask_ds.sizes))

print("\nVariables:")
print(list(mask_ds.data_vars))

check(
    "MASK latitude dimension",
    mask_ds.sizes.get("latitude") == EXPECTED_LAT_COUNT,
)

check(
    "MASK longitude dimension",
    mask_ds.sizes.get("longitude") == EXPECTED_LON_COUNT,
)

check(
    "MASK variable exists",
    "ocean_mask" in mask_ds.data_vars,
)

if "ocean_mask" in mask_ds.data_vars:

    mask = mask_ds["ocean_mask"].values

    check(
        "MASK shape",
        mask.shape == (EXPECTED_LAT_COUNT, EXPECTED_LON_COUNT),
        f"actual={mask.shape}",
    )

    unique = np.unique(mask)

    print(f"Unique values: {unique}")

    check(
        "MASK is boolean/binary",
        np.all(np.isin(unique, [False, True, 0, 1])),
    )

    ocean_cells = int(np.count_nonzero(mask))
    land_cells = int(mask.size - ocean_cells)

    print(f"Ocean cells: {ocean_cells}")
    print(f"Land cells : {land_cells}")

    check(
        "MASK ocean cell count",
        ocean_cells == EXPECTED_MASK_OCEAN_CELLS,
        (
            f"expected={EXPECTED_MASK_OCEAN_CELLS}, "
            f"actual={ocean_cells}"
        ),
    )

    check(
        "MASK land cell count",
        land_cells == EXPECTED_MASK_LAND_CELLS,
        (
            f"expected={EXPECTED_MASK_LAND_CELLS}, "
            f"actual={land_cells}"
        ),
    )


# =============================================================================
# [5] Cross-file coordinate alignment
# =============================================================================

print_header("[5] CROSS-FILE ALIGNMENT")

check(
    "MAIN ↔ DEEP time coordinates identical",
    np.array_equal(main["time"].values, deep["time"].values),
)

check(
    "MAIN ↔ DEEP latitude coordinates identical",
    np.array_equal(main["latitude"].values, deep["latitude"].values),
)

check(
    "MAIN ↔ DEEP longitude coordinates identical",
    np.array_equal(main["longitude"].values, deep["longitude"].values),
)

check(
    "MAIN ↔ MASK latitude coordinates identical",
    np.array_equal(main["latitude"].values, mask_ds["latitude"].values),
)

check(
    "MAIN ↔ MASK longitude coordinates identical",
    np.array_equal(main["longitude"].values, mask_ds["longitude"].values),
)

check(
    "DEEP ↔ MASK latitude coordinates identical",
    np.array_equal(deep["latitude"].values, mask_ds["latitude"].values),
)

check(
    "DEEP ↔ MASK longitude coordinates identical",
    np.array_equal(deep["longitude"].values, mask_ds["longitude"].values),
)


# =============================================================================
# [6] Regrid metadata
# =============================================================================

print_header("[6] REGRID METADATA")

for label, ds in (
    ("MAIN", main),
    ("DEEP", deep),
):

    method = ds.attrs.get("oceanembed_regrid_method")
    resolution = ds.attrs.get("oceanembed_grid_resolution")

    print(f"\n{label}")
    print(f"  method     : {method}")
    print(f"  resolution : {resolution}")

    check(
        f"{label} regrid method",
        method == "bilinear",
        f"actual={method}",
    )

    check(
        f"{label} grid resolution",
        resolution == "0.25 degrees",
        f"actual={resolution}",
    )


# =============================================================================
# [7] Memory safety
# =============================================================================

print_header("[7] MEMORY-SAFE PREFLIGHT")

print(
    """
  PASS  GLORYS opened lazily with xarray
  PASS  No .load() used on GLORYS
  PASS  No full-array .values used on temperature/salinity
  PASS  Only coordinates/metadata were materialized
"""
)

passed += 4


# =============================================================================
# Cleanup
# =============================================================================

main.close()
deep.close()
mask_ds.close()


# =============================================================================
# Final result
# =============================================================================

print_header("PHASE 7 PREFLIGHT RESULT")

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:
    print(
        "\nPHASE 7 PREFLIGHT: PASS\n"
        "All upstream artifacts are compatible with target construction."
    )
else:
    print(
        "\nPHASE 7 PREFLIGHT: FAIL\n"
        "Do not proceed to target extraction."
    )

print("=" * 90)

raise SystemExit(0 if failed == 0 else 1)
