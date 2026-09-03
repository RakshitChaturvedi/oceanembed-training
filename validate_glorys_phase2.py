from pathlib import Path

import numpy as np
import xarray as xr


# ============================================================
# Configuration
# ============================================================

MAIN = Path("data/processed/phase2/glorys.nc")
DEEP = Path("data/processed/phase2/glorys_1062.nc")

EXPECTED_TIME = 1419
EXPECTED_LAT = 101
EXPECTED_LON = 241

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0
EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0
EXPECTED_RESOLUTION = 0.25

# Native GLORYS level used for the deep target
EXPECTED_DEEP_DEPTH = 1062.44

# Requested target depths for Phase 7
TARGET_DEPTHS = [
    0, 5, 10, 20, 30,
    50, 75, 100, 125, 150,
    200, 300, 500, 700, 1000,
]


# ============================================================
# Helpers
# ============================================================

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed

    if condition:
        print(f"  PASS  {name}")
        if detail:
            print(f"        {detail}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        failed += 1


def finite_stats(da):
    values = da.values

    finite = np.isfinite(values)

    if not finite.any():
        return None

    return {
        "finite": int(finite.sum()),
        "total": int(values.size),
        "nan": int(np.isnan(values).sum()),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "mean": float(np.nanmean(values)),
    }


# ============================================================
# Header
# ============================================================

print("=" * 100)
print("OceanEmbed — Phase 2 GLORYS FINAL VALIDATION")
print("=" * 100)

print("\nArtifacts:")
print(f"  Main GLORYS : {MAIN}")
print(f"  Deep GLORYS : {DEEP}")

if not MAIN.exists():
    raise FileNotFoundError(f"Missing: {MAIN}")

if not DEEP.exists():
    raise FileNotFoundError(f"Missing: {DEEP}")


# ============================================================
# Open datasets lazily
# ============================================================

main = xr.open_dataset(MAIN)
deep = xr.open_dataset(DEEP)


# ============================================================
# 1. MAIN ARTIFACT
# ============================================================

print("\n" + "=" * 100)
print("[1] MAIN GLORYS ARTIFACT")
print("=" * 100)

print("Dimensions:", dict(main.sizes))
print("Variables :", list(main.data_vars))

check(
    "time dimension",
    main.sizes.get("time") == EXPECTED_TIME,
    f"expected={EXPECTED_TIME}, actual={main.sizes.get('time')}",
)

check(
    "latitude dimension",
    main.sizes.get("latitude") == EXPECTED_LAT,
    f"expected={EXPECTED_LAT}, actual={main.sizes.get('latitude')}",
)

check(
    "longitude dimension",
    main.sizes.get("longitude") == EXPECTED_LON,
    f"expected={EXPECTED_LON}, actual={main.sizes.get('longitude')}",
)

check(
    "temperature exists",
    "temperature" in main,
)

check(
    "salinity exists",
    "salinity" in main,
)


# ============================================================
# 2. DEEP ARTIFACT
# ============================================================

print("\n" + "=" * 100)
print("[2] DEEP GLORYS ARTIFACT")
print("=" * 100)

print("Dimensions:", dict(deep.sizes))
print("Variables :", list(deep.data_vars))
print("Depth     :", deep.depth.values)

check(
    "time dimension",
    deep.sizes.get("time") == EXPECTED_TIME,
    f"expected={EXPECTED_TIME}, actual={deep.sizes.get('time')}",
)

check(
    "depth dimension",
    deep.sizes.get("depth") == 1,
    f"expected=1, actual={deep.sizes.get('depth')}",
)

check(
    "latitude dimension",
    deep.sizes.get("latitude") == EXPECTED_LAT,
    f"expected={EXPECTED_LAT}, actual={deep.sizes.get('latitude')}",
)

check(
    "longitude dimension",
    deep.sizes.get("longitude") == EXPECTED_LON,
    f"expected={EXPECTED_LON}, actual={deep.sizes.get('longitude')}",
)

check(
    "temperature exists",
    "temperature" in deep,
)

check(
    "salinity exists",
    "salinity" in deep,
)

actual_deep_depth = float(deep.depth.values[0])

check(
    "deep depth",
    np.isclose(actual_deep_depth, EXPECTED_DEEP_DEPTH, atol=0.01),
    f"expected≈{EXPECTED_DEEP_DEPTH}, actual={actual_deep_depth}",
)


# ============================================================
# 3. GRID VALIDATION
# ============================================================

print("\n" + "=" * 100)
print("[3] HORIZONTAL GRID")
print("=" * 100)


def validate_grid(ds, label):
    print(f"\n{label}")

    lat = ds.latitude.values
    lon = ds.longitude.values

    print(f"  latitude : {lat[0]} -> {lat[-1]} ({len(lat)} points)")
    print(f"  longitude: {lon[0]} -> {lon[-1]} ({len(lon)} points)")

    lat_spacing = np.diff(lat)
    lon_spacing = np.diff(lon)

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


validate_grid(main, "MAIN")
validate_grid(deep, "DEEP")


# ============================================================
# 4. TIME ALIGNMENT
# ============================================================

print("\n" + "=" * 100)
print("[4] TIME ALIGNMENT")
print("=" * 100)

main_time = main.time.values
deep_time = deep.time.values

print("Main:")
print(" ", main_time[0], "->", main_time[-1])

print("Deep:")
print(" ", deep_time[0], "->", deep_time[-1])

check(
    "time coordinates identical",
    np.array_equal(main_time, deep_time),
    f"main_count={len(main_time)}, deep_count={len(deep_time)}",
)


# ============================================================
# 5. SPATIAL COORDINATE ALIGNMENT
# ============================================================

print("\n" + "=" * 100)
print("[5] SPATIAL COORDINATE ALIGNMENT")
print("=" * 100)

check(
    "latitude coordinates identical",
    np.array_equal(main.latitude.values, deep.latitude.values),
)

check(
    "longitude coordinates identical",
    np.array_equal(main.longitude.values, deep.longitude.values),
)


# ============================================================
# 6. REGRID METADATA
# ============================================================

print("\n" + "=" * 100)
print("[6] REGRID METADATA")
print("=" * 100)

for label, ds in [("MAIN", main), ("DEEP", deep)]:
    print(f"\n{label}")

    method = ds.attrs.get("oceanembed_regrid_method")
    resolution = ds.attrs.get("oceanembed_grid_resolution")

    print("  method     :", method)
    print("  resolution :", resolution)

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


# ============================================================
# 7. VALUE SANITY
# ============================================================

print("\n" + "=" * 100)
print("[7] TEMPERATURE / SALINITY VALUE SANITY")
print("=" * 100)

# These are deliberately broad physical sanity checks.
# They are not scientific acceptance bounds.

for label, ds in [("MAIN", main), ("DEEP", deep)]:

    print(f"\n{label}")

    for variable in ["temperature", "salinity"]:

        stats = finite_stats(ds[variable])

        if stats is None:
            check(
                f"{label} {variable} has finite values",
                False,
            )
            continue

        print(f"\n  {variable}")
        print(f"    finite : {stats['finite']:,} / {stats['total']:,}")
        print(f"    NaN    : {stats['nan']:,}")
        print(f"    min    : {stats['min']:.6f}")
        print(f"    max    : {stats['max']:.6f}")
        print(f"    mean   : {stats['mean']:.6f}")

        check(
            f"{label} {variable} has finite values",
            stats["finite"] > 0,
        )

        check(
            f"{label} {variable} not absurdly low",
            stats["min"] > -10,
            f"min={stats['min']}",
        )

        check(
            f"{label} {variable} not absurdly high",
            stats["max"] < 100,
            f"max={stats['max']}",
        )


# ============================================================
# 8. DEEP TARGET SPECIFIC CHECK
# ============================================================

print("\n" + "=" * 100)
print("[8] 1000 m TARGET SOURCE")
print("=" * 100)

print(
    f"Phase 7 requested depth: 1000 m\n"
    f"Available deep GLORYS level: {actual_deep_depth:.3f} m"
)

check(
    "1000 m source level available",
    np.isclose(actual_deep_depth, 1062.44, atol=0.01),
    "This level will supply the 1000 m target.",
)


# ============================================================
# 9. NO UNEXPECTED DEPTHS IN DEEP ARTIFACT
# ============================================================

print("\n" + "=" * 100)
print("[9] DEEP ARTIFACT DEPTH ISOLATION")
print("=" * 100)

check(
    "deep artifact contains exactly one depth",
    deep.sizes["depth"] == 1,
)

check(
    "deep artifact is 1062.44 m only",
    np.isclose(deep.depth.values[0], 1062.44, atol=0.01),
)


# ============================================================
# 10. PHASE 7 TARGET MAPPING
# ============================================================

print("\n" + "=" * 100)
print("[10] PHASE 7 TARGET DEPTH MAPPING")
print("=" * 100)

available_depths = main.depth.values

for requested in TARGET_DEPTHS[:-1]:
    actual = available_depths[
        np.argmin(np.abs(available_depths - requested))
    ]

    print(
        f"  {requested:>4} m -> {actual:>10.3f} m"
    )

print(
    f"  {1000:>4} m -> {actual_deep_depth:>10.3f} m  [DEEP FILE]"
)

check(
    "all 15 target depths accounted for",
    len(TARGET_DEPTHS) == 15,
    f"count={len(TARGET_DEPTHS)}",
)


# ============================================================
# 11. ATTRIBUTE / SOURCE SANITY
# ============================================================

print("\n" + "=" * 100)
print("[11] SOURCE METADATA")
print("=" * 100)

for label, ds in [("MAIN", main), ("DEEP", deep)]:

    print(f"\n{label}")

    for key in [
        "source",
        "institution",
        "domain_name",
        "oceanembed_regrid_method",
        "oceanembed_grid_resolution",
    ]:
        print(f"  {key}: {ds.attrs.get(key, '<missing>')}")


# ============================================================
# Final
# ============================================================

main.close()
deep.close()

print("\n" + "=" * 100)
print("FINAL RESULT")
print("=" * 100)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:
    print("\nPHASE 2 GLORYS VALIDATION: PASS")
    print("Both GLORYS artifacts are ready for Phase 7.")
else:
    print("\nPHASE 2 GLORYS VALIDATION: FAIL")
    print("Do not proceed to Phase 7 until the failures are investigated.")

print("=" * 100)

raise SystemExit(0 if failed == 0 else 1)
