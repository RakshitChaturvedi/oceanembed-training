from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# configs
PROJECT_ROOT = Path("/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training")

WIND_PATH = PROJECT_ROOT / "data/processed/phase2/wind.nc"

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0
EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0
EXPECTED_RESOLUTION = 0.25

EXPECTED_VARS = ["wind_u", "wind_v", "wind_speed"]

def section(title):
    print()
    print("="*90)
    print(title)
    print("="*90)

def subsection(title):
    print()
    print("-"*90)
    print(title)
    print("-"*90)

def pct(part, total):
    if total == 0:
        return 0.0
    return 100.0*part/total

def contiguous_true_runs(mask):
    # return lengths of contig true runs in 1d bool array
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    return (ends - starts).tolist()

def main():

    section("PHASE 4 — WIND STRESS CURL PRE-FLIGHT DIAGNOSTIC")

    print(f"File: {WIND_PATH}")

    if not WIND_PATH.exists():
        print("\nFAIL: wind.nc does not exist.")
        raise SystemExit(1)

    print("File exists: PASS")

    # ------------------------------------------------------------------------
    # Open dataset
    # ------------------------------------------------------------------------

    subsection("1. DATASET")

    ds = xr.open_dataset(WIND_PATH)

    print(ds)

    print("\nDimensions:")
    for name, size in ds.sizes.items():
        print(f"  {name:12s}: {size}")

    print("\nVariables:")
    for name in ds.data_vars:
        print(f"  {name}")

    # ------------------------------------------------------------------------
    # Variable validation
    # ------------------------------------------------------------------------

    subsection("2. VARIABLE VALIDATION")

    actual_vars = list(ds.data_vars)

    print(f"Expected: {EXPECTED_VARS}")
    print(f"Actual:   {actual_vars}")

    if set(actual_vars) == set(EXPECTED_VARS):
        print("Variable set: PASS")
    else:
        print("Variable set: FAIL")

    required_dims = {"time", "latitude", "longitude"}

    for var_name in EXPECTED_VARS:
        if var_name not in ds:
            continue

        var = ds[var_name]

        print(f"\n{var_name}")
        print(f"  dims:  {var.dims}")
        print(f"  shape: {var.shape}")
        print(f"  dtype: {var.dtype}")
        print(f"  units: {var.attrs.get('units', 'NOT SPECIFIED')}")

        if set(var.dims) == required_dims:
            print("  dimensions: PASS")
        else:
            print("  dimensions: FAIL")

    # ------------------------------------------------------------------------
    # Time validation
    # ------------------------------------------------------------------------

    subsection("3. TIME")

    time = ds["time"].values

    print(f"Start: {time[0]}")
    print(f"End:   {time[-1]}")
    print(f"Count: {len(time)}")

    time_series = pd.Series(pd.to_datetime(time))

    duplicates = int(time_series.duplicated().sum())

    print(f"Duplicates: {duplicates}")

    if duplicates == 0:
        print("Duplicates: PASS")
    else:
        print("Duplicates: FAIL")

    intervals = np.diff(time.astype("datetime64[D]")).astype(int)

    if len(intervals) > 0:
        unique_intervals = np.unique(intervals)

        print(f"Unique day intervals: {unique_intervals}")

        if np.all(intervals == 1):
            print("Daily spacing: PASS")
        else:
            print("Daily spacing: FAIL")

    # ------------------------------------------------------------------------
    # Spatial grid
    # ------------------------------------------------------------------------

    subsection("4. HORIZONTAL GRID")

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    print("Latitude:")
    print(f"  First: {lat[0]}")
    print(f"  Last:  {lat[-1]}")
    print(f"  Count: {len(lat)}")

    lat_spacing = np.diff(lat)

    if len(lat_spacing):
        print(f"  Min spacing: {lat_spacing.min()}")
        print(f"  Max spacing: {lat_spacing.max()}")
        print(f"  Unique spacing: {np.unique(lat_spacing)}")

    print("\nLongitude:")
    print(f"  First: {lon[0]}")
    print(f"  Last:  {lon[-1]}")
    print(f"  Count: {len(lon)}")

    lon_spacing = np.diff(lon)

    if len(lon_spacing):
        print(f"  Min spacing: {lon_spacing.min()}")
        print(f"  Max spacing: {lon_spacing.max()}")
        print(f"  Unique spacing: {np.unique(lon_spacing)}")

    lat_ok = (
        np.isclose(lat[0], EXPECTED_LAT_MIN)
        and np.isclose(lat[-1], EXPECTED_LAT_MAX)
        and np.allclose(lat_spacing, EXPECTED_RESOLUTION)
    )

    lon_ok = (
        np.isclose(lon[0], EXPECTED_LON_MIN)
        and np.isclose(lon[-1], EXPECTED_LON_MAX)
        and np.allclose(lon_spacing, EXPECTED_RESOLUTION)
    )

    print("\nGrid validation:")

    print(
        f"  Latitude increasing: "
        f"{'PASS' if np.all(lat_spacing > 0) else 'FAIL'}"
    )

    print(
        f"  Longitude increasing: "
        f"{'PASS' if np.all(lon_spacing > 0) else 'FAIL'}"
    )

    print(f"  Latitude grid:  {'PASS' if lat_ok else 'FAIL'}")
    print(f"  Longitude grid: {'PASS' if lon_ok else 'FAIL'}")

    # ------------------------------------------------------------------------
    # Wind variable diagnostics
    # ------------------------------------------------------------------------

    subsection("5. WIND DATA QUALITY")

    total_points = ds.sizes["time"] * ds.sizes["latitude"] * ds.sizes["longitude"]

    for var_name in ["wind_u", "wind_v", "wind_speed"]:

        if var_name not in ds:
            continue

        arr = ds[var_name].values

        nan_count = int(np.isnan(arr).sum())
        inf_count = int(np.isinf(arr).sum())
        zero_count = int((arr == 0).sum())

        finite = arr[np.isfinite(arr)]

        print(f"\n{var_name}")

        print(f"  Total:   {total_points:,}")
        print(f"  NaNs:    {nan_count:,} ({pct(nan_count, total_points):.4f}%)")
        print(f"  Infs:    {inf_count:,} ({pct(inf_count, total_points):.4f}%)")
        print(f"  Zeros:   {zero_count:,} ({pct(zero_count, total_points):.4f}%)")

        if len(finite):

            print(f"  Min:     {finite.min()}")
            print(f"  Max:     {finite.max()}")
            print(f"  Mean:    {finite.mean()}")
            print(f"  Median:  {np.median(finite)}")
            print(f"  Std:     {finite.std()}")

        print(
            f"  dtype:   {arr.dtype}"
        )

    # ------------------------------------------------------------------------
    # Units
    # ------------------------------------------------------------------------

    subsection("6. UNIT VALIDATION")

    for var_name in ["wind_u", "wind_v", "wind_speed"]:

        if var_name not in ds:
            continue

        units = ds[var_name].attrs.get("units")

        print(f"{var_name}: {units}")

    units_ok = (
        ds["wind_u"].attrs.get("units") == "m s-1"
        and ds["wind_v"].attrs.get("units") == "m s-1"
        and ds["wind_speed"].attrs.get("units") == "m s-1"
    )

    print(
        f"\nWind units: {'PASS' if units_ok else 'CHECK MANUALLY'}"
    )

    # ------------------------------------------------------------------------
    # U/V mask consistency
    # ------------------------------------------------------------------------

    subsection("7. U/V MASK CONSISTENCY")

    u = ds["wind_u"].values
    v = ds["wind_v"].values

    u_nan = np.isnan(u)
    v_nan = np.isnan(v)

    masks_identical = np.array_equal(u_nan, v_nan)

    print(
        f"U NaNs: {u_nan.sum():,}"
    )

    print(
        f"V NaNs: {v_nan.sum():,}"
    )

    print(
        f"U/V NaN masks identical: "
        f"{'PASS' if masks_identical else 'FAIL'}"
    )

    if not masks_identical:

        only_u = np.sum(u_nan & ~v_nan)
        only_v = np.sum(v_nan & ~u_nan)

        print(f"  NaN only in U: {only_u:,}")
        print(f"  NaN only in V: {only_v:,}")

    # ------------------------------------------------------------------------
    # Spatial missingness
    # ------------------------------------------------------------------------

    subsection("8. SPATIAL MISSINGNESS")

    # This is intentionally based on whether a cell EVER contains NaN.
    u_spatial_nan = np.isnan(u).any(axis=0)
    v_spatial_nan = np.isnan(v).any(axis=0)

    print(
        f"U spatial cells containing any NaN: "
        f"{u_spatial_nan.sum():,} / {u_spatial_nan.size:,} "
        f"({pct(u_spatial_nan.sum(), u_spatial_nan.size):.2f}%)"
    )

    print(
        f"V spatial cells containing any NaN: "
        f"{v_spatial_nan.sum():,} / {v_spatial_nan.size:,} "
        f"({pct(v_spatial_nan.sum(), v_spatial_nan.size):.2f}%)"
    )

    # Permanently missing spatial cells
    u_permanent_nan = np.isnan(u).all(axis=0)
    v_permanent_nan = np.isnan(v).all(axis=0)

    print(
        f"\nU permanently NaN cells: "
        f"{u_permanent_nan.sum():,}"
    )

    print(
        f"V permanently NaN cells: "
        f"{v_permanent_nan.sum():,}"
    )

    # Intermittent spatial cells
    u_intermittent = u_spatial_nan & ~u_permanent_nan
    v_intermittent = v_spatial_nan & ~v_permanent_nan

    print(
        f"U intermittently NaN cells: "
        f"{u_intermittent.sum():,}"
    )

    print(
        f"V intermittently NaN cells: "
        f"{v_intermittent.sum():,}"
    )

    # ------------------------------------------------------------------------
    # Detect whether NaNs form coherent spatial regions
    # ------------------------------------------------------------------------

    subsection("9. POSSIBLE LAND / MASK STRUCTURE")

    permanent_mask = u_permanent_nan | v_permanent_nan

    print(
        "This section checks whether permanently missing cells form "
        "large coherent spatial regions."
    )

    print(
        f"\nPermanent missing cells: "
        f"{permanent_mask.sum():,} / {permanent_mask.size:,}"
    )

    # Count missing cells per latitude / longitude
    missing_by_lat = permanent_mask.sum(axis=1)
    missing_by_lon = permanent_mask.sum(axis=0)

    print("\nPermanent missing cells by latitude:")

    top_lat_indices = np.argsort(missing_by_lat)[-10:][::-1]

    for idx in top_lat_indices:
        if missing_by_lat[idx] > 0:
            print(
                f"  lat={lat[idx]:7.2f}: "
                f"{missing_by_lat[idx]:5d} cells"
            )

    print("\nPermanent missing cells by longitude:")

    top_lon_indices = np.argsort(missing_by_lon)[-10:][::-1]

    for idx in top_lon_indices:
        if missing_by_lon[idx] > 0:
            print(
                f"  lon={lon[idx]:7.2f}: "
                f"{missing_by_lon[idx]:5d} cells"
            )

    # ------------------------------------------------------------------------
    # Check whether wind contains suspicious zero regions
    # ------------------------------------------------------------------------

    subsection("10. ZERO-VALUE STRUCTURE")

    u_zero = u == 0.0
    v_zero = v == 0.0

    print(
        f"U zero values: {u_zero.sum():,} "
        f"({pct(u_zero.sum(), total_points):.4f}%)"
    )

    print(
        f"V zero values: {v_zero.sum():,} "
        f"({pct(v_zero.sum(), total_points):.4f}%)"
    )

    both_zero = u_zero & v_zero

    print(
        f"U=0 AND V=0: {both_zero.sum():,} "
        f"({pct(both_zero.sum(), total_points):.4f}%)"
    )

    zero_spatial = both_zero.any(axis=0)

    print(
        f"Spatial cells containing at least one U=V=0: "
        f"{zero_spatial.sum():,} / {zero_spatial.size:,}"
    )

    permanent_zero = both_zero.all(axis=0)

    print(
        f"Spatial cells permanently U=V=0: "
        f"{permanent_zero.sum():,}"
    )

    # ------------------------------------------------------------------------
    # Physical sanity
    # ------------------------------------------------------------------------

    subsection("11. PHYSICAL SANITY")

    ranges = {
        "wind_u": (-100.0, 100.0),
        "wind_v": (-100.0, 100.0),
        "wind_speed": (0.0, 100.0),
    }

    for var_name, (lo, hi) in ranges.items():

        if var_name not in ds:
            continue

        arr = ds[var_name].values
        finite = arr[np.isfinite(arr)]

        below = int(np.sum(finite < lo))
        above = int(np.sum(finite > hi))

        print(f"\n{var_name}")
        print(f"  Allowed range: [{lo}, {hi}]")
        print(f"  Below range:   {below:,}")
        print(f"  Above range:   {above:,}")

        if below == 0 and above == 0:
            print("  Result: PASS")
        else:
            print("  Result: FAIL")

    # ------------------------------------------------------------------------
    # Wind speed consistency
    # ------------------------------------------------------------------------

    subsection("12. WIND SPEED CONSISTENCY")

    calculated_speed = np.sqrt(u.astype(np.float64) ** 2 + v.astype(np.float64) ** 2)

    provided_speed = ds["wind_speed"].values.astype(np.float64)

    valid = (
        np.isfinite(calculated_speed)
        & np.isfinite(provided_speed)
    )

    if np.any(valid):

        abs_error = np.abs(
            calculated_speed[valid] - provided_speed[valid]
        )

        print(f"Valid points: {valid.sum():,}")
        print(f"Maximum absolute error: {abs_error.max()}")
        print(f"Mean absolute error:    {abs_error.mean()}")

        if abs_error.max() < 1e-3:
            print("Result: PASS")
        else:
            print("Result: CHECK")

    # ------------------------------------------------------------------------
    # Geographic derivative spacing
    # ------------------------------------------------------------------------

    subsection("13. GEOGRAPHIC DERIVATIVE SPACING")

    # Earth's mean radius
    R = 6_371_000.0

    lat_rad = np.deg2rad(lat)

    # dx depends on latitude
    dx = R * np.cos(lat_rad) * np.deg2rad(EXPECTED_RESOLUTION)

    # dy is approximately constant for constant latitude spacing
    dy = R * np.deg2rad(EXPECTED_RESOLUTION)

    print(
        f"Latitude resolution:  {EXPECTED_RESOLUTION} degrees"
    )

    print(
        f"Longitude resolution: {EXPECTED_RESOLUTION} degrees"
    )

    print(
        f"\nMeridional spacing dy: "
        f"{dy:,.2f} m"
    )

    print(
        "\nZonal spacing dx:"
    )

    print(
        f"  At {lat[0]:.2f}°N:  {dx[0]:,.2f} m"
    )

    mid = len(lat) // 2

    print(
        f"  At {lat[mid]:.2f}°N: {dx[mid]:,.2f} m"
    )

    print(
        f"  At {lat[-1]:.2f}°N: {dx[-1]:,.2f} m"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  dx varies with latitude."
    )

    print(
        "  The curl implementation must NOT use the same"
        " x-distance for every latitude."
    )

    # ------------------------------------------------------------------------
    # Synthetic derivative test
    # ------------------------------------------------------------------------

    subsection("14. NUMERICAL DIFFERENTIATION TEST")

    print(
        "Testing xarray.differentiate() on a known analytical field."
    )

    test_lat = np.linspace(5.0, 30.0, 101)
    test_lon = np.linspace(45.0, 105.0, 241)

    test_da = xr.DataArray(
        np.outer(test_lat, np.ones_like(test_lon)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": test_lat,
            "longitude": test_lon,
        },
    )

    # d(lat)/d(lat) should equal 1
    derivative_lat = test_da.differentiate("latitude")

    interior_lat_error = np.abs(
        derivative_lat.values[1:-1, :] - 1.0
    )

    edge_lat_error = np.abs(
        np.concatenate(
            [
                derivative_lat.values[0:1, :],
                derivative_lat.values[-1:, :],
            ],
            axis=0,
        ) - 1.0
    )

    print(
        f"\nLatitude derivative interior max error: "
        f"{interior_lat_error.max():.6e}"
    )

    print(
        f"Latitude derivative boundary max error: "
        f"{edge_lat_error.max():.6e}"
    )

    print(
        "\nInterpretation:"
    )

    print(
        "  Interior points use centered finite differences."
    )

    print(
        "  Boundary points use one-sided finite differences."
    )

    # ------------------------------------------------------------------------
    # Boundary diagnostics
    # ------------------------------------------------------------------------

    subsection("15. DOMAIN EDGE DIAGNOSTIC")

    print(
        "Domain boundaries:"
    )

    print(f"  South: {lat[0]}°N")
    print(f"  North: {lat[-1]}°N")
    print(f"  West:  {lon[0]}°E")
    print(f"  East:  {lon[-1]}°E")

    for var_name in ["wind_u", "wind_v"]:

        arr = ds[var_name].values

        print(f"\n{var_name} boundary statistics:")

        boundaries = {
            "south": arr[:, 0, :],
            "north": arr[:, -1, :],
            "west": arr[:, :, 0],
            "east": arr[:, :, -1],
        }

        for name, boundary in boundaries.items():

            finite = boundary[np.isfinite(boundary)]

            if len(finite):

                print(
                    f"  {name:6s}: "
                    f"min={finite.min():8.3f}, "
                    f"max={finite.max():8.3f}, "
                    f"mean={finite.mean():8.3f}, "
                    f"NaNs={np.isnan(boundary).sum():,}"
                )

    # ------------------------------------------------------------------------
    # Derivative readiness
    # ------------------------------------------------------------------------

    subsection("16. DERIVATIVE READINESS")

    readiness_checks = []

    check = (
        set(actual_vars) == set(EXPECTED_VARS)
    )
    readiness_checks.append(("Required variables", check))

    check = (
        np.all(lat_spacing > 0)
        and np.allclose(lat_spacing, EXPECTED_RESOLUTION)
    )
    readiness_checks.append(("Latitude grid", check))

    check = (
        np.all(lon_spacing > 0)
        and np.allclose(lon_spacing, EXPECTED_RESOLUTION)
    )
    readiness_checks.append(("Longitude grid", check))

    check = (
        not np.isinf(u).any()
        and not np.isinf(v).any()
    )
    readiness_checks.append(("No infinite U/V values", check))

    check = (
        masks_identical
    )
    readiness_checks.append(("U/V missing masks identical", check))

    check = (
        abs_error.max() < 1e-3
    )
    readiness_checks.append(("Wind-speed consistency", check))

    for name, result in readiness_checks:
        print(
            f"  {name:35s}: "
            f"{'PASS' if result else 'CHECK'}"
        )

    # ------------------------------------------------------------------------
    # Final recommendation
    # ------------------------------------------------------------------------

    subsection("17. AUTOMATED RECOMMENDATION")

    permanent_nan_count = int(permanent_mask.sum())
    total_spatial = permanent_mask.size

    print(
        f"Permanent NaN spatial cells: "
        f"{permanent_nan_count:,} / {total_spatial:,}"
    )

    if permanent_nan_count == 0:

        print(
            "\nWARNING:"
        )

        print(
            "  No permanently masked spatial cells were detected."
        )

        print(
            "  Do NOT assume the domain is ocean-only."
        )

        print(
            "  The wind dataset appears fully populated spatially."
        )

        print(
            "  A land-sea mask may therefore be required before"
            " calculating stress curl."
        )

    else:

        print(
            "\nPermanent missing spatial cells detected."
        )

        print(
            "  These may represent land/masked regions."
        )

        print(
            "  Their spatial structure must be inspected before"
            " computing curl."
        )

    if np.isnan(u).sum() == 0 and np.isnan(v).sum() == 0:

        print(
            "\nU/V contain no NaNs."
        )

        print(
            "  Therefore land is NOT explicitly represented as NaN"
            " in the current wind dataset."
        )

        print(
            "  Do not differentiate across the raw grid blindly."
        )

    print(
        "\nNEXT STEP:"
    )

    print(
        "  Review this diagnostic output."
    )

    print(
        "  If land masking is required, establish the mask first."
    )

    print(
        "  Then implement wind-stress calculation and geographic"
        " curl using latitude-dependent dx and constant dy."
    )

    ds.close()

    print()
    print("=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()