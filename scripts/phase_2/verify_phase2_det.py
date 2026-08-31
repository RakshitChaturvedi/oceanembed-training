from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "data" / "processed" / "phase2"


FILES = [
    "sst.nc",
    "sss.nc",
    "ssh.nc",
    "wind.nc",
    "currents.nc",
    "armor3d.nc",
    "glorys.nc",
]

SURFACE_FILES = [
    "sst.nc",
    "sss.nc",
    "ssh.nc",
    "wind.nc",
    "currents.nc",
]

SUBSURFACE_FILES = [
    "armor3d.nc",
    "glorys.nc",
]


# ============================================================================
# Expected global configuration
# ============================================================================

EXPECTED_START = pd.Timestamp("2021-01-01")
EXPECTED_END = pd.Timestamp("2024-11-19")

EXPECTED_TIME_COUNT = 1419

EXPECTED_LAT_MIN = 5.0
EXPECTED_LAT_MAX = 30.0

EXPECTED_LON_MIN = 45.0
EXPECTED_LON_MAX = 105.0

EXPECTED_RESOLUTION = 0.25

EXPECTED_LAT_COUNT = 101
EXPECTED_LON_COUNT = 241


# Number of time steps loaded at once for large datasets.
#
# armor3d:
#   16 days × 36 depths × 101 × 241
#   ≈ 14 million values
#   ≈ 110 MB for float64
#
# This avoids loading the ~20 GB dataset into RAM.
VALIDATION_TIME_CHUNK = 16


# ============================================================================
# Expected variables
# ============================================================================

EXPECTED_VARIABLES = {
    "sst.nc": ["sst"],
    "sss.nc": ["sss"],
    "ssh.nc": ["ssh"],
    "wind.nc": [
        "wind_u",
        "wind_v",
        "wind_speed",
    ],
    "currents.nc": [
        "current_u",
        "current_v",
    ],
    "armor3d.nc": [
        "temperature",
        "salinity",
    ],
    "glorys.nc": [
        "temperature",
        "salinity",
    ],
}


# ============================================================================
# Physical sanity ranges
#
# These are intentionally broad.
# They are intended to catch corruption / impossible values,
# not reject legitimate oceanographic extremes.
# ============================================================================

PHYSICAL_RANGES = {
    "sst": {
        "min": 200.0,       # Kelvin
        "max": 330.0,
    },
    "sss": {
        "min": 0.0,
        "max": 50.0,
    },
    "ssh": {
        "min": -5.0,
        "max": 5.0,
    },
    "wind_u": {
        "min": -100.0,
        "max": 100.0,
    },
    "wind_v": {
        "min": -100.0,
        "max": 100.0,
    },
    "wind_speed": {
        "min": 0.0,
        "max": 100.0,
    },
    "current_u": {
        "min": -10.0,
        "max": 10.0,
    },
    "current_v": {
        "min": -10.0,
        "max": 10.0,
    },
    "temperature": {
        "min": -5.0,        # Celsius
        "max": 50.0,
    },
    "salinity": {
        "min": 0.0,
        "max": 50.0,
    },
}


# ============================================================================
# Helpers
# ============================================================================

def check_spacing(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return {
            "min": None,
            "max": None,
            "unique": [],
        }

    diffs = np.diff(values)

    return {
        "min": float(diffs.min()),
        "max": float(diffs.max()),
        "unique": np.unique(
            np.round(diffs, 8)
        ),
    }


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def close_enough(a, b, rtol=1e-6, atol=1e-6):
    return np.allclose(
        a,
        b,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )


# ============================================================================
# Basic validation
# ============================================================================

def validate_time(ds):
    time = pd.DatetimeIndex(ds.time.values)

    print("\n[TIME]")

    if len(time):
        print("Start:", time[0])
        print("End:  ", time[-1])

    print("Count:", len(time))

    duplicate_count = time.duplicated().sum()

    print("Duplicates:", duplicate_count)

    diffs = time.to_series().diff().dropna()

    print("\nTime intervals:")

    if len(diffs):
        print(diffs.value_counts().head(10))
    else:
        print("N/A")

    expected_days = pd.date_range(
        EXPECTED_START,
        EXPECTED_END,
        freq="D",
    )

    missing_dates = expected_days.difference(time)
    extra_dates = time.difference(expected_days)

    print("\nMissing dates:", len(missing_dates))

    if len(missing_dates):
        print(missing_dates[:20])

    print("Extra dates:", len(extra_dates))

    if len(extra_dates):
        print(extra_dates[:20])

    count_pass = len(time) == EXPECTED_TIME_COUNT

    range_pass = (
        len(time) > 0
        and time[0] == EXPECTED_START
        and time[-1] == EXPECTED_END
    )

    duplicate_pass = duplicate_count == 0
    missing_pass = len(missing_dates) == 0
    extra_pass = len(extra_dates) == 0

    print("\nTime validation:")
    print(
        "  Count:       ",
        pass_fail(count_pass),
    )
    print(
        "  Range:       ",
        pass_fail(range_pass),
    )
    print(
        "  Duplicates:  ",
        pass_fail(duplicate_pass),
    )
    print(
        "  Missing:     ",
        pass_fail(missing_pass),
    )
    print(
        "  Extra:       ",
        pass_fail(extra_pass),
    )

    return (
        count_pass
        and range_pass
        and duplicate_pass
        and missing_pass
        and extra_pass
    )


def validate_horizontal_grid(ds):
    lat = ds.latitude.values
    lon = ds.longitude.values

    lat_spacing = check_spacing(lat)
    lon_spacing = check_spacing(lon)

    lat_pass = (
        len(lat) == EXPECTED_LAT_COUNT
        and np.isclose(
            lat[0],
            EXPECTED_LAT_MIN,
        )
        and np.isclose(
            lat[-1],
            EXPECTED_LAT_MAX,
        )
        and np.allclose(
            np.diff(lat),
            EXPECTED_RESOLUTION,
        )
    )

    lon_pass = (
        len(lon) == EXPECTED_LON_COUNT
        and np.isclose(
            lon[0],
            EXPECTED_LON_MIN,
        )
        and np.isclose(
            lon[-1],
            EXPECTED_LON_MAX,
        )
        and np.allclose(
            np.diff(lon),
            EXPECTED_RESOLUTION,
        )
    )

    print("\n[LATITUDE]")
    print("First:", lat[0])
    print("Last: ", lat[-1])
    print("Count:", len(lat))
    print("Spacing:", lat_spacing)

    print("\n[LONGITUDE]")
    print("First:", lon[0])
    print("Last: ", lon[-1])
    print("Count:", len(lon))
    print("Spacing:", lon_spacing)

    print("\nHorizontal grid validation:")
    print(
        "  Latitude:  ",
        pass_fail(lat_pass),
    )
    print(
        "  Longitude: ",
        pass_fail(lon_pass),
    )

    return lat_pass and lon_pass


def validate_variables(ds, filename):
    print("\n[VARIABLES]")

    expected = EXPECTED_VARIABLES[filename]
    actual = list(ds.data_vars)

    variables_pass = set(actual) == set(expected)

    print("Expected:", expected)
    print("Actual:  ", actual)

    print(
        "Variable set:",
        pass_fail(variables_pass),
    )

    for name, var in ds.data_vars.items():

        print("\n" + "-" * 80)
        print(name)

        print("  dims:", var.dims)
        print("  shape:", var.shape)
        print("  dtype:", var.dtype)
        print(
            "  units:",
            var.attrs.get(
                "units",
                "N/A",
            ),
        )
        print(
            "  long_name:",
            var.attrs.get(
                "long_name",
                "N/A",
            ),
        )

    return variables_pass


# ============================================================================
# Chunked physical sanity
# ============================================================================

def validate_variable_physical_sanity(
    var,
    name,
    limits,
):
    """
    Validate a single variable without loading the entire array.

    For large 4-D subsurface variables this processes time in chunks.
    """

    total = int(np.prod(var.shape))

    nan_count = 0
    inf_count = 0
    finite_count = 0

    below_count = 0
    above_count = 0

    min_value = np.inf
    max_value = -np.inf

    # --------------------------------------------------------------
    # Determine time dimension
    # --------------------------------------------------------------

    if "time" in var.dims:

        time_axis = var.dims.index("time")
        time_count = var.sizes["time"]

        for start in range(
            0,
            time_count,
            VALIDATION_TIME_CHUNK,
        ):

            end = min(
                start + VALIDATION_TIME_CHUNK,
                time_count,
            )

            chunk = var.isel(
                time=slice(start, end)
            ).values

            finite_mask = np.isfinite(chunk)

            nan_count += int(
                np.isnan(chunk).sum()
            )

            inf_count += int(
                np.isinf(chunk).sum()
            )

            finite = chunk[finite_mask]

            if finite.size:

                finite_count += finite.size

                chunk_min = float(
                    np.min(finite)
                )

                chunk_max = float(
                    np.max(finite)
                )

                min_value = min(
                    min_value,
                    chunk_min,
                )

                max_value = max(
                    max_value,
                    chunk_max,
                )

                below_count += int(
                    np.sum(
                        finite < limits["min"]
                    )
                )

                above_count += int(
                    np.sum(
                        finite > limits["max"]
                    )
                )

    else:

        data = var.values

        finite_mask = np.isfinite(data)
        finite = data[finite_mask]

        nan_count = int(
            np.isnan(data).sum()
        )

        inf_count = int(
            np.isinf(data).sum()
        )

        finite_count = int(
            finite.size
        )

        if finite.size:

            min_value = float(
                np.min(finite)
            )

            max_value = float(
                np.max(finite)
            )

            below_count = int(
                np.sum(
                    finite < limits["min"]
                )
            )

            above_count = int(
                np.sum(
                    finite > limits["max"]
                )
            )

    if finite_count == 0:

        print(f"\n{name}")
        print("  No finite values.")
        print("  Result: FAIL")

        return False

    variable_pass = (
        inf_count == 0
        and below_count == 0
        and above_count == 0
    )

    print(f"\n{name}")
    print(
        f"  Allowed range: "
        f"[{limits['min']}, {limits['max']}]"
    )
    print(
        f"  Actual min:    {min_value}"
    )
    print(
        f"  Actual max:    {max_value}"
    )
    print(
        f"  NaNs:          {nan_count}"
    )
    print(
        f"  Infs:          {inf_count}"
    )
    print(
        f"  Below range:   {below_count}"
    )
    print(
        f"  Above range:   {above_count}"
    )
    print(
        f"  Result:        "
        f"{pass_fail(variable_pass)}"
    )

    return variable_pass


def validate_physical_sanity(ds):
    print("\n[PHYSICAL SANITY]")

    overall_pass = True

    for name, limits in PHYSICAL_RANGES.items():

        if name not in ds.data_vars:
            continue

        variable_pass = (
            validate_variable_physical_sanity(
                ds[name],
                name,
                limits,
            )
        )

        overall_pass &= variable_pass

    return overall_pass


# ============================================================================
# SST special diagnostics
# ============================================================================

def validate_sst_fill_values(ds):
    """
    SST has an important issue in the current Phase 2 output:

    0.0 K values are present and are outside the physical SST range.

    We do NOT simply change the allowed range to include 0 K.
    Instead, we explicitly report them.

    These may represent land / masked cells introduced during processing.
    """

    if "sst" not in ds.data_vars:
        return True

    print("\n[SST MASK / ZERO DIAGNOSTICS]")

    var = ds["sst"]

    zero_count = 0
    total_count = 0

    if "time" in var.dims:

        time_count = var.sizes["time"]

        for start in range(
            0,
            time_count,
            VALIDATION_TIME_CHUNK,
        ):

            end = min(
                start + VALIDATION_TIME_CHUNK,
                time_count,
            )

            chunk = var.isel(
                time=slice(start, end)
            ).values

            zero_count += int(
                np.sum(chunk == 0.0)
            )

            total_count += chunk.size

    else:

        data = var.values

        zero_count = int(
            np.sum(data == 0.0)
        )

        total_count = data.size

    zero_pct = (
        100.0 * zero_count / total_count
        if total_count
        else 0.0
    )

    print("Zero values:", zero_count)
    print(f"Zero %:      {zero_pct:.6f}%")

    print(
        "\nInterpretation:"
    )
    print(
        "  0 K is physically invalid for ocean SST."
    )
    print(
        "  These values should be treated as a masking/"
    )
    print(
        "  fill-value issue rather than legitimate SST."
    )

    # This is diagnostic only.
    #
    # Physical sanity remains responsible for failing the dataset
    # until the processing pipeline handles these values correctly.
    return True


# ============================================================================
# NaN / missing-value diagnostics
# ============================================================================

def validate_missing_values(ds):
    print("\n[MISSING VALUE DIAGNOSTICS]")

    for name, var in ds.data_vars.items():

        total = int(
            np.prod(var.shape)
        )

        nan_count = 0
        inf_count = 0

        # --------------------------------------------------------------
        # Process time-dependent variables in chunks.
        # --------------------------------------------------------------

        if "time" in var.dims:

            time_count = var.sizes["time"]

            always_nan = None
            never_nan = None

            for start in range(
                0,
                time_count,
                VALIDATION_TIME_CHUNK,
            ):

                end = min(
                    start + VALIDATION_TIME_CHUNK,
                    time_count,
                )

                chunk = var.isel(
                    time=slice(start, end)
                ).values

                nan_count += int(
                    np.isnan(chunk).sum()
                )

                inf_count += int(
                    np.isinf(chunk).sum()
                )

                # Spatial persistence.
                #
                # Reduce every non-time dimension.
                time_axis = var.dims.index("time")

                if time_axis != 0:
                    chunk_nan = np.moveaxis(
                        np.isnan(chunk),
                        time_axis,
                        0,
                    )
                else:
                    chunk_nan = np.isnan(chunk)

                chunk_always = np.all(
                    chunk_nan,
                    axis=0,
                )

                chunk_never = np.all(
                    ~chunk_nan,
                    axis=0,
                )

                if always_nan is None:
                    always_nan = chunk_always
                    never_nan = chunk_never
                else:
                    always_nan &= chunk_always
                    never_nan &= chunk_never

            finite_count = (
                total
                - nan_count
                - inf_count
            )

            nan_pct = (
                100.0 * nan_count / total
                if total
                else 0.0
            )

            print(f"\n{name}")
            print(f"  Total:       {total}")
            print(f"  NaNs:        {nan_count}")
            print(f"  NaN %:       {nan_pct:.4f}")
            print(f"  Infs:        {inf_count}")
            print(f"  Finite:      {finite_count}")

            if always_nan is not None:

                spatial_cells = always_nan.size

                print(
                    "  NaN spatial distribution:"
                )

                print(
                    f"    Always NaN:    "
                    f"{always_nan.sum()} "
                    f"("
                    f"{100 * always_nan.sum() / spatial_cells:.2f}"
                    f"%)"
                )

                print(
                    f"    Never NaN:     "
                    f"{never_nan.sum()} "
                    f"("
                    f"{100 * never_nan.sum() / spatial_cells:.2f}"
                    f"%)"
                )

                sometimes_nan = (
                    ~always_nan
                    & ~never_nan
                )

                print(
                    f"    Sometimes NaN: "
                    f"{sometimes_nan.sum()} "
                    f"("
                    f"{100 * sometimes_nan.sum() / spatial_cells:.2f}"
                    f"%)"
                )

        else:

            data = var.values

            nan_count = int(
                np.isnan(data).sum()
            )

            inf_count = int(
                np.isinf(data).sum()
            )

            finite_count = (
                total
                - nan_count
                - inf_count
            )

            nan_pct = (
                100.0 * nan_count / total
                if total
                else 0.0
            )

            print(f"\n{name}")
            print(f"  Total:       {total}")
            print(f"  NaNs:        {nan_count}")
            print(f"  NaN %:       {nan_pct:.4f}")
            print(f"  Infs:        {inf_count}")
            print(f"  Finite:      {finite_count}")


# ============================================================================
# Wind consistency
# ============================================================================

def validate_wind_consistency(ds):
    print("\n[WIND CONSISTENCY]")

    required = [
        "wind_u",
        "wind_v",
        "wind_speed",
    ]

    if not all(
        name in ds.data_vars
        for name in required
    ):

        print("Required wind variables missing.")
        print("Result: FAIL")

        return False

    wind_u = ds["wind_u"]
    wind_v = ds["wind_v"]
    wind_speed = ds["wind_speed"]

    max_error = 0.0
    total_error = 0.0
    valid_count = 0

    time_count = ds.sizes["time"]

    for start in range(
        0,
        time_count,
        VALIDATION_TIME_CHUNK,
    ):

        end = min(
            start + VALIDATION_TIME_CHUNK,
            time_count,
        )

        u = wind_u.isel(
            time=slice(start, end)
        ).values.astype(np.float64)

        v = wind_v.isel(
            time=slice(start, end)
        ).values.astype(np.float64)

        actual = wind_speed.isel(
            time=slice(start, end)
        ).values.astype(np.float64)

        expected = np.sqrt(
            u ** 2 + v ** 2
        )

        valid = (
            np.isfinite(expected)
            & np.isfinite(actual)
        )

        if not np.any(valid):
            continue

        difference = np.abs(
            actual[valid]
            - expected[valid]
        )

        max_error = max(
            max_error,
            float(np.max(difference)),
        )

        total_error += float(
            np.sum(difference)
        )

        valid_count += int(
            valid.sum()
        )

    if valid_count == 0:

        print("No finite wind-speed values.")
        print("Result: FAIL")

        return False

    mean_error = (
        total_error / valid_count
    )

    # wind_speed is float32, so this is appropriate.
    consistent = (
        max_error <= 1e-5
    )

    print("Formula:")
    print(
        "  wind_speed = "
        "sqrt(wind_u² + wind_v²)"
    )

    print()
    print(
        "Maximum absolute error:",
        max_error,
    )

    print(
        "Mean absolute error:   ",
        mean_error,
    )

    print(
        "Valid points:           ",
        valid_count,
    )

    print(
        "Result:                 ",
        pass_fail(consistent),
    )

    return consistent


# ============================================================================
# Subsurface validation
# ============================================================================

def validate_subsurface(ds, filename):
    print("\n[SUBSURFACE VALIDATION]")

    overall_pass = True

    # ------------------------------------------------------------------
    # Required dimensions
    # ------------------------------------------------------------------

    required_dims = {
        "time",
        "depth",
        "latitude",
        "longitude",
    }

    dims_pass = required_dims.issubset(
        ds.dims
    )

    print("\nRequired dimensions:")
    print(" ", required_dims)
    print(
        "Result:",
        pass_fail(dims_pass),
    )

    if not dims_pass:
        return False

    # ------------------------------------------------------------------
    # Time
    # ------------------------------------------------------------------

    time = pd.DatetimeIndex(
        ds.time.values
    )

    time_pass = (
        len(time) == EXPECTED_TIME_COUNT
        and time[0] == EXPECTED_START
        and time[-1] == EXPECTED_END
        and not time.duplicated().any()
    )

    print("\nTime:")
    print("  Count:", len(time))
    print("  Start:", time[0])
    print("  End:  ", time[-1])
    print(
        "  Result:",
        pass_fail(time_pass),
    )

    overall_pass &= time_pass

    # ------------------------------------------------------------------
    # Depth
    # ------------------------------------------------------------------

    depth = ds.depth.values.astype(
        np.float64
    )

    depth_increasing = np.all(
        np.diff(depth) > 0
    )

    depth_nonnegative = np.all(
        depth >= 0
    )

    depth_finite = np.all(
        np.isfinite(depth)
    )

    depth_pass = (
        len(depth) > 0
        and depth_finite
        and depth_increasing
        and depth_nonnegative
    )

    print("\nDepth:")
    print("  Count:", len(depth))
    print("  First:", depth[0])
    print("  Last: ", depth[-1])
    print(
        "  Finite:             ",
        pass_fail(depth_finite),
    )
    print(
        "  Strictly increasing:",
        pass_fail(depth_increasing),
    )
    print(
        "  Non-negative:       ",
        pass_fail(depth_nonnegative),
    )

    print("  Levels:")
    print("   ", depth)

    print(
        "  Result:",
        pass_fail(depth_pass),
    )

    overall_pass &= depth_pass

    # ------------------------------------------------------------------
    # Depth spacing diagnostics
    # ------------------------------------------------------------------

    if len(depth) >= 2:

        depth_diffs = np.diff(depth)

        print("\nDepth spacing:")
        print(
            "  Min spacing:",
            float(depth_diffs.min()),
        )
        print(
            "  Max spacing:",
            float(depth_diffs.max()),
        )

    # ------------------------------------------------------------------
    # Variable dimensions
    # ------------------------------------------------------------------

    expected_dims = (
        "time",
        "depth",
        "latitude",
        "longitude",
    )

    print("\nVariable dimensions:")

    for name in [
        "temperature",
        "salinity",
    ]:

        if name not in ds.data_vars:

            print(
                f"  {name}: MISSING"
            )

            overall_pass = False
            continue

        actual_dims = ds[name].dims

        variable_pass = (
            actual_dims == expected_dims
        )

        print(
            f"  {name}: "
            f"{actual_dims} "
            f"-> "
            f"{pass_fail(variable_pass)}"
        )

        overall_pass &= variable_pass

    # ------------------------------------------------------------------
    # Horizontal grid
    # ------------------------------------------------------------------

    lat = ds.latitude.values
    lon = ds.longitude.values

    lat_pass = (
        len(lat) == EXPECTED_LAT_COUNT
        and np.isclose(
            lat[0],
            EXPECTED_LAT_MIN,
        )
        and np.isclose(
            lat[-1],
            EXPECTED_LAT_MAX,
        )
        and np.allclose(
            np.diff(lat),
            EXPECTED_RESOLUTION,
        )
    )

    lon_pass = (
        len(lon) == EXPECTED_LON_COUNT
        and np.isclose(
            lon[0],
            EXPECTED_LON_MIN,
        )
        and np.isclose(
            lon[-1],
            EXPECTED_LON_MAX,
        )
        and np.allclose(
            np.diff(lon),
            EXPECTED_RESOLUTION,
        )
    )

    print("\nHorizontal grid:")
    print(
        "  Latitude:",
        pass_fail(lat_pass),
    )
    print(
        "  Longitude:",
        pass_fail(lon_pass),
    )

    overall_pass &= lat_pass
    overall_pass &= lon_pass

    # ------------------------------------------------------------------
    # Subsurface coverage by depth
    #
    # This catches a dataset where deeper levels are completely empty.
    # Only a small spatial slice is reduced at a time.
    # ------------------------------------------------------------------

    print("\n[DEPTH COVERAGE]")

    coverage_pass = True

    for name in [
        "temperature",
        "salinity",
    ]:

        if name not in ds.data_vars:
            continue

        var = ds[name]

        depth_count = var.sizes["depth"]

        finite_by_depth = np.zeros(
            depth_count,
            dtype=np.int64,
        )

        total_by_depth = np.zeros(
            depth_count,
            dtype=np.int64,
        )

        time_count = var.sizes["time"]

        for start in range(
            0,
            time_count,
            VALIDATION_TIME_CHUNK,
        ):

            end = min(
                start + VALIDATION_TIME_CHUNK,
                time_count,
            )

            chunk = var.isel(
                time=slice(start, end)
            ).values

            finite = np.isfinite(chunk)

            # Dimensions are expected to be:
            # time, depth, latitude, longitude
            finite_counts = finite.sum(
                axis=(0, 2, 3)
            )

            cell_count = (
                chunk.shape[0]
                * chunk.shape[2]
                * chunk.shape[3]
            )

            finite_by_depth += (
                finite_counts
            )

            total_by_depth += cell_count

        coverage = (
            100.0
            * finite_by_depth
            / total_by_depth
        )

        print(f"\n{name}")

        print(
            "  Surface depth coverage:",
            f"{coverage[0]:.2f}%"
        )

        print(
            "  Deepest depth coverage:",
            f"{coverage[-1]:.2f}%"
        )

        empty_levels = np.sum(
            finite_by_depth == 0
        )

        print(
            "  Completely empty levels:",
            int(empty_levels),
        )

        if empty_levels > 0:
            coverage_pass = False

        print(
            "  Result:",
            pass_fail(
                empty_levels == 0
            ),
        )

    overall_pass &= coverage_pass

    # ------------------------------------------------------------------
    # Physical sanity
    # ------------------------------------------------------------------

    print("\n[SUBSURFACE PHYSICAL SANITY]")

    physical_pass = True

    for name in [
        "temperature",
        "salinity",
    ]:

        if name not in ds.data_vars:
            physical_pass = False
            continue

        limits = PHYSICAL_RANGES[name]

        variable_pass = (
            validate_variable_physical_sanity(
                ds[name],
                name,
                limits,
            )
        )

        physical_pass &= variable_pass

    overall_pass &= physical_pass

    # ------------------------------------------------------------------
    # Depth-wise physical sanity
    #
    # We do NOT enforce a monotonic temperature/salinity profile.
    # That would be scientifically wrong because real ocean profiles
    # can contain inversions, mixed layers, fronts, etc.
    #
    # Instead, we check that every depth has at least some finite,
    # physically valid observations.
    # ------------------------------------------------------------------

    print("\n[DEPTH-WISE PHYSICAL COVERAGE]")

    depth_physical_pass = True

    for name in [
        "temperature",
        "salinity",
    ]:

        if name not in ds.data_vars:
            continue

        var = ds[name]
        limits = PHYSICAL_RANGES[name]

        depth_count = var.sizes["depth"]

        valid_by_depth = np.zeros(
            depth_count,
            dtype=np.int64,
        )

        total_by_depth = np.zeros(
            depth_count,
            dtype=np.int64,
        )

        time_count = var.sizes["time"]

        for start in range(
            0,
            time_count,
            VALIDATION_TIME_CHUNK,
        ):

            end = min(
                start + VALIDATION_TIME_CHUNK,
                time_count,
            )

            chunk = var.isel(
                time=slice(start, end)
            ).values

            valid = (
                np.isfinite(chunk)
                & (chunk >= limits["min"])
                & (chunk <= limits["max"])
            )

            valid_counts = valid.sum(
                axis=(0, 2, 3)
            )

            cell_count = (
                chunk.shape[0]
                * chunk.shape[2]
                * chunk.shape[3]
            )

            valid_by_depth += (
                valid_counts
            )

            total_by_depth += cell_count

        coverage = (
            100.0
            * valid_by_depth
            / total_by_depth
        )

        print(f"\n{name}")

        print(
            "  Minimum valid physical coverage:",
            f"{coverage.min():.2f}%"
        )

        print(
            "  Maximum valid physical coverage:",
            f"{coverage.max():.2f}%"
        )

        invalid_levels = np.sum(
            valid_by_depth == 0
        )

        print(
            "  Levels with zero valid values:",
            int(invalid_levels),
        )

        level_pass = (
            invalid_levels == 0
        )

        print(
            "  Result:",
            pass_fail(level_pass),
        )

        depth_physical_pass &= level_pass

    overall_pass &= depth_physical_pass

    print("\nSubsurface validation result:")
    print(
        " ",
        pass_fail(overall_pass),
    )

    return overall_pass


# ============================================================================
# Individual dataset inspection
# ============================================================================

def inspect_dataset(filename):
    path = PHASE2 / filename

    print("\n" + "=" * 100)
    print(f"FILE: {filename}")
    print(f"PATH: {path}")
    print("=" * 100)

    if not path.exists():

        print("FILE DOES NOT EXIST")

        return False

    try:
        ds = xr.open_dataset(path)
    except Exception as e:

        print(
            f"FAILED TO OPEN: {e}"
        )

        return False

    print("\nDATASET:")
    print(ds)

    results = []

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------

    results.append(
        validate_time(ds)
    )

    results.append(
        validate_horizontal_grid(ds)
    )

    results.append(
        validate_variables(
            ds,
            filename,
        )
    )

    # ------------------------------------------------------------------
    # Missing values
    # ------------------------------------------------------------------

    validate_missing_values(ds)

    # ------------------------------------------------------------------
    # Physical sanity
    # ------------------------------------------------------------------

    results.append(
        validate_physical_sanity(ds)
    )

    # ------------------------------------------------------------------
    # SST diagnostic
    # ------------------------------------------------------------------

    if filename == "sst.nc":

        validate_sst_fill_values(ds)

    # ------------------------------------------------------------------
    # Wind
    # ------------------------------------------------------------------

    if filename == "wind.nc":

        results.append(
            validate_wind_consistency(ds)
        )

    # ------------------------------------------------------------------
    # Subsurface
    # ------------------------------------------------------------------

    if filename in SUBSURFACE_FILES:

        # The generic physical check already ran above.
        #
        # validate_subsurface performs the deeper structural,
        # coverage and depth-specific checks.
        results.append(
            validate_subsurface(
                ds,
                filename,
            )
        )

    dataset_pass = all(results)

    print("\n" + "-" * 100)

    print(
        f"{filename} FINAL RESULT: "
        f"{pass_fail(dataset_pass)}"
    )

    print("-" * 100)

    ds.close()

    return dataset_pass


# ============================================================================
# Cross-dataset alignment
# ============================================================================

def validate_cross_dataset_alignment():
    print("\n" + "=" * 100)
    print("CROSS-DATASET ALIGNMENT")
    print("=" * 100)

    datasets = {}

    try:

        for filename in FILES:

            path = PHASE2 / filename

            if not path.exists():

                print(
                    f"{filename}: MISSING"
                )

                return False

            datasets[filename] = (
                xr.open_dataset(path)
            )

        reference_name = FILES[0]
        reference = datasets[
            reference_name
        ]

        overall_pass = True

        print(
            "\nReference dataset:",
            reference_name,
        )

        reference_time = (
            reference.time.values
        )

        reference_lat = (
            reference.latitude.values
        )

        reference_lon = (
            reference.longitude.values
        )

        for filename in FILES:

            ds = datasets[filename]

            time_match = np.array_equal(
                reference_time,
                ds.time.values,
            )

            lat_match = np.array_equal(
                reference_lat,
                ds.latitude.values,
            )

            lon_match = np.array_equal(
                reference_lon,
                ds.longitude.values,
            )

            dataset_pass = (
                time_match
                and lat_match
                and lon_match
            )

            print(f"\n{filename}")

            print(
                "  Time:      ",
                pass_fail(time_match),
            )

            print(
                "  Latitude:  ",
                pass_fail(lat_match),
            )

            print(
                "  Longitude: ",
                pass_fail(lon_match),
            )

            print(
                "  Overall:   ",
                pass_fail(dataset_pass),
            )

            overall_pass &= dataset_pass

        print(
            "\nCross-dataset alignment result:"
        )

        print(
            " ",
            pass_fail(overall_pass),
        )

        return overall_pass

    finally:

        for ds in datasets.values():
            ds.close()


# ============================================================================
# Main
# ============================================================================

def main():

    dataset_results = {}

    # ------------------------------------------------------------------
    # Individual datasets
    # ------------------------------------------------------------------

    for filename in FILES:

        path = PHASE2 / filename

        if not path.exists():

            print("\n" + "=" * 100)
            print(
                f"{filename}: "
                "NOT YET AVAILABLE"
            )
            print("=" * 100)

            dataset_results[filename] = False

            continue

        dataset_results[filename] = (
            inspect_dataset(filename)
        )

    # ------------------------------------------------------------------
    # Cross-dataset validation
    #
    # Only run when all seven datasets exist.
    # ------------------------------------------------------------------

    all_files_exist = all(
        (
            PHASE2 / filename
        ).exists()
        for filename in FILES
    )

    if all_files_exist:

        cross_alignment_pass = (
            validate_cross_dataset_alignment()
        )

    else:

        print("\n" + "=" * 100)
        print("CROSS-DATASET ALIGNMENT")
        print("=" * 100)

        print(
            "Skipped: not all seven "
            "Phase 2 files exist yet."
        )

        cross_alignment_pass = None

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n\n" + "=" * 100)
    print("PHASE 2 VALIDATION SUMMARY")
    print("=" * 100)

    for filename, result in (
        dataset_results.items()
    ):

        status = (
            "PASS"
            if result
            else "FAIL / INCOMPLETE"
        )

        print(
            f"{filename:15s}: {status}"
        )

    print()

    if cross_alignment_pass is True:

        print(
            f"{'Cross-dataset alignment':15s}: "
            "PASS"
        )

    elif cross_alignment_pass is False:

        print(
            f"{'Cross-dataset alignment':15s}: "
            "FAIL"
        )

    else:

        print(
            f"{'Cross-dataset alignment':15s}: "
            "NOT RUN"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()