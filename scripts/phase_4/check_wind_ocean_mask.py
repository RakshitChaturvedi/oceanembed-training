from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training"
)

SST_PATH = ROOT / "data/processed/phase2/sst.nc"
WIND_PATH = ROOT / "data/processed/phase2/wind.nc"


# =============================================================================
# CONFIG
# =============================================================================

ZERO_TOL = 1e-12


# =============================================================================
# HELPERS
# =============================================================================

def print_section(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def describe_mask(name, mask, lat, lon):
    """
    Print count, percentage and coordinate bounds for a boolean spatial mask.
    """
    count = int(mask.sum().item())
    total = mask.size
    pct = 100.0 * count / total

    print(f"\n{name}")
    print(f"  Count:      {count:,}")
    print(f"  Percentage: {pct:.4f}%")

    if count > 0:
        y, x = np.where(mask.values)

        print(
            f"  Latitude:   {float(lat.values[y].min()):.2f}"
            f" -> {float(lat.values[y].max()):.2f}"
        )
        print(
            f"  Longitude:  {float(lon.values[x].min()):.2f}"
            f" -> {float(lon.values[x].max()):.2f}"
        )

    return count


# =============================================================================
# LOAD
# =============================================================================

print_section("WIND / SST OCEAN MASK DIAGNOSTIC")

print(f"SST : {SST_PATH}")
print(f"WIND: {WIND_PATH}")

sst_ds = xr.open_dataset(SST_PATH)
wind_ds = xr.open_dataset(WIND_PATH)

print("\nDatasets opened successfully.")

print("\nSST variables :", list(sst_ds.data_vars))
print("Wind variables:", list(wind_ds.data_vars))


# =============================================================================
# BASIC ALIGNMENT CHECK
# =============================================================================

print_section("1. GRID ALIGNMENT")

for coord in ["time", "latitude", "longitude"]:
    sst_coord = sst_ds[coord]
    wind_coord = wind_ds[coord]

    identical = np.array_equal(
        sst_coord.values,
        wind_coord.values,
    )

    print(
        f"{coord:10s}: "
        f"{'PASS' if identical else 'FAIL'}"
    )

    if not identical:
        raise RuntimeError(
            f"SST and wind {coord} coordinates do not match."
        )


# =============================================================================
# LOAD VARIABLES
# =============================================================================

sst = sst_ds["sst"]
wind_u = wind_ds["wind_u"]
wind_v = wind_ds["wind_v"]

lat = sst_ds["latitude"]
lon = sst_ds["longitude"]


# =============================================================================
# 2. SST PERMANENTLY MISSING MASK
# =============================================================================

print_section("2. SST PERMANENTLY MISSING MASK")

sst_nan = sst.isnull()

# A cell is permanently missing if it is NaN for every day.
sst_permanent_nan = sst_nan.all(dim="time")

# A cell has at least one valid observation.
sst_ever_valid = sst_nan.any(dim="time") == False

describe_mask(
    "SST permanently NaN cells",
    sst_permanent_nan,
    lat,
    lon,
)

describe_mask(
    "SST cells with at least one valid observation",
    sst_ever_valid,
    lat,
    lon,
)


# =============================================================================
# 3. WIND PERMANENT ZERO MASK
# =============================================================================

print_section("3. WIND PERMANENT ZERO MASK")

u_zero = abs(wind_u) <= ZERO_TOL
v_zero = abs(wind_v) <= ZERO_TOL

# Both components are zero on a given day.
uv_zero = u_zero & v_zero

# Permanently (0,0) across the entire time series.
wind_permanent_zero = uv_zero.all(dim="time")

# Has at least one nonzero observation.
wind_ever_nonzero = (~uv_zero).any(dim="time")

# Has (0,0) on at least one day.
wind_ever_zero = uv_zero.any(dim="time")

# Intermittently zero:
# zero at least once, but NOT zero for the entire period.
intermittent_zero = wind_ever_zero & ~wind_permanent_zero

describe_mask(
    "Wind cells permanently (U,V) = (0,0)",
    wind_permanent_zero,
    lat,
    lon,
)

describe_mask(
    "Wind cells with at least one nonzero observation",
    wind_ever_nonzero,
    lat,
    lon,
)


# =============================================================================
# 4. MASK OVERLAP
# =============================================================================

print_section("4. SST / WIND MASK OVERLAP")

wind_zero = wind_permanent_zero
sst_land = sst_permanent_nan

both = wind_zero & sst_land
wind_zero_sst_valid = wind_zero & ~sst_land
sst_nan_wind_nonzero = sst_land & ~wind_zero

describe_mask(
    "A. Wind permanently zero AND SST permanently NaN",
    both,
    lat,
    lon,
)

describe_mask(
    "B. Wind permanently zero BUT SST has valid data",
    wind_zero_sst_valid,
    lat,
    lon,
)

describe_mask(
    "C. SST permanently NaN BUT wind is not permanently zero",
    sst_nan_wind_nonzero,
    lat,
    lon,
)


# =============================================================================
# 5. AGREEMENT STATISTICS
# =============================================================================

print_section("5. MASK AGREEMENT")

wind_zero_count = int(wind_zero.sum().item())
sst_nan_count = int(sst_land.sum().item())
overlap_count = int(both.sum().item())

print(f"Wind permanently zero cells : {wind_zero_count:,}")
print(f"SST permanently NaN cells   : {sst_nan_count:,}")
print(f"Overlap                     : {overlap_count:,}")

if wind_zero_count > 0:
    print(
        "\nOf permanently-zero wind cells,"
        f" {100.0 * overlap_count / wind_zero_count:.2f}%"
        " are permanently NaN in SST."
    )

if sst_nan_count > 0:
    print(
        "Of permanently-NaN SST cells,"
        f" {100.0 * overlap_count / sst_nan_count:.2f}%"
        " are permanently-zero in wind."
    )


# =============================================================================
# 6. COASTLINE / SPATIAL COHERENCE CHECK
# =============================================================================

print_section("6. SPATIAL COHERENCE OF WIND-ZERO MASK")

mask = wind_zero.values

height, width = mask.shape

neighbor_counts = np.zeros_like(mask, dtype=np.int8)

# North
neighbor_counts[1:, :] += mask[:-1, :]

# South
neighbor_counts[:-1, :] += mask[1:, :]

# West
neighbor_counts[:, 1:] += mask[:, :-1]

# East
neighbor_counts[:, :-1] += mask[:, 1:]

isolated = mask & (neighbor_counts == 0)
clustered = mask & (neighbor_counts > 0)

isolated_count = int(isolated.sum())
clustered_count = int(clustered.sum())

print(f"Wind-zero cells:        {int(mask.sum()):,}")
print(f"Clustered cells:        {clustered_count:,}")
print(f"Isolated cells:         {isolated_count:,}")

if mask.sum() > 0:
    print(
        f"Clustered fraction:     "
        f"{100.0 * clustered_count / mask.sum():.2f}%"
    )
    print(
        f"Isolated fraction:      "
        f"{100.0 * isolated_count / mask.sum():.2f}%"
    )


# =============================================================================
# 7. COORDINATES OF NON-OVERLAPPING CASES
# =============================================================================

print_section("7. NON-OVERLAPPING MASK COORDINATES")

y, x = np.where(wind_zero_sst_valid.values)

if len(y) == 0:
    print(
        "No cells found where wind is permanently zero "
        "but SST is not permanently NaN."
    )
else:
    print(
        f"Found {len(y):,} cells where wind is permanently zero "
        "but SST has valid data."
    )

    print("\nFirst 30:")
    for i in range(min(30, len(y))):
        print(
            f"  lat={float(lat.values[y[i]]):6.2f}, "
            f"lon={float(lon.values[x[i]]):7.2f}"
        )

y, x = np.where(sst_nan_wind_nonzero.values)

if len(y) == 0:
    print(
        "\nNo cells found where SST is permanently NaN "
        "but wind is not permanently zero."
    )
else:
    print(
        f"\nFound {len(y):,} cells where SST is permanently NaN "
        "but wind is not permanently zero."
    )

    print("\nFirst 30:")
    for i in range(min(30, len(y))):
        print(
            f"  lat={float(lat.values[y[i]]):6.2f}, "
            f"lon={float(lon.values[x[i]]):7.2f}"
        )


# =============================================================================
# 8. TEMPORAL WIND ZERO DIAGNOSTIC
# =============================================================================

print_section("8. TEMPORAL WIND ZERO DIAGNOSTIC")


describe_mask(
    "Cells that are (0,0) on at least one day",
    wind_ever_zero,
    lat,
    lon,
)

describe_mask(
    "Cells intermittently (0,0)",
    intermittent_zero,
    lat,
    lon,
)


# =============================================================================
# 9. WIND NaN / INF CHECK
# =============================================================================

print_section("9. RAW WIND NaN / INF CHECK")

for name, da in [
    ("wind_u", wind_u),
    ("wind_v", wind_v),
]:

    nan_count = int(da.isnull().sum().item())
    inf_count = int(
        np.isinf(da.values).sum()
    )

    print(f"\n{name}")
    print(f"  NaNs: {nan_count:,}")
    print(f"  Infs: {inf_count:,}")


# =============================================================================
# 10. FINAL RECOMMENDATION
# =============================================================================

print_section("10. AUTOMATED RECOMMENDATION")

if (
    wind_zero_count > 0
    and overlap_count / max(wind_zero_count, 1) >= 0.95
    and len(np.where(wind_zero_sst_valid.values)[0]) == 0
):
    print(
        "SAFE TO PROCEED WITH MASKING."
    )
    print(
        "The permanently-zero wind cells correspond to "
        "permanently-missing SST cells."
    )
    print(
        "Use the permanent SST-NaN mask as the ocean/land exclusion "
        "mask before calculating spatial derivatives."
    )

elif wind_zero_count == 0:
    print(
        "NO PERMANENT WIND-ZERO MASK FOUND."
    )
    print(
        "Inspect wind metadata / another ocean mask before curl."
    )

else:
    print(
        "DO NOT PROCEED WITH CURL YET."
    )
    print(
        "The SST and wind masks do not agree sufficiently."
    )
    print(
        "Inspect the non-overlapping cells before deciding how "
        "to construct the derivative mask."
    )


# =============================================================================
# CLEANUP
# =============================================================================

sst_ds.close()
wind_ds.close()

print_section("DIAGNOSTIC COMPLETE")