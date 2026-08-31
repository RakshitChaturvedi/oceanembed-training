#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]

CURL_FILE = ROOT / "data/processed/phase4/wind_stress_curl.nc"
MASK_FILE = ROOT / "data/processed/phase5/ocean_mask.nc"


def main():
    print("=" * 100)
    print("OceanEmbed — Curl NaN Geometry Check")
    print("=" * 100)

    curl_ds = xr.open_dataset(CURL_FILE)
    mask_ds = xr.open_dataset(MASK_FILE)

    curl = curl_ds["wind_stress_curl"].transpose(
        "time", "latitude", "longitude"
    )

    ocean_mask = mask_ds["ocean_mask"].transpose(
        "latitude", "longitude"
    )

    # ------------------------------------------------------------------
    # Curl NaNs inside common ocean
    # ------------------------------------------------------------------

    curl_nan = np.isnan(curl.values)
    ocean = ocean_mask.values.astype(bool)

    ocean_nan = curl_nan & ocean[None, :, :]

    print()
    print("BASIC COUNTS")
    print("-" * 100)

    total_nan = int(curl_nan.sum())
    ocean_nan_count = int(ocean_nan.sum())

    print(f"Total curl NaNs                 : {total_nan:,}")
    print(f"Curl NaNs inside common ocean   : {ocean_nan_count:,}")

    # ------------------------------------------------------------------
    # Spatial frequency of ocean curl NaNs
    # ------------------------------------------------------------------

    spatial_nan_count = ocean_nan.sum(axis=0)

    ocean_cell_count = int(ocean.sum())

    intermittent_cells = int(
        ((spatial_nan_count > 0) & ocean).sum()
    )

    complete_ocean_cells = int(
        ((spatial_nan_count == 0) & ocean).sum()
    )

    print()
    print("SPATIAL DISTRIBUTION")
    print("-" * 100)

    print(f"Common ocean cells              : {ocean_cell_count:,}")
    print(f"Ocean cells ever NaN in curl    : {intermittent_cells:,}")
    print(f"Ocean cells never NaN in curl   : {complete_ocean_cells:,}")

    # ------------------------------------------------------------------
    # Categorize by missingness fraction
    # ------------------------------------------------------------------

    print()
    print("MISSINGNESS FRACTION OF OCEAN CELLS")
    print("-" * 100)

    fractions = np.zeros_like(spatial_nan_count, dtype=np.float64)
    fractions[ocean] = spatial_nan_count[ocean] / curl.sizes["time"]

    bins = [
        (0.0, 0.0, "0%"),
        (0.0, 0.01, "(0%, 1%]"),
        (0.01, 0.10, "(1%, 10%]"),
        (0.10, 0.50, "(10%, 50%]"),
        (0.50, 0.90, "(50%, 90%]"),
        (0.90, 1.00, "(90%, 100%]"),
        (1.00, 1.01, "100%"),
    ]

    for lo, hi, label in bins:
        if label == "0%":
            count = int(((fractions == 0) & ocean).sum())
        elif label == "100%":
            count = int(((fractions == 1) & ocean).sum())
        else:
            count = int(
                ((fractions > lo) & (fractions <= hi) & ocean).sum()
            )

        print(f"{label:12s}: {count:,}")

    # ------------------------------------------------------------------
    # Interior vs edge test
    #
    # A curl NaN is considered "near a mask boundary" if any of its
    # 8 neighboring spatial cells is permanently unavailable.
    # ------------------------------------------------------------------

    print()
    print("SPATIAL BOUNDARY TEST")
    print("-" * 100)

    padded = np.pad(
        ~ocean,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=True,
    )

    adjacent_to_unavailable = np.zeros_like(ocean, dtype=bool)

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue

            adjacent_to_unavailable |= (
                padded[
                    1 + di : 1 + di + ocean.shape[0],
                    1 + dj : 1 + dj + ocean.shape[1],
                ]
            )

    curl_nan_boundary = (
        ocean_nan
        & adjacent_to_unavailable[None, :, :]
    )

    curl_nan_interior = (
        ocean_nan
        & ~adjacent_to_unavailable[None, :, :]
    )

    boundary_count = int(curl_nan_boundary.sum())
    interior_count = int(curl_nan_interior.sum())

    print(f"Curl NaNs near unavailable cells : {boundary_count:,}")
    print(f"Curl NaNs away from boundaries   : {interior_count:,}")

    if ocean_nan_count:
        print(
            f"Boundary fraction               : "
            f"{100.0 * boundary_count / ocean_nan_count:.3f}%"
        )

    # ------------------------------------------------------------------
    # Spatial examples
    # ------------------------------------------------------------------

    print()
    print("TOP 25 OCEAN CELLS BY CURL MISSINGNESS")
    print("-" * 100)

    lat = curl["latitude"].values
    lon = curl["longitude"].values

    candidates = []

    for i, j in zip(*np.where(ocean & (spatial_nan_count > 0))):
        candidates.append(
            (
                int(spatial_nan_count[i, j]),
                float(lat[i]),
                float(lon[j]),
                bool(adjacent_to_unavailable[i, j]),
            )
        )

    candidates.sort(reverse=True)

    for rank, (count, la, lo, boundary) in enumerate(
        candidates[:25],
        start=1,
    ):
        frac = 100.0 * count / curl.sizes["time"]

        print(
            f"{rank:2d}. "
            f"lat={la:7.3f} "
            f"lon={lo:8.3f} "
            f"missing={count:4d}/{curl.sizes['time']} "
            f"({frac:6.2f}%) "
            f"boundary={'YES' if boundary else 'NO'}"
        )

    curl_ds.close()
    mask_ds.close()

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()