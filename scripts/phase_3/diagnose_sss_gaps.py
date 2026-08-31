import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def diagnose(path: str):
    print("=" * 90)
    print("SSS TEMPORAL GAP DIAGNOSTIC")
    print("=" * 90)
    print(f"File: {path}\n")

    ds = xr.open_dataset(path)

    if "sss" not in ds:
        raise ValueError("Dataset does not contain variable 'sss'")

    da = ds["sss"]

    print("Dataset:")
    print(ds)
    print()

    values = da.values

    # Expected shape: (time, latitude, longitude)
    if values.ndim != 3:
        raise ValueError(
            f"Expected 3D SSS array (time, latitude, longitude), "
            f"got shape {values.shape}"
        )

    time = pd.DatetimeIndex(ds["time"].values)
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    n_time, n_lat, n_lon = values.shape
    n_cells = n_lat * n_lon

    valid = np.isfinite(values)

    # ------------------------------------------------------------------
    # 1. CLASSIFY EVERY SPATIAL CELL
    # ------------------------------------------------------------------

    has_valid = valid.any(axis=0)
    has_missing = (~valid).any(axis=0)

    permanent_missing = ~has_valid
    never_missing = has_valid & ~has_missing
    intermittent = has_valid & has_missing

    print("-" * 90)
    print("1. SPATIAL CELL CLASSIFICATION")
    print("-" * 90)

    print(f"Total spatial cells:       {n_cells:,}")
    print(
        f"Permanent missing:         "
        f"{permanent_missing.sum():,} "
        f"({permanent_missing.mean() * 100:.2f}%)"
    )
    print(
        f"Never missing:             "
        f"{never_missing.sum():,} "
        f"({never_missing.mean() * 100:.2f}%)"
    )
    print(
        f"Intermittently missing:    "
        f"{intermittent.sum():,} "
        f"({intermittent.mean() * 100:.2f}%)"
    )

    # ------------------------------------------------------------------
    # 2. TEMPORAL GAP STATISTICS
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("2. INTERMITTENT CELL GAP STATISTICS")
    print("-" * 90)

    intermittent_indices = np.argwhere(intermittent)

    if len(intermittent_indices) == 0:
        print("No intermittently missing cells.")
        ds.close()
        return

    records = []

    for lat_idx, lon_idx in intermittent_indices:

        series = valid[:, lat_idx, lon_idx]

        missing_idx = np.where(~series)[0]

        # Missing observations for this spatial cell
        missing_count = len(missing_idx)

        # Identify contiguous missing runs
        runs = []

        start = missing_idx[0]
        previous = missing_idx[0]

        for idx in missing_idx[1:]:
            if idx == previous + 1:
                previous = idx
            else:
                runs.append((start, previous))
                start = idx
                previous = idx

        runs.append((start, previous))

        run_lengths = [
            end - start + 1
            for start, end in runs
        ]

        longest_gap = max(run_lengths)

        # First / last missing date
        first_missing = time[missing_idx[0]]
        last_missing = time[missing_idx[-1]]

        records.append(
            {
                "lat_idx": int(lat_idx),
                "lon_idx": int(lon_idx),
                "latitude": float(lat[lat_idx]),
                "longitude": float(lon[lon_idx]),
                "missing_days": missing_count,
                "missing_pct": missing_count / n_time * 100,
                "num_gap_runs": len(runs),
                "longest_gap_days": longest_gap,
                "first_missing": first_missing,
                "last_missing": last_missing,
            }
        )

    df = pd.DataFrame(records)

    print(f"Intermittent cells analyzed: {len(df):,}")

    print("\nMissing observations per intermittent cell:")
    print(f"  Minimum: {df['missing_days'].min()}")
    print(f"  Median:  {df['missing_days'].median():.1f}")
    print(f"  Mean:    {df['missing_days'].mean():.1f}")
    print(f"  Maximum: {df['missing_days'].max()}")

    print("\nMissing percentage per intermittent cell:")
    print(f"  Minimum: {df['missing_pct'].min():.3f}%")
    print(f"  Median:  {df['missing_pct'].median():.3f}%")
    print(f"  Mean:    {df['missing_pct'].mean():.3f}%")
    print(f"  Maximum: {df['missing_pct'].max():.3f}%")

    print("\nNumber of separate gap runs per cell:")
    print(f"  Minimum: {df['num_gap_runs'].min()}")
    print(f"  Median:  {df['num_gap_runs'].median():.1f}")
    print(f"  Mean:    {df['num_gap_runs'].mean():.1f}")
    print(f"  Maximum: {df['num_gap_runs'].max()}")

    print("\nLongest contiguous gap per cell:")
    print(f"  Minimum: {df['longest_gap_days'].min()} days")
    print(f"  Median:  {df['longest_gap_days'].median():.1f} days")
    print(f"  Mean:    {df['longest_gap_days'].mean():.1f} days")
    print(f"  Maximum: {df['longest_gap_days'].max()} days")

    # ------------------------------------------------------------------
    # 3. DISTRIBUTION OF GAP LENGTHS
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("3. LONGEST-GAP DISTRIBUTION")
    print("-" * 90)

    bins = [
        0,
        1,
        2,
        3,
        5,
        7,
        14,
        30,
        60,
        90,
        180,
        365,
        np.inf,
    ]

    labels = [
        "1 day",
        "2 days",
        "3 days",
        "4-5 days",
        "6-7 days",
        "8-14 days",
        "15-30 days",
        "31-60 days",
        "61-90 days",
        "91-180 days",
        "181-365 days",
        ">365 days",
    ]

    categories = pd.cut(
        df["longest_gap_days"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    distribution = categories.value_counts().sort_index()

    for label, count in distribution.items():
        pct = count / len(df) * 100
        print(f"{str(label):>15}: {count:>6,} cells ({pct:6.2f}%)")

    # ------------------------------------------------------------------
    # 4. HOW MANY ACTUAL MISSING OBSERVATIONS?
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("4. TOTAL INTERMITTENT MISSING OBSERVATIONS")
    print("-" * 90)

    total_intermittent_missing = int((~valid[:, :, :]).sum())

    # Only count missing values belonging to cells that have at least
    # one valid observation.
    intermittent_missing_mask = (
        (~valid)
        & intermittent[np.newaxis, :, :]
    )

    total_intermittent_missing = int(
        intermittent_missing_mask.sum()
    )

    total_values = values.size

    print(
        f"Intermittent missing observations: "
        f"{total_intermittent_missing:,}"
    )

    print(
        f"Percentage of entire dataset: "
        f"{total_intermittent_missing / total_values * 100:.4f}%"
    )

    print(
        f"Percentage of missing observations: "
        f"{total_intermittent_missing / (~valid).sum() * 100:.4f}%"
    )

    # ------------------------------------------------------------------
    # 5. LARGE GAPS
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("5. CELLS WITH LARGE CONTIGUOUS GAPS")
    print("-" * 90)

    thresholds = [3, 7, 14, 30, 60, 90, 180]

    for threshold in thresholds:
        count = int(
            (df["longest_gap_days"] >= threshold).sum()
        )

        pct = count / len(df) * 100

        print(
            f">= {threshold:3d} days: "
            f"{count:>6,} cells ({pct:6.2f}%)"
        )

    # ------------------------------------------------------------------
    # 6. WORST CELLS
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("6. TOP 30 WORST CELLS")
    print("-" * 90)

    worst = (
        df.sort_values(
            ["longest_gap_days", "missing_days"],
            ascending=False,
        )
        .head(30)
    )

    print(
        worst[
            [
                "latitude",
                "longitude",
                "missing_days",
                "missing_pct",
                "num_gap_runs",
                "longest_gap_days",
                "first_missing",
                "last_missing",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------------
    # 7. TEMPORAL DISTRIBUTION
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("7. MISSING OBSERVATIONS BY DATE")
    print("-" * 90)

    daily_missing = intermittent_missing_mask.sum(axis=(1, 2))

    nonzero = np.where(daily_missing > 0)[0]

    print(
        f"Days containing intermittent gaps: "
        f"{len(nonzero):,} / {n_time:,}"
    )

    if len(nonzero) > 0:
        print(
            f"First affected date: {time[nonzero[0]]}"
        )
        print(
            f"Last affected date:  {time[nonzero[-1]]}"
        )

        print(
            f"Maximum missing cells on one day: "
            f"{daily_missing.max():,}"
        )

        worst_days = (
            pd.DataFrame(
                {
                    "date": time,
                    "missing_cells": daily_missing,
                }
            )
            .sort_values(
                "missing_cells",
                ascending=False,
            )
            .head(20)
        )

        print("\nWorst affected dates:")
        print(worst_days.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. SPATIAL CLUSTERING
    # ------------------------------------------------------------------

    print("\n" + "-" * 90)
    print("8. SPATIAL CLUSTERING OF INTERMITTENT GAPS")
    print("-" * 90)

    mask = intermittent

    # Count neighboring intermittent cells.
    neighbor_count = np.zeros_like(mask, dtype=np.int16)

    neighbor_count[:-1, :] += mask[1:, :]
    neighbor_count[1:, :] += mask[:-1, :]
    neighbor_count[:, :-1] += mask[:, 1:]
    neighbor_count[:, 1:] += mask[:, :-1]

    cells_with_neighbors = (
        mask & (neighbor_count > 0)
    )

    isolated = mask & (neighbor_count == 0)

    print(
        f"Intermittent cells with >=1 neighboring "
        f"intermittent cell: {cells_with_neighbors.sum():,}"
    )

    print(
        f"Isolated intermittent cells: "
        f"{isolated.sum():,}"
    )

    print(
        f"Fraction clustered: "
        f"{cells_with_neighbors.sum() / mask.sum() * 100:.2f}%"
    )

    # ------------------------------------------------------------------
    # 9. FINAL RECOMMENDATION
    # ------------------------------------------------------------------

    print("\n" + "=" * 90)
    print("9. AUTOMATED RECOMMENDATION")
    print("=" * 90)

    max_gap = int(df["longest_gap_days"].max())
    median_gap = float(df["longest_gap_days"].median())

    cells_7 = int((df["longest_gap_days"] >= 7).sum())
    cells_30 = int((df["longest_gap_days"] >= 30).sum())

    if max_gap <= 3 and cells_7 == 0:
        recommendation = (
            "SKIP CNN/DINEOF. "
            "Intermittent gaps are short enough for simple temporal handling."
        )
    elif max_gap <= 14 and cells_30 == 0:
        recommendation = (
            "LIKELY SKIP CNN/DINEOF. "
            "Gaps exist but appear limited. "
            "Use simple temporal interpolation and validate."
        )
    else:
        recommendation = (
            "DO NOT SKIP INFILLING YET. "
            "There are substantial temporal gaps. "
            "Inspect the large-gap spatial/temporal patterns before choosing "
            "CNN/DINEOF."
        )

    print(recommendation)

    # ------------------------------------------------------------------
    # 10. SAVE DETAILED CSV
    # ------------------------------------------------------------------

    output = Path(path).with_name(
        Path(path).stem + "_gap_diagnostic.csv"
    )

    df.to_csv(output, index=False)

    print("\nDetailed cell-level diagnostics saved to:")
    print(output)

    ds.close()

    print("\n" + "=" * 90)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose intermittent temporal gaps in SSS NetCDF."
    )

    parser.add_argument(
        "path",
        nargs="?",
        default=(
            "/home/rakshitchaturvedi/Desktop/Projects/"
            "oceanembed-training/data/processed/phase2/sss.nc"
        ),
        help="Path to sss.nc",
    )

    args = parser.parse_args()

    diagnose(args.path)


if __name__ == "__main__":
    main()