from __future__ import annotations

from pathlib import Path

import pandas as pd
import xarray as xr


ROOT = Path("data/raw")

DATASETS = {
    "sst": sorted((ROOT / "sst").glob("*.nc")),
    "sss": sorted((ROOT / "sss").glob("*.nc")),
    "ssh": sorted((ROOT / "ssh").glob("*.nc")),
    "wind": sorted((ROOT / "wind").glob("*.nc")),
    "currents": sorted((ROOT / "currents").glob("*.nc")),
    "glorys": [ROOT / "glorys" / "glorys_ts.nc"],
    "armor3d": sorted((ROOT / "armor3d").glob("*.nc")),
}

def to_timestamp(value) -> pd.Timestamp:
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.Timestamp(
            year=value.year,
            month=value.month,
            day=value.day,
            hour=getattr(value, "hour", 0),
            minute=getattr(value, "minute", 0),
            second=getattr(value, "second", 0),
        )

def get_time_coverage(files: list[Path]) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not files:
        raise FileNotFoundError("No NetCDF files found.")

    starts = []
    ends = []

    for path in files:

        with xr.open_dataset(path, decode_times=True) as ds:
            if "time" not in ds.coords:
                raise ValueError(f"{path} has no time coordinate.")

            time = ds["time"].values

            if len(time) == 0:
                raise ValueError(f"{path} has empty time coordinate.")

            starts.append(to_timestamp(time[0]))
            ends.append(to_timestamp(time[-1]))

    return min(starts), max(ends)


def main() -> None:
    print("=" * 70)
    print("OceanEmbed Temporal Coverage Verification")
    print("=" * 70)

    coverage = {}

    for name, files in DATASETS.items():
        print(f"\n{name.upper()}")

        if not files:
            print("  NO FILES FOUND")
            continue

        start, end = get_time_coverage(files)

        coverage[name] = (start, end)

        print(f"  Files : {len(files)}")
        print(f"  Start : {start}")
        print(f"  End   : {end}")

    if len(coverage) != len(DATASETS):
        raise RuntimeError(
            "\nCannot determine common intersection because "
            "one or more datasets are missing."
        )

    common_start = max(start for start, _ in coverage.values())
    common_end = min(end for _, end in coverage.values())

    print("\n" + "=" * 70)
    print("COMMON TEMPORAL INTERSECTION")
    print("=" * 70)

    print(f"Start : {common_start}")
    print(f"End   : {common_end}")

    if common_start > common_end:
        raise RuntimeError("Datasets have NO common temporal intersection.")

    expected = pd.date_range(
        common_start,
        common_end,
        freq="1D",
    )

    print(f"Days  : {len(expected)}")

    print("\n" + "=" * 70)
    print("REQUESTED PHASE 2 RANGE")
    print("=" * 70)

    requested_start = pd.Timestamp("2021-01-01")
    requested_end = pd.Timestamp("2024-12-31")

    print(f"Start : {requested_start}")
    print(f"End   : {requested_end}")

    if requested_start < common_start:
        print(
            f"\nWARNING: requested start is {common_start - requested_start} "
            "earlier than the common intersection."
        )

    if requested_end > common_end:
        print(
            f"\nWARNING: requested end is {requested_end - common_end} "
            "later than the common intersection."
        )

    if requested_start >= common_start and requested_end <= common_end:
        print("\nOK: requested Phase 2 range is fully covered.")

    print("\n" + "=" * 70)
    print("DATASET COVERAGE SUMMARY")
    print("=" * 70)

    for name, (start, end) in coverage.items():
        print(
            f"{name:<12} "
            f"{start.date()} -> {end.date()}"
        )

    print("\nVerification complete.")


if __name__ == "__main__":
    main()