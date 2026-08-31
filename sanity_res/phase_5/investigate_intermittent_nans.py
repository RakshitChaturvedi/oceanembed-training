#!/usr/bin/env python3

"""
OceanEmbed — Phase 5 Prerequisite #1c
=====================================

Purpose
-------
Investigate INTERMITTENT NaNs in the surface datasets after establishing that
permanent NaNs are geographically consistent with land masking.

This script is READ ONLY.

It does NOT:
    - modify any NetCDF files
    - fill NaNs
    - interpolate data
    - open ARMOR3D
    - open GLORYS
    - load an entire 3-D dataset into RAM

It DOES:
    1. Scan datasets one time slice at a time.
    2. Identify spatial cells with intermittent missing values.
    3. Measure missing counts/fractions.
    4. Find consecutive missing runs.
    5. Report exact missing-date ranges.
    6. Check whether intermittent cells are adjacent to permanent land.
    7. Compare SSS and current intermittent masks.
    8. Verify current_u/current_v missingness consistency.
    9. Write machine-readable JSON/CSV reports.

Expected datasets:
    data/processed/phase2/sss.nc
    data/processed/phase2/currents.nc
    data/processed/phase2/sst.nc
    data/processed/phase2/ssh.nc
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training"
)

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "phase2"

OUTPUT_DIR = PROJECT_ROOT / "sanity_res" / "phase_5" / "intermittent_nan_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "sss": {
        "path": DATA_DIR / "sss.nc",
        "variables": ["sss"],
    },
    "currents": {
        "path": DATA_DIR / "currents.nc",
        "variables": ["current_u", "current_v"],
    },
}


# =============================================================================
# UTILITIES
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def fail(message: str) -> None:
    print()
    print("FATAL ERROR:")
    print(message)
    sys.exit(1)


def check_file(path: Path) -> None:
    if not path.exists():
        fail(f"File does not exist:\n  {path}")


def consecutive_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """
    Given sorted integer indices, return inclusive consecutive runs.

    Example:
        [1, 2, 3, 7, 8] -> [(1, 3), (7, 8)]
    """
    if len(indices) == 0:
        return []

    runs = []

    start = int(indices[0])
    previous = int(indices[0])

    for value in indices[1:]:
        value = int(value)

        if value == previous + 1:
            previous = value
        else:
            runs.append((start, previous))
            start = value
            previous = value

    runs.append((start, previous))

    return runs


def run_to_dates(
    runs: list[tuple[int, int]],
    times: np.ndarray,
) -> list[dict]:
    result = []

    for start, end in runs:
        result.append(
            {
                "start_index": start,
                "end_index": end,
                "length": end - start + 1,
                "start_date": str(times[start]),
                "end_date": str(times[end]),
            }
        )

    return result


def calculate_neighbor_land_mask(permanent_land_mask: np.ndarray) -> np.ndarray:
    """
    Return cells that are directly adjacent to permanent land.

    8-neighbor connectivity is used.

    This is deliberately a simple spatial diagnostic. It does not claim
    that adjacency proves a missing value is caused by land.
    """

    land = permanent_land_mask

    neighbor = np.zeros_like(land, dtype=bool)

    neighbor[:-1, :] |= land[1:, :]
    neighbor[1:, :] |= land[:-1, :]

    neighbor[:, :-1] |= land[:, 1:]
    neighbor[:, 1:] |= land[:, :-1]

    neighbor[:-1, :-1] |= land[1:, 1:]
    neighbor[:-1, 1:] |= land[1:, :-1]
    neighbor[1:, :-1] |= land[:-1, 1:]
    neighbor[1:, 1:] |= land[:-1, :-1]

    return neighbor


def load_coordinates(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    time = ds["time"].values

    return lat, lon, time


# =============================================================================
# FIRST PASS — TEMPORAL MISSINGNESS
# =============================================================================

def scan_variable(
    path: Path,
    variable: str,
) -> dict:
    """
    Scan one variable one time slice at a time.

    Returns all information needed for later analysis.
    """

    print_header(f"SCANNING: {variable}")

    print(f"File:     {path}")
    print(f"Variable: {variable}")
    print("Mode:     one time slice at a time")
    print()

    with xr.open_dataset(
        path,
        decode_times=True,
        cache=False,
    ) as ds:

        if variable not in ds:
            fail(f"Variable '{variable}' not found in {path}")

        da = ds[variable]

        required_dims = {"time", "latitude", "longitude"}

        if not required_dims.issubset(set(da.dims)):
            fail(
                f"{variable} does not have expected dimensions "
                f"{sorted(required_dims)}.\n"
                f"Found: {da.dims}"
            )

        da = da.transpose("time", "latitude", "longitude")

        lat, lon, times = load_coordinates(ds)

        nt = len(times)
        ny = len(lat)
        nx = len(lon)

        print(f"Time steps : {nt}")
        print(f"Latitude   : {ny}")
        print(f"Longitude  : {nx}")
        print(f"Grid cells : {ny * nx}")
        print()

        # ---------------------------------------------------------------------
        # First pass:
        # Count how many times each spatial cell is NaN.
        #
        # Shape:
        #     (latitude, longitude)
        #
        # Only one 2-D time slice is materialized at a time.
        # ---------------------------------------------------------------------

        nan_count = np.zeros((ny, nx), dtype=np.int32)

        for t in range(nt):

            values = da.isel(time=t).values

            nan_mask = ~np.isfinite(values)

            nan_count += nan_mask

            if (t + 1) % 100 == 0 or t == nt - 1:
                print(
                    f"\r  scanned {t + 1:4d} / {nt} time steps",
                    end="",
                    flush=True,
                )

        print()

        permanent = nan_count == nt
        complete = nan_count == 0
        intermittent = (nan_count > 0) & (nan_count < nt)

        print()
        print("TEMPORAL MISSINGNESS")
        print("-" * 100)
        print(f"Permanent NaN cells : {np.count_nonzero(permanent):,}")
        print(f"Intermittent cells  : {np.count_nonzero(intermittent):,}")
        print(f"Complete cells      : {np.count_nonzero(complete):,}")

        if np.count_nonzero(intermittent) == 0:
            print("\nNo intermittent cells found.")
            return {
                "variable": variable,
                "path": str(path),
                "time_steps": nt,
                "lat_size": ny,
                "lon_size": nx,
                "permanent_cells": int(np.count_nonzero(permanent)),
                "intermittent_cells": 0,
                "complete_cells": int(np.count_nonzero(complete)),
                "cells": [],
                "_internal": {
                    "lat": lat,
                    "lon": lon,
                    "times": times,
                    "permanent": permanent,
                    "intermittent": intermittent,
                },
            }

        # ---------------------------------------------------------------------
        # Build detailed information only for intermittent cells.
        #
        # Number of intermittent cells is small, so this is cheap.
        # ---------------------------------------------------------------------

        candidate_indices = np.argwhere(intermittent)

        print()
        print(f"Detailed analysis for {len(candidate_indices):,} cells...")

        # Missing-date boolean matrix ONLY for intermittent cells.
        #
        # Shape:
        #     (intermittent_cells, time)
        #
        # For SSS this is at most ~857 x 1419.
        #
        missing_history = np.zeros(
            (len(candidate_indices), nt),
            dtype=bool,
        )

        for t in range(nt):

            values = da.isel(time=t).values
            nan_mask = ~np.isfinite(values)

            for i, (iy, ix) in enumerate(candidate_indices):
                missing_history[i, t] = nan_mask[iy, ix]

        cells = []

        for i, (iy, ix) in enumerate(candidate_indices):

            missing_indices = np.flatnonzero(missing_history[i])

            runs = consecutive_runs(missing_indices)
            dated_runs = run_to_dates(runs, times)

            missing_count = len(missing_indices)
            missing_fraction = missing_count / nt

            longest_run = max(
                (run["length"] for run in dated_runs),
                default=0,
            )

            cells.append(
                {
                    "lat_index": int(iy),
                    "lon_index": int(ix),
                    "latitude": float(lat[iy]),
                    "longitude": float(lon[ix]),
                    "missing_count": int(missing_count),
                    "valid_count": int(nt - missing_count),
                    "missing_fraction": float(missing_fraction),
                    "longest_consecutive_missing": int(longest_run),
                    "number_of_missing_runs": int(len(dated_runs)),
                    "missing_runs": dated_runs,
                }
            )

        # Sort worst cases first.
        cells.sort(
            key=lambda x: (
                -x["missing_count"],
                -x["longest_consecutive_missing"],
            )
        )

        return {
            "variable": variable,
            "path": str(path),
            "time_steps": nt,
            "lat_size": ny,
            "lon_size": nx,
            "permanent_cells": int(np.count_nonzero(permanent)),
            "intermittent_cells": int(np.count_nonzero(intermittent)),
            "complete_cells": int(np.count_nonzero(complete)),
            "cells": cells,
            "_internal": {
                "lat": lat,
                "lon": lon,
                "times": times,
                "permanent": permanent,
                "intermittent": intermittent,
            },
        }


# =============================================================================
# CURRENT U/V CONSISTENCY
# =============================================================================

def compare_current_masks(
    u_result: dict,
    v_result: dict,
) -> dict:

    print_header("CURRENT U/V MISSINGNESS CONSISTENCY")

    u_internal = u_result["_internal"]
    v_internal = v_result["_internal"]

    u_mask = u_internal["intermittent"]
    v_mask = v_internal["intermittent"]

    if u_mask.shape != v_mask.shape:
        fail("current_u and current_v masks have different shapes.")

    identical_intermit = np.array_equal(u_mask, v_mask)

    print(f"Intermittent spatial masks identical: {identical_intermit}")

    u_perm = u_internal["permanent"]
    v_perm = v_internal["permanent"]

    identical_perm = np.array_equal(u_perm, v_perm)

    print(f"Permanent spatial masks identical:     {identical_perm}")

    return {
        "intermittent_mask_identical": bool(identical_intermit),
        "permanent_mask_identical": bool(identical_perm),
    }


# =============================================================================
# CROSS-DATASET COMPARISON
# =============================================================================

def compare_intermittent_masks(
    results: dict,
) -> dict:

    print_header("INTERMITTENT MASK CROSS-DATASET ANALYSIS")

    sss_mask = results["sss"]["_internal"]["intermittent"]
    cur_mask = results["current_u"]["_internal"]["intermittent"]

    intersection = sss_mask & cur_mask
    union = sss_mask | cur_mask

    sss_only = sss_mask & ~cur_mask
    cur_only = cur_mask & ~sss_mask

    intersection_count = int(np.count_nonzero(intersection))
    union_count = int(np.count_nonzero(union))

    sss_count = int(np.count_nonzero(sss_mask))
    cur_count = int(np.count_nonzero(cur_mask))

    print(f"SSS intermittent cells       : {sss_count:,}")
    print(f"Currents intermittent cells  : {cur_count:,}")
    print(f"Intersection                 : {intersection_count:,}")
    print(f"Union                        : {union_count:,}")
    print(f"SSS-only                     : {np.count_nonzero(sss_only):,}")
    print(f"Currents-only                : {np.count_nonzero(cur_only):,}")

    if union_count:
        iou = intersection_count / union_count
    else:
        iou = 1.0

    print(f"Intersection over Union      : {iou:.6%}")

    return {
        "sss_cells": sss_count,
        "currents_cells": cur_count,
        "intersection": intersection_count,
        "union": union_count,
        "sss_only": int(np.count_nonzero(sss_only)),
        "currents_only": int(np.count_nonzero(cur_only)),
        "iou": float(iou),
    }


# =============================================================================
# LAND-ADJACENCY ANALYSIS
# =============================================================================

def analyze_land_adjacency(
    result: dict,
) -> dict:

    variable = result["variable"]

    internal = result["_internal"]

    permanent_land = internal["permanent"]
    intermittent = internal["intermittent"]

    adjacent = calculate_neighbor_land_mask(permanent_land)

    intermittent_adjacent = intermittent & adjacent
    intermittent_not_adjacent = intermittent & ~adjacent

    total = int(np.count_nonzero(intermittent))

    adjacent_count = int(np.count_nonzero(intermittent_adjacent))
    nonadjacent_count = int(np.count_nonzero(intermittent_not_adjacent))

    fraction = adjacent_count / total if total else 0.0

    print_header(f"LAND-ADJACENCY: {variable}")

    print(f"Intermittent cells                  : {total:,}")
    print(f"Adjacent to permanent land          : {adjacent_count:,}")
    print(f"Not adjacent to permanent land      : {nonadjacent_count:,}")
    print(f"Adjacent fraction                   : {fraction:.6%}")

    return {
        "intermittent_cells": total,
        "adjacent_to_permanent_land": adjacent_count,
        "not_adjacent_to_permanent_land": nonadjacent_count,
        "adjacent_fraction": float(fraction),
    }


# =============================================================================
# PRINT TOP CASES
# =============================================================================

def print_top_cases(
    result: dict,
    n: int = 25,
) -> None:

    variable = result["variable"]
    cells = result["cells"]

    print_header(f"TOP {min(n, len(cells))} INTERMITTENT CELLS: {variable}")

    if not cells:
        print("None.")
        return

    for i, cell in enumerate(cells[:n], start=1):

        print(
            f"{i:3d}. "
            f"lat={cell['latitude']:8.3f} "
            f"lon={cell['longitude']:9.3f} "
            f"missing={cell['missing_count']:4d}/{result['time_steps']} "
            f"({cell['missing_fraction']:.2%}) "
            f"longest_run={cell['longest_consecutive_missing']:4d} "
            f"runs={cell['number_of_missing_runs']}"
        )

        # Print the first few missing runs so we can inspect temporal pattern.
        for run in cell["missing_runs"][:5]:
            print(
                f"       "
                f"{run['start_date']} -> {run['end_date']} "
                f"({run['length']} step(s))"
            )

        if len(cell["missing_runs"]) > 5:
            print(
                f"       ... {len(cell['missing_runs']) - 5} more run(s)"
            )


# =============================================================================
# SAVE CSV
# =============================================================================

def save_csv(
    result: dict,
    output_path: Path,
) -> None:

    fieldnames = [
        "lat_index",
        "lon_index",
        "latitude",
        "longitude",
        "missing_count",
        "valid_count",
        "missing_fraction",
        "longest_consecutive_missing",
        "number_of_missing_runs",
        "first_missing_date",
        "last_missing_date",
    ]

    with output_path.open("w", newline="") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for cell in result["cells"]:

            runs = cell["missing_runs"]

            row = {
                "lat_index": cell["lat_index"],
                "lon_index": cell["lon_index"],
                "latitude": cell["latitude"],
                "longitude": cell["longitude"],
                "missing_count": cell["missing_count"],
                "valid_count": cell["valid_count"],
                "missing_fraction": cell["missing_fraction"],
                "longest_consecutive_missing": cell[
                    "longest_consecutive_missing"
                ],
                "number_of_missing_runs": cell["number_of_missing_runs"],
                "first_missing_date": (
                    runs[0]["start_date"] if runs else ""
                ),
                "last_missing_date": (
                    runs[-1]["end_date"] if runs else ""
                ),
            }

            writer.writerow(row)


# =============================================================================
# SAVE JSON
# =============================================================================

def make_json_safe(obj):
    """
    Convert NumPy values into normal Python values.
    """

    if isinstance(obj, dict):
        return {
            key: make_json_safe(value)
            for key, value in obj.items()
            if key != "_internal"
        }

    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    return obj


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header("OceanEmbed — Phase 5 Prerequisite #1c")

    print(
        "Purpose:\n"
        "Investigate intermittent NaNs in SSS and currents.\n"
    )

    print(
        "READ ONLY.\n"
        "ARMOR3D and GLORYS are NOT opened.\n"
        "Datasets are processed one time slice at a time."
    )

    # -------------------------------------------------------------------------
    # Validate files.
    # -------------------------------------------------------------------------

    for config in DATASETS.values():
        check_file(config["path"])

    # -------------------------------------------------------------------------
    # Scan SSS.
    # -------------------------------------------------------------------------

    sss_result = scan_variable(
        DATASETS["sss"]["path"],
        "sss",
    )

    # -------------------------------------------------------------------------
    # Scan current U/V.
    # -------------------------------------------------------------------------

    current_u_result = scan_variable(
        DATASETS["currents"]["path"],
        "current_u",
    )

    current_v_result = scan_variable(
        DATASETS["currents"]["path"],
        "current_v",
    )

    results = {
        "sss": sss_result,
        "current_u": current_u_result,
        "current_v": current_v_result,
    }

    # -------------------------------------------------------------------------
    # Current U/V consistency.
    # -------------------------------------------------------------------------

    uv_consistency = compare_current_masks(
        current_u_result,
        current_v_result,
    )

    # -------------------------------------------------------------------------
    # Cross-dataset intermittent-mask comparison.
    # -------------------------------------------------------------------------

    cross_comparison = compare_intermittent_masks(
        results,
    )

    # -------------------------------------------------------------------------
    # Land adjacency.
    # -------------------------------------------------------------------------

    adjacency = {
        "sss": analyze_land_adjacency(sss_result),
        "current_u": analyze_land_adjacency(current_u_result),
        "current_v": analyze_land_adjacency(current_v_result),
    }

    # -------------------------------------------------------------------------
    # Print top cases.
    # -------------------------------------------------------------------------

    print_top_cases(sss_result, 25)
    print_top_cases(current_u_result, 25)

    # -------------------------------------------------------------------------
    # Save CSV reports.
    # -------------------------------------------------------------------------

    print_header("WRITING REPORTS")

    save_csv(
        sss_result,
        OUTPUT_DIR / "sss_intermittent_cells.csv",
    )

    save_csv(
        current_u_result,
        OUTPUT_DIR / "current_u_intermittent_cells.csv",
    )

    save_csv(
        current_v_result,
        OUTPUT_DIR / "current_v_intermittent_cells.csv",
    )

    print(
        f"Wrote:\n"
        f"  {OUTPUT_DIR / 'sss_intermittent_cells.csv'}\n"
        f"  {OUTPUT_DIR / 'current_u_intermittent_cells.csv'}\n"
        f"  {OUTPUT_DIR / 'current_v_intermittent_cells.csv'}"
    )

    # -------------------------------------------------------------------------
    # Build JSON report.
    # -------------------------------------------------------------------------

    report = {
        "phase": 5,
        "prerequisite": "1c",
        "purpose": (
            "Investigate intermittent NaNs without modifying datasets "
            "or loading ARMOR3D/GLORYS."
        ),
        "read_only": True,
        "datasets": {
            "sss": str(DATASETS["sss"]["path"]),
            "currents": str(DATASETS["currents"]["path"]),
        },
        "results": results,
        "current_uv_consistency": uv_consistency,
        "intermittent_cross_dataset_comparison": cross_comparison,
        "land_adjacency": adjacency,
    }

    json_path = OUTPUT_DIR / "intermittent_nan_analysis.json"

    with json_path.open("w") as f:
        json.dump(
            make_json_safe(report),
            f,
            indent=2,
        )

    print(f"  {json_path}")

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    print_header("FINAL SUMMARY")

    print(
        f"SSS:\n"
        f"  intermittent cells = "
        f"{sss_result['intermittent_cells']:,}"
    )

    print(
        f"Currents U:\n"
        f"  intermittent cells = "
        f"{current_u_result['intermittent_cells']:,}"
    )

    print(
        f"Currents V:\n"
        f"  intermittent cells = "
        f"{current_v_result['intermittent_cells']:,}"
    )

    print()
    print(
        "Current U/V intermittent masks identical:",
        uv_consistency["intermittent_mask_identical"],
    )

    print(
        "SSS/current intermittent-mask IoU:",
        f"{cross_comparison['iou']:.6%}",
    )

    print()
    print("No dataset was modified.")
    print("No ARMOR3D/GLORYS file was opened.")

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()