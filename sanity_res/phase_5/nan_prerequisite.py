#!/usr/bin/env python3

"""
OceanEmbed — Phase 5 Prerequisite #1
====================================

Determine whether NaNs in SST, SSS, SSH and currents correspond primarily
to land cells or to genuine missing ocean observations.

IMPORTANT:
    - READ ONLY
    - Does NOT modify any NetCDF
    - Does NOT perform interpolation
    - Does NOT create the Phase 5 tensor
    - Does NOT load ARMOR3D / GLORYS entirely into RAM

Strategy
--------
1. Read the four observational/derived datasets.
2. Determine their static spatial NaN masks.
3. Use the smallest available spatial validity information from GLORYS
   without loading the complete dataset.
4. Fall back to a consensus mask from the four variables if an authoritative
   GLORYS mask cannot be identified.
5. Separate:
       permanent/static NaNs
       intermittent ocean NaNs
6. Produce a JSON diagnostic report.

The script intentionally does NOT decide how to fill anything.
That is Phase 5 Prerequisite #2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training"
)

PHASE2_DIR = PROJECT_ROOT / "data" / "processed" / "phase2"
PHASE4_DIR = PROJECT_ROOT / "data" / "processed" / "phase4"
PHASE5_DIR = PROJECT_ROOT / "data" / "processed" / "phase5"

REPORT_PATH = PHASE5_DIR / "nan_prerequisite.json"


FILES = {
    "sst": PHASE2_DIR / "sst.nc",
    "sss": PHASE2_DIR / "sss.nc",
    "ssh": PHASE2_DIR / "ssh.nc",
    "currents": PHASE2_DIR / "currents.nc",
    "glorys": PHASE2_DIR / "glorys.nc",
    "armor3d": PHASE2_DIR / "armor3d.nc",
}


# =============================================================================
# Configuration
# =============================================================================

OBSERVATIONAL_DATASETS = {
    "sst": {
        "variable": "sst",
    },
    "sss": {
        "variable": "sss",
    },
    "ssh": {
        "variable": "ssh",
    },
    "currents": {
        # Currents has two variables. They should have the same NaN mask,
        # but we verify that rather than assuming it.
        "variables": ["current_u", "current_v"],
    },
}


# =============================================================================
# Utilities
# =============================================================================

def print_header(title: str):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


def get_dataset(path: Path):
    """
    Open lazily.

    No .load() is used.
    """
    return xr.open_dataset(
        path,
        chunks="auto",
        cache=False,
    )


def find_spatial_dims(da: xr.DataArray):
    """
    Find latitude/longitude dimensions without assuming exact names.
    """

    lat_candidates = ["latitude", "lat", "Latitude", "LATITUDE"]
    lon_candidates = ["longitude", "lon", "Longitude", "LONGITUDE"]

    lat_dim = next(
        (x for x in lat_candidates if x in da.dims),
        None,
    )

    lon_dim = next(
        (x for x in lon_candidates if x in da.dims),
        None,
    )

    if lat_dim is None or lon_dim is None:
        raise ValueError(
            f"Could not identify spatial dimensions for {da.name}. "
            f"dims={da.dims}"
        )

    return lat_dim, lon_dim


def get_time_dim(da: xr.DataArray):
    for candidate in ["time", "Time"]:
        if candidate in da.dims:
            return candidate

    return None


def static_nan_mask(da: xr.DataArray):
    """
    Calculate a cell-wise 'always NaN' mask across time.

    Returns a 2-D boolean DataArray:

        True  = NaN at every timestep
        False = at least one valid observation exists

    This is computed lazily and only the resulting 101x241 mask is loaded.
    """

    time_dim = get_time_dim(da)

    if time_dim is None:
        return np.isnan(da).astype(bool)

    return np.isnan(da).all(dim=time_dim).compute()


def valid_count_map(da: xr.DataArray):
    """
    Count valid observations per spatial cell.

    Only the resulting 101x241 array is materialized.
    """

    time_dim = get_time_dim(da)

    if time_dim is None:
        return (~np.isnan(da)).astype(np.int32).compute()

    return da.notnull().sum(dim=time_dim).compute()


def nan_count(da: xr.DataArray):
    """
    Count NaNs without loading the full dataset.
    """

    return int(np.isnan(da).sum().compute().item())


def total_count(da: xr.DataArray):
    return int(da.size)


def describe_mask(mask: np.ndarray):
    total = int(mask.size)
    true_count = int(mask.sum())

    return {
        "total_cells": total,
        "true_cells": true_count,
        "fraction": true_count / total if total else 0.0,
    }


def compare_coordinate_grids(ds_a, ds_b):
    """
    Compare lat/lon coordinate values.

    Only coordinates are loaded, not data variables.
    """

    result = {
        "same_grid": False,
        "latitude": {},
        "longitude": {},
    }

    def find_coord(ds, candidates):
        for name in candidates:
            if name in ds.coords:
                return name
        return None

    lat_a = find_coord(ds_a, ["latitude", "lat", "Latitude"])
    lon_a = find_coord(ds_a, ["longitude", "lon", "Longitude"])

    lat_b = find_coord(ds_b, ["latitude", "lat", "Latitude"])
    lon_b = find_coord(ds_b, ["longitude", "lon", "Longitude"])

    if not all([lat_a, lon_a, lat_b, lon_b]):
        return result

    lat1 = np.asarray(ds_a[lat_a].values)
    lon1 = np.asarray(ds_a[lon_a].values)

    lat2 = np.asarray(ds_b[lat_b].values)
    lon2 = np.asarray(ds_b[lon_b].values)

    same_lat = (
        lat1.shape == lat2.shape
        and np.array_equal(lat1, lat2)
    )

    same_lon = (
        lon1.shape == lon2.shape
        and np.array_equal(lon1, lon2)
    )

    result["latitude"] = {
        "dataset_a_coord": lat_a,
        "dataset_b_coord": lat_b,
        "same": bool(same_lat),
        "shape_a": list(lat1.shape),
        "shape_b": list(lat2.shape),
    }

    result["longitude"] = {
        "dataset_a_coord": lon_a,
        "dataset_b_coord": lon_b,
        "same": bool(same_lon),
        "shape_a": list(lon1.shape),
        "shape_b": list(lon2.shape),
    }

    result["same_grid"] = bool(same_lat and same_lon)

    return result


# =============================================================================
# Dataset inspection
# =============================================================================

def inspect_file(name: str, path: Path):
    print_header(f"INSPECTING: {name}")

    if not path.exists():
        print(f"MISSING: {path}")
        return {
            "exists": False,
            "path": str(path),
        }

    print(f"Path: {path}")
    print("Opening lazily...")

    ds = get_dataset(path)

    result = {
        "exists": True,
        "path": str(path),
        "dimensions": {
            k: int(v)
            for k, v in ds.sizes.items()
        },
        "coordinates": list(ds.coords),
        "data_variables": list(ds.data_vars),
    }

    print("\nDimensions:")
    for k, v in ds.sizes.items():
        print(f"  {k:20s}: {v}")

    print("\nCoordinates:")
    for name in ds.coords:
        da = ds[name]
        print(
            f"  {name:20s}: "
            f"shape={tuple(da.shape)}, "
            f"dtype={da.dtype}"
        )

    print("\nData variables:")
    for name in ds.data_vars:
        da = ds[name]
        print(
            f"  {name:20s}: "
            f"shape={tuple(da.shape)}, "
            f"dtype={da.dtype}"
        )

    ds.close()

    return result


# =============================================================================
# Build NaN masks
# =============================================================================

def analyse_single_variable(
    dataset_name: str,
    variable_name: str,
    path: Path,
):
    print_header(
        f"ANALYSING NaNs: {dataset_name}/{variable_name}"
    )

    ds = get_dataset(path)
    da = ds[variable_name]

    print(f"Dimensions: {da.dims}")
    print(f"Shape:      {da.shape}")

    spatial_dims = find_spatial_dims(da)

    # -------------------------------------------------------------------------
    # Overall NaNs
    # -------------------------------------------------------------------------

    total = total_count(da)
    nan_total = nan_count(da)

    print()
    print(f"Total values: {total:,}")
    print(
        f"Total NaNs:   {nan_total:,} "
        f"({nan_total / total:.6%})"
    )

    # -------------------------------------------------------------------------
    # Permanently missing cells
    # -------------------------------------------------------------------------

    print("\nComputing permanent spatial NaN mask...")
    permanent_mask_da = static_nan_mask(da)

    permanent_mask = np.asarray(
        permanent_mask_da.values,
        dtype=bool,
    )

    print(
        f"Permanent NaN cells: "
        f"{permanent_mask.sum():,} / "
        f"{permanent_mask.size:,} "
        f"({permanent_mask.mean():.6%})"
    )

    # -------------------------------------------------------------------------
    # Valid observation count per cell
    # -------------------------------------------------------------------------

    print("Computing valid-observation counts...")
    valid_counts_da = valid_count_map(da)

    valid_counts = np.asarray(
        valid_counts_da.values,
        dtype=np.int32,
    )

    # Cells with some missingness but also some observations.
    intermittent_mask = (
        (valid_counts > 0)
        & (valid_counts < da.sizes[get_time_dim(da)])
    )

    # Cells that are completely valid.
    complete_mask = (
        valid_counts == da.sizes[get_time_dim(da)]
    )

    print(
        f"Complete cells:     "
        f"{complete_mask.sum():,}"
    )

    print(
        f"Intermittent cells: "
        f"{intermittent_mask.sum():,}"
    )

    print(
        f"Permanent NaN cells:"
        f"{permanent_mask.sum():,}"
    )

    result = {
        "dataset": dataset_name,
        "variable": variable_name,
        "shape": list(da.shape),
        "spatial_dims": list(spatial_dims),
        "total_values": total,
        "total_nan_values": nan_total,
        "total_nan_fraction": nan_total / total,
        "permanent_nan_cells": int(permanent_mask.sum()),
        "permanent_nan_fraction": float(permanent_mask.mean()),
        "intermittent_cells": int(intermittent_mask.sum()),
        "intermittent_fraction": float(intermittent_mask.mean()),
        "complete_cells": int(complete_mask.sum()),
        "complete_fraction": float(complete_mask.mean()),
        "valid_count_min": int(valid_counts.min()),
        "valid_count_max": int(valid_counts.max()),
        "valid_count_mean": float(valid_counts.mean()),
        "_permanent_mask": permanent_mask,
        "_intermittent_mask": intermittent_mask,
        "_valid_counts": valid_counts,
    }

    ds.close()

    return result


# =============================================================================
# Consensus mask
# =============================================================================

def build_consensus_land_mask(results):
    """
    Build a conservative consensus permanent-NaN mask.

    A cell is considered permanently missing if ALL four datasets are
    permanently NaN there.

    This is NOT declared the final authoritative land mask.

    It is only a diagnostic to reveal the common spatial missingness pattern.
    """

    masks = [
        results["sst"]["_permanent_mask"],
        results["sss"]["_permanent_mask"],
        results["ssh"]["_permanent_mask"],
        results["currents_u"]["_permanent_mask"],
    ]

    consensus = np.logical_and.reduce(masks)

    return consensus


# =============================================================================
# Pairwise comparison
# =============================================================================

def pairwise_overlap(a: np.ndarray, b: np.ndarray):
    """
    Jaccard-style overlap plus directional coverage.
    """

    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()

    return {
        "a_cells": int(a.sum()),
        "b_cells": int(b.sum()),
        "intersection": int(intersection),
        "union": int(union),
        "intersection_over_union": (
            float(intersection / union)
            if union
            else 1.0
        ),
        "a_covered_by_b": (
            float(intersection / a.sum())
            if a.sum()
            else 1.0
        ),
        "b_covered_by_a": (
            float(intersection / b.sum())
            if b.sum()
            else 1.0
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():

    print_header(
        "OceanEmbed — Phase 5 Prerequisite #1"
    )

    print(
        "Goal: determine whether permanent NaNs are spatially consistent "
        "with land."
    )

    print(
        "\nThis script is READ ONLY and uses lazy NetCDF access."
    )

    PHASE5_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "phase": 5,
        "prerequisite": 1,
        "purpose": (
            "Determine whether NaNs correspond to land or genuine "
            "missing ocean observations."
        ),
        "read_only": True,
        "datasets": {},
    }

    # =========================================================================
    # Basic inspection
    # =========================================================================

    for name, path in FILES.items():
        report["datasets"][name] = inspect_file(
            name,
            path,
        )

    # =========================================================================
    # Analyze the four relevant datasets
    # =========================================================================

    results = {}

    # SST
    if FILES["sst"].exists():
        results["sst"] = analyse_single_variable(
            "sst",
            "sst",
            FILES["sst"],
        )

    # SSS
    if FILES["sss"].exists():
        results["sss"] = analyse_single_variable(
            "sss",
            "sss",
            FILES["sss"],
        )

    # SSH
    if FILES["ssh"].exists():
        results["ssh"] = analyse_single_variable(
            "ssh",
            "ssh",
            FILES["ssh"],
        )

    # Currents U
    if FILES["currents"].exists():
        results["currents_u"] = analyse_single_variable(
            "currents",
            "current_u",
            FILES["currents"],
        )

        results["currents_v"] = analyse_single_variable(
            "currents",
            "current_v",
            FILES["currents"],
        )

    report["nan_analysis"] = {}

    # =========================================================================
    # Strip internal masks before JSON serialization
    # =========================================================================

    for name, result in results.items():

        clean_result = {
            k: v
            for k, v in result.items()
            if not k.startswith("_")
        }

        report["nan_analysis"][name] = clean_result

    # =========================================================================
    # Verify current U/V masks agree
    # =========================================================================

    if "currents_u" in results and "currents_v" in results:

        u_mask = results["currents_u"]["_permanent_mask"]
        v_mask = results["currents_v"]["_permanent_mask"]

        same_current_mask = np.array_equal(
            u_mask,
            v_mask,
        )

        print_header(
            "CURRENT U/V MASK CONSISTENCY"
        )

        print(
            f"Permanent NaN masks identical: "
            f"{same_current_mask}"
        )

        report["current_uv_mask_consistency"] = {
            "identical": bool(same_current_mask),
        }

    # =========================================================================
    # Build common permanent mask
    # =========================================================================

    required = [
        "sst",
        "sss",
        "ssh",
        "currents_u",
    ]

    if all(name in results for name in required):

        print_header(
            "COMMON PERMANENT-MISSING PATTERN"
        )

        consensus = build_consensus_land_mask(
            results
        )

        print(
            "Cells permanently NaN in ALL four datasets:"
        )

        print(
            f"  {consensus.sum():,} / "
            f"{consensus.size:,} "
            f"({consensus.mean():.6%})"
        )

        report["common_permanent_missing"] = {
            "cells": int(consensus.sum()),
            "fraction": float(consensus.mean()),
        }

        # Pairwise comparisons
        print_header(
            "PAIRWISE PERMANENT-NaN OVERLAP"
        )

        names = required

        overlap_report = {}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                a_name = names[i]
                b_name = names[j]

                key = f"{a_name}__{b_name}"

                overlap = pairwise_overlap(
                    results[a_name]["_permanent_mask"],
                    results[b_name]["_permanent_mask"],
                )

                overlap_report[key] = overlap

                print(
                    f"\n{a_name} vs {b_name}"
                )

                print(
                    f"  A cells:       "
                    f"{overlap['a_cells']:,}"
                )

                print(
                    f"  B cells:       "
                    f"{overlap['b_cells']:,}"
                )

                print(
                    f"  Intersection:  "
                    f"{overlap['intersection']:,}"
                )

                print(
                    f"  IoU:           "
                    f"{overlap['intersection_over_union']:.6%}"
                )

                print(
                    f"  A covered B:   "
                    f"{overlap['a_covered_by_b']:.6%}"
                )

                print(
                    f"  B covered A:   "
                    f"{overlap['b_covered_by_a']:.6%}"
                )

        report["pairwise_permanent_nan_overlap"] = (
            overlap_report
        )

    # =========================================================================
    # Inspect GLORYS / ARMOR3D without loading them
    # =========================================================================

    print_header(
        "REFERENCE DATASET METADATA"
    )

    for name in ["glorys", "armor3d"]:

        path = FILES[name]

        if not path.exists():
            print(f"{name}: MISSING")
            continue

        print(
            f"\n{name.upper()}: "
            "metadata inspection only"
        )

        ds = get_dataset(path)

        print(
            "  Data variables:"
        )

        for var in ds.data_vars:
            da = ds[var]

            print(
                f"    {var:30s} "
                f"shape={tuple(da.shape)} "
                f"dtype={da.dtype}"
            )

        print(
            "  No data variables loaded into memory."
        )

        ds.close()

    # =========================================================================
    # Save report
    # =========================================================================

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # =========================================================================
    # Final message
    # =========================================================================

    print_header(
        "PREREQUISITE #1 COMPLETE"
    )

    print(
        f"Report written to:\n"
        f"  {REPORT_PATH}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  No interpolation performed."
    )

    print(
        "  No values modified."
    )

    print(
        "  No masks written back to NetCDF."
    )

    print(
        "  No ARMOR3D/GLORYS full-memory load performed."
    )

    print(
        "\nUse the results to decide what requires handling "
        "in Prerequisite #2."
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)

    except Exception as exc:
        print(
            f"\nFATAL ERROR: {exc}"
        )
        raise