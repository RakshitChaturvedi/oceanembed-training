#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =============================================================================
# Paths
# =============================================================================

ROOT = Path(
    "/home/rakshitchaturvedi/Desktop/Projects/oceanembed-training"
)

PHASE2 = ROOT / "data" / "processed" / "phase2"

OUT = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "nan_land_verification"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


DATASETS = {
    "SST": (PHASE2 / "sst.nc", "sst"),
    "SSS": (PHASE2 / "sss.nc", "sss"),
    "SSH": (PHASE2 / "ssh.nc", "ssh"),
    "Current": (PHASE2 / "currents.nc", "current_u"),
}


# =============================================================================
# Load only 2-D permanent masks
# =============================================================================

def load_mask(path, variable):

    print(f"Opening: {path}")

    ds = xr.open_dataset(
        path,
        chunks="auto",
        cache=False,
    )

    da = ds[variable]

    print("Computing permanent NaN mask...")

    mask = (
        da.isnull()
        .all(dim="time")
        .compute()
        .values
    )

    lat = ds["latitude"].values
    lon = ds["longitude"].values

    ds.close()

    return (
        np.asarray(lat),
        np.asarray(lon),
        np.asarray(mask, dtype=bool),
    )


# =============================================================================
# Plot
# =============================================================================

def plot_mask(
    lat,
    lon,
    mask,
    title,
    output,
):

    fig = plt.figure(
        figsize=(13, 8)
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        [
            45,
            105,
            5,
            30,
        ],
        crs=ccrs.PlateCarree(),
    )

    # -------------------------------------------------------------------------
    # Permanent NaN cells
    # -------------------------------------------------------------------------

    mesh = ax.pcolormesh(
        lon,
        lat,
        mask.astype(float),
        transform=ccrs.PlateCarree(),
        shading="auto",
    )

    # -------------------------------------------------------------------------
    # Coastline
    # -------------------------------------------------------------------------

    ax.coastlines(
        resolution="50m",
        linewidth=0.8,
    )

    # -------------------------------------------------------------------------
    # Country borders
    # -------------------------------------------------------------------------

    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.4,
    )

    # -------------------------------------------------------------------------
    # Gridlines WITHOUT Cartopy gridliner
    #
    # This avoids the Shapely LinearRing problem.
    # -------------------------------------------------------------------------

    ax.set_xticks(
        np.arange(45, 106, 10),
        crs=ccrs.PlateCarree(),
    )

    ax.set_yticks(
        np.arange(5, 31, 5),
        crs=ccrs.PlateCarree(),
    )

    ax.set_xlabel(
        "Longitude (°E)"
    )

    ax.set_ylabel(
        "Latitude (°N)"
    )

    # -------------------------------------------------------------------------
    # Colorbar
    # -------------------------------------------------------------------------

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        pad=0.03,
    )

    cbar.set_label(
        "Permanent NaN"
    )

    ax.set_title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 100)
    print("OceanEmbed — Permanent NaN / Land Verification")
    print("=" * 100)

    masks = {}

    reference_lat = None
    reference_lon = None

    # -------------------------------------------------------------------------
    # Load masks
    # -------------------------------------------------------------------------

    for name, (path, variable) in DATASETS.items():

        lat, lon, mask = load_mask(
            path,
            variable,
        )

        if reference_lat is None:

            reference_lat = lat
            reference_lon = lon

        else:

            assert np.array_equal(
                reference_lat,
                lat,
            ), f"{name}: latitude mismatch"

            assert np.array_equal(
                reference_lon,
                lon,
            ), f"{name}: longitude mismatch"

        masks[name] = mask

        print(
            f"{name:10s}: "
            f"{mask.sum():,} permanent NaN cells"
        )

    # -------------------------------------------------------------------------
    # Common mask
    # -------------------------------------------------------------------------

    common = np.logical_and.reduce(
        list(masks.values())
    )

    print()
    print(
        f"Common permanent NaN cells: "
        f"{common.sum():,}"
    )

    print(
        f"Common fraction: "
        f"{common.mean():.6%}"
    )

    # -------------------------------------------------------------------------
    # Plot individual masks
    # -------------------------------------------------------------------------

    for name, mask in masks.items():

        filename = (
            OUT
            / f"{name.lower()}_permanent_nan.png"
        )

        plot_mask(
            reference_lat,
            reference_lon,
            mask,
            f"{name} — Permanent NaN Mask",
            filename,
        )

    # -------------------------------------------------------------------------
    # Plot common mask
    # -------------------------------------------------------------------------

    plot_mask(
        reference_lat,
        reference_lon,
        common,
        "Common Permanent NaN Mask — SST + SSS + SSH + Currents",
        OUT / "common_permanent_nan.png",
    )

    # -------------------------------------------------------------------------
    # Save masks
    # -------------------------------------------------------------------------

    np.savez_compressed(
        OUT / "permanent_nan_masks.npz",
        latitude=reference_lat,
        longitude=reference_lon,
        sst=masks["SST"],
        sss=masks["SSS"],
        ssh=masks["SSH"],
        current=masks["Current"],
        common=common,
    )

    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)

    print(
        f"\nOutput directory:\n{OUT}"
    )

    print(
        "\nNo source dataset was modified."
    )


if __name__ == "__main__":
    main()