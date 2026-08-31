#!/usr/bin/env python3

"""
OceanEmbed — Phase 5 NaN Handling

Purpose:
    Prepare NaN-safe surface fields for Phase 5 tensor assembly.

READ ONLY on Phase-2 inputs.
Does NOT open GLORYS or ARMOR3D.

Policy:
    1. Identify permanent/common spatial missingness.
    2. Preserve an ocean/land mask.
    3. Fill intermittent missing values temporally.
    4. Fill leading/trailing temporal gaps from nearest valid observation.
    5. Never modify the original Phase-2 files.
    6. Verify that produced fields contain zero NaNs/Infs.

Outputs:
    data/processed/phase5/
        sst_filled.nc
        sss_filled.nc
        ssh_filled.nc
        currents_filled.nc
        ocean_mask.nc
        missingness_masks.nc

NOTE:
    This script deliberately does not perform normalization.
    Normalization belongs to Phase 6.
"""

from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "processed" / "phase2"
OUTPUT_DIR = ROOT / "data" / "processed" / "phase5"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "sst": {
        "file": INPUT_DIR / "sst.nc",
        "variable": "sst",
    },
    "sss": {
        "file": INPUT_DIR / "sss.nc",
        "variable": "sss",
    },
    "ssh": {
        "file": INPUT_DIR / "ssh.nc",
        "variable": "ssh",
    },
    "current_u": {
        "file": INPUT_DIR / "currents.nc",
        "variable": "current_u",
    },
    "current_v": {
        "file": INPUT_DIR / "currents.nc",
        "variable": "current_v",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def permanent_nan_mask(da):
    """
    True where a spatial cell is NaN for every timestep.
    """
    return da.isnull().all(dim="time").values


def fill_temporally_tiled(
    da,
    lat_chunk=20,
    lon_chunk=20,
    max_gap_steps=30,
):
    """
    Fill temporal NaNs tile-by-tile.

    Policy
    ------
    1. Interior gaps <= max_gap_steps:
         linear interpolation.

    2. Interior gaps > max_gap_steps:
         nearest valid temporal observation.

    3. Leading gaps:
         nearest valid observation.

    4. Trailing gaps:
         nearest valid observation.

    5. Permanently missing cells:
         remain NaN.

    The permanent NaNs are handled later by the caller and encoded
    separately using ocean_mask.
    """

    nlat = da.sizes["latitude"]
    nlon = da.sizes["longitude"]

    output_rows = []

    for lat_start in range(0, nlat, lat_chunk):
        lat_end = min(lat_start + lat_chunk, nlat)

        output_tiles = []

        for lon_start in range(0, nlon, lon_chunk):
            lon_end = min(lon_start + lon_chunk, nlon)

            print(
                f"      tile "
                f"lat={lat_start}:{lat_end} "
                f"lon={lon_start}:{lon_end}"
            )

            tile = da.isel(
                latitude=slice(lat_start, lat_end),
                longitude=slice(lon_start, lon_end),
            ).load()

            values = tile.values
            filled = values.copy()

            ntime, nt_lat, nt_lon = values.shape

            for i in range(nt_lat):
                for j in range(nt_lon):

                    series = values[:, i, j]

                    nan_mask = np.isnan(series)

                    if not nan_mask.any():
                        continue

                    valid_idx = np.flatnonzero(~nan_mask)

                    # Permanently missing spatial cell.
                    if len(valid_idx) == 0:
                        continue

                    # -------------------------------------------------
                    # Leading NaNs
                    # -------------------------------------------------

                    first_valid = valid_idx[0]

                    if first_valid > 0:
                        filled[:first_valid, i, j] = (
                            series[first_valid]
                        )

                    # -------------------------------------------------
                    # Trailing NaNs
                    # -------------------------------------------------

                    last_valid = valid_idx[-1]

                    if last_valid < ntime - 1:
                        filled[last_valid + 1:, i, j] = (
                            series[last_valid]
                        )

                    # -------------------------------------------------
                    # Interior NaN runs
                    # -------------------------------------------------

                    interior_nan = nan_mask.copy()

                    interior_nan[:first_valid] = False
                    interior_nan[last_valid + 1:] = False

                    padded = np.concatenate(
                        ([False], interior_nan, [False])
                    )

                    changes = np.diff(
                        padded.astype(np.int8)
                    )

                    starts = np.where(changes == 1)[0]
                    ends = np.where(changes == -1)[0]

                    for start, end in zip(starts, ends):

                        gap_length = end - start

                        left = series[start - 1]
                        right = series[end]

                        if (
                            np.isnan(left)
                            or np.isnan(right)
                        ):
                            continue

                        # -------------------------------------------------
                        # Short gap → linear interpolation
                        # -------------------------------------------------

                        if gap_length <= max_gap_steps:

                            filled[start:end, i, j] = np.linspace(
                                left,
                                right,
                                gap_length + 2,
                            )[1:-1]

                        # -------------------------------------------------
                        # Long gap → nearest observation
                        # -------------------------------------------------

                        else:

                            left_distance = np.arange(
                                gap_length
                            ) + 1

                            right_distance = (
                                gap_length
                                - np.arange(gap_length)
                            )

                            use_left = (
                                left_distance <= right_distance
                            )

                            gap_values = np.where(
                                use_left,
                                left,
                                right,
                            )

                            filled[start:end, i, j] = gap_values

            result_tile = xr.DataArray(
                filled,
                dims=tile.dims,
                coords=tile.coords,
                attrs=tile.attrs,
                name=tile.name,
            )

            output_tiles.append(result_tile)

            del tile
            del values
            del filled

        output_rows.append(
            xr.concat(
                output_tiles,
                dim="longitude",
            )
        )

    return xr.concat(
        output_rows,
        dim="latitude",
    )


def write_dataset(
    name,
    da,
    output_path,
    attrs=None,
    encoding=None,
):
    # Tiled processing can leave the resulting DataArray unnamed.
    if da.name != name:
        da = da.rename(name)

    if attrs:
        da.attrs.update(attrs)

    # IMPORTANT:
    # Do not reuse da.encoding. It can contain stale keys such as None.
    clean_encoding = {}

    if encoding is not None:
        clean_encoding = {
            key: value
            for key, value in encoding.items()
            if key is not None
        }

    ds = da.to_dataset(name=name)

    ds.to_netcdf(
        output_path,
        encoding=clean_encoding,
    )

    ds.close()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 100)
    print("OceanEmbed — Phase 5 NaN Handling")
    print("=" * 100)
    print()
    print("READ ONLY INPUTS")
    print("GLORYS and ARMOR3D are NOT opened.")
    print()

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------

    for name, cfg in DATASETS.items():

        if not cfg["file"].exists():
            raise FileNotFoundError(
                f"Missing input for {name}: {cfg['file']}"
            )

    # -------------------------------------------------------------------------
    # Load only the four surface datasets needed for mask construction.
    #
    # IMPORTANT:
    # We use xarray lazy access. We do not call .load() on the full arrays.
    # -------------------------------------------------------------------------

    print("=" * 100)
    print("BUILDING PERMANENT MASKS")
    print("=" * 100)

    masks = {}

    opened = {}

    try:

        for name, cfg in DATASETS.items():

            if cfg["file"] not in opened:
                opened[cfg["file"]] = xr.open_dataset(
                    cfg["file"],
                    chunks={"time": 1},
                )

            ds = opened[cfg["file"]]
            da = ds[cfg["variable"]]

            mask = permanent_nan_mask(da)

            masks[name] = mask

            print(
                f"{name:12s}: "
                f"{mask.sum():6d} / {mask.size} permanent NaN cells"
            )

        # ---------------------------------------------------------------------
        # Common permanent mask
        #
        # A cell is considered common land/masked only when ALL FOUR
        # physical surface datasets agree that it is permanently missing.
        # ---------------------------------------------------------------------

        common_land_mask = (
            masks["sst"]
            & masks["sss"]
            & masks["ssh"]
            & masks["current_u"]
            & masks["current_v"]
        )

        print()
        print(
            f"COMMON PERMANENT MASK : "
            f"{common_land_mask.sum()} / {common_land_mask.size} cells"
        )

        # ---------------------------------------------------------------------
        # Save common ocean mask.
        # ---------------------------------------------------------------------

        reference_ds = opened[
            DATASETS["sst"]["file"]
        ]

        lat = reference_ds["latitude"]
        lon = reference_ds["longitude"]

        ocean_mask = xr.DataArray(
            ~common_land_mask,
            dims=("latitude", "longitude"),
            coords={
                "latitude": lat,
                "longitude": lon,
            },
            name="ocean_mask",
            attrs={
                "description":
                    "1 where the cell is not permanently missing "
                    "in all Phase-5 physical surface datasets; "
                    "0 where all datasets are permanently missing.",
                "phase": "phase5",
            },
        )

        ocean_mask.to_dataset().to_netcdf(
            OUTPUT_DIR / "ocean_mask.nc"
        )

        print(
            f"Wrote: {OUTPUT_DIR / 'ocean_mask.nc'}"
        )

        # ---------------------------------------------------------------------
        # Process individual variables.
        # ---------------------------------------------------------------------

        print()
        print("=" * 100)
        print("TEMPORAL NaN HANDLING")
        print("=" * 100)

        missingness = {}

        for name, cfg in DATASETS.items():

            print()
            print("-" * 100)
            print(f"PROCESSING: {name}")
            print("-" * 100)

            ds = opened[cfg["file"]]
            da = ds[cfg["variable"]]

            # -------------------------------------------------------------
            # Original missingness mask.
            # -------------------------------------------------------------

            original_nan = da.isnull()

            # -------------------------------------------------------------
            # Fill along time.
            #
            # This handles intermittent gaps and temporal edge gaps.
            # -------------------------------------------------------------

            filled = fill_temporally_tiled(da, lat_chunk=20, lon_chunk=20, max_gap_steps=30)

            # Permanent missingness for this variable.
            permanent_mask = permanent_nan_mask(da)

            # Convert the 2-D NumPy mask back to an xarray DataArray.
            permanent_mask_da = xr.DataArray(
                permanent_mask,
                dims=("latitude", "longitude"),
                coords={
                    "latitude": da.latitude,
                    "longitude": da.longitude,
                },
            )

            # Only permanent missing cells become numerical zero.
            filled = xr.where(
                permanent_mask_da,
                0.0,
                filled,
            )

            # -------------------------------------------------------------
            # Identify values that STILL remain NaN.
            #
            # These are cells with no valid observation anywhere in time.
            # -------------------------------------------------------------

            remaining_nan = filled.isnull()

            remaining_count = int(
                remaining_nan.sum().compute()
            )

            # -------------------------------------------------------------
            # Permanently unavailable cells have no temporal information.
            #
            # Numerically encode them as zero.
            #
            # IMPORTANT:
            # The separate ocean_mask preserves the distinction between
            # this numerical zero and a genuine physical value.
            # -------------------------------------------------------------


            # -------------------------------------------------------------
            # Verify.
            # -------------------------------------------------------------

            nan_count = int(
                filled.isnull().sum().compute()
            )

            inf_count = int(
                xr.apply_ufunc(
                    np.isinf,
                    filled,
                    dask="parallelized",
                    output_dtypes=[bool],
                ).sum().compute()
            )

            original_count = int(
                original_nan.sum().compute()
            )

            filled_count = int(
                (original_nan & ~remaining_nan).sum().compute()
            )

            print(f"Original NaNs       : {original_count:,}")
            print(f"Filled values       : {filled_count:,}")
            print(f"Still unavailable  : {remaining_count:,}")
            print(f"Final NaNs          : {nan_count:,}")
            print(f"Final Infs          : {inf_count:,}")

            if nan_count != 0:
                raise RuntimeError(
                    f"{name}: NaNs remain after processing."
                )

            if inf_count != 0:
                raise RuntimeError(
                    f"{name}: Infs remain after processing."
                )

            # -------------------------------------------------------------
            # Store missingness mask.
            #
            # 1 = originally missing
            # 0 = originally observed
            # -------------------------------------------------------------

            missingness[name] = original_nan

            output_file = OUTPUT_DIR / f"{name}_filled.nc"

            write_dataset(
                name,
                filled,
                output_file,
                attrs={
                    "oceanembed_phase": "phase5",
                    "oceanembed_nan_policy":
                        "temporal linear interpolation; "
                        "nearest valid value for temporal edges; "
                        "unavailable cells encoded as 0",
                },
            )

            print(f"Wrote: {output_file}")

        # ---------------------------------------------------------------------
        # Save missingness masks.
        # ---------------------------------------------------------------------

        print()
        print("=" * 100)
        print("WRITING MISSINGNESS MASKS")
        print("=" * 100)

        mask_vars = {}

        for name, mask in missingness.items():
            mask_vars[name] = mask.astype(np.uint8).rename(
                f"{name}_original_missing"
            )

        missingness_ds = xr.Dataset(mask_vars)

        missingness_ds.to_netcdf(
            OUTPUT_DIR / "missingness_masks.nc"
        )

        print(
            f"Wrote: {OUTPUT_DIR / 'missingness_masks.nc'}"
        )

    finally:

        for ds in opened.values():
            ds.close()

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("PHASE 5 NaN HANDLING COMPLETE")
    print("=" * 100)

    print()
    print("Outputs:")
    print(f"  {OUTPUT_DIR / 'sst_filled.nc'}")
    print(f"  {OUTPUT_DIR / 'sss_filled.nc'}")
    print(f"  {OUTPUT_DIR / 'ssh_filled.nc'}")
    print(f"  {OUTPUT_DIR / 'current_u_filled.nc'}")
    print(f"  {OUTPUT_DIR / 'current_v_filled.nc'}")
    print(f"  {OUTPUT_DIR / 'ocean_mask.nc'}")
    print(f"  {OUTPUT_DIR / 'missingness_masks.nc'}")

    print()
    print("GLORYS/ARMOR3D: NOT TOUCHED")
    print("Phase 6 normalization: NOT PERFORMED")
    print("Original Phase-2 files: UNMODIFIED")
    print()


if __name__ == "__main__":
    main()