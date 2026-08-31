from __future__ import annotations

import sys
import re
from pathlib import Path
from datetime import datetime

import xarray as xr
import pandas as pd
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from src.data.preprocessing.schema import open_standardized
from src.data.preprocessing.spatial import subset_domain
from src.data.preprocessing.temporal import (
    align_daily,
    daily_mean,
    derive_wind_speed,
    deduplicate_time
)
from src.data.preprocessing.regrid import (
    create_target_grid,
    regrid_dataset,
    create_regridder
)
from src.data.preprocessing.harmonize import (
    harmonize_sst,
    harmonize_sss,
    harmonize_ssh,
    harmonize_wind,
    harmonize_currents,
    harmonize_subsurface,
    finalize_dataset,
)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

def load_config() -> dict:
    path = ROOT / "configs" / "preprocessing.yaml"

    with path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------

def save_dataset(
    ds: xr.Dataset,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    encoding = {}

    for variable in ds.data_vars:
        encoding[variable] = {
            "zlib": True,
            "complevel": 4,
        }

    print(f"    Writing: {output_path}")

    ds.to_netcdf(
        output_path,
        encoding=encoding,
    )

    print(f"    Finished: {output_path}")


# ---------------------------------------------------------------------------
# COMMON PER-FILE PROCESSING
# ---------------------------------------------------------------------------

def filter_files_by_time(
        files: list[Path],
        *, start: str, end: str
) -> list[Path]: 
    start_date = pd.Timestamp(start).date()
    end_date = pd.Timestamp(end).date()

    filtered = []

    for path in files:
        match = re.search(r"\d{8}", path.name)

        if not match:
            filtered.append(path)
            continue

        file_date = datetime.strptime(
            match.group(0),
            "%Y%m%d",
        ).date()

        if start_date <= file_date <= end_date:
            filtered.append(path)

    return filtered

def prepare_file(
    path: Path,
    *,
    dataset_name: str,
    cfg: dict,
    target_grid: xr.Dataset,
    regrid_method: str,
) -> xr.Dataset:

    ds = open_standardized(
        path,
        dataset_name=dataset_name,
    )

    print("      Subsetting domain")

    ds = subset_domain(
        ds,
        lat_min=cfg["spatial"]["lat_min"],
        lat_max=cfg["spatial"]["lat_max"],
        lon_min=cfg["spatial"]["lon_min"],
        lon_max=cfg["spatial"]["lon_max"],
    )
    print(
        f"      Domain size: "
        f"{ds.sizes.get('latitude')} × "
        f"{ds.sizes.get('longitude')}"
    )

    print(
        f"      Regridding: {regrid_method}"
    )

    ds = regrid_dataset(
        ds,
        target_grid,
        method=regrid_method,
    )

    return ds


# ---------------------------------------------------------------------------
# SURFACE DATASET
# ---------------------------------------------------------------------------

def process_surface_files(
    *,
    files: list[Path],
    dataset_name: str,
    cfg: dict,
    output_path: Path,
    harmonizer,
    regrid_method: str,
    temporal_mode: str = "interpolate"
) -> None:

    if not files:
        raise FileNotFoundError(
            f"No {dataset_name} files found."
        )

    target_grid = create_target_grid(
        **cfg["spatial"],
    )

    processed = []

    print(
        f"  Processing {len(files)} files"
    )

    # ---------------------------------------------------------
    # Process each file independently
    # ---------------------------------------------------------

    for index, path in enumerate(files, start=1):

        print(
            f"\n    [{index}/{len(files)}] "
            f"{path.name}"
        )

        ds = prepare_file(
            path,
            dataset_name=dataset_name,
            cfg=cfg,
            target_grid=target_grid,
            regrid_method=regrid_method,
        )

        processed.append(ds)

    # ---------------------------------------------------------
    # Concatenate
    # ---------------------------------------------------------

    print(
        "\n  Concatenating processed files"
    )

    ds = xr.concat(
        processed,
        dim="time",
    )

    ds = ds.sortby("time")

    # ---------------------------------------------------------
    # Deduplicate timestamps
    # ---------------------------------------------------------

    print(
        "  Removing duplicate timestamps"
    )

    _, unique_indices = np.unique(
        ds["time"].values,
        return_index=True,
    )

    ds = ds.isel(
        time=np.sort(unique_indices)
    )

    # ---------------------------------------------------------
    # Daily temporal alignment
    # ---------------------------------------------------------

    print(
        f"  Temporal processing: {temporal_mode}"
    )

    if temporal_mode in ("interpolate", "align"):
        ds = align_daily(
            ds,
            start=cfg["temporal"]["start"],
            end=cfg["temporal"]["end"],
        )

    elif temporal_mode == "mean":
        ds = daily_mean(ds)

    else:
        raise ValueError(
            f"Unknown temporal_mode: {temporal_mode}"
        )
    # ---------------------------------------------------------
    # Harmonization
    # ---------------------------------------------------------

    ds = harmonizer(ds)

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------

    ds = finalize_dataset(
        ds,
        dataset_name=dataset_name,
        spatial_cfg=cfg["spatial"],
    )

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    save_dataset(
        ds,
        output_path,
    )


# ---------------------------------------------------------------------------
# SUBSURFACE
# ---------------------------------------------------------------------------
def process_subsurface(
    files,
    output_path,
    cfg,
    dataset_name,
    temperature_name,
    salinity_name,
    regrid_method="bilinear",
):
    import gc
    import shutil
    import tempfile

    target_grid = create_target_grid(
        **cfg["spatial"]
    )

    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f"oceanembed_{dataset_name}_"
        )
    )

    print(
        f"  Processing {len(files)} files"
    )

    try:

        processed_files = []

        for index, path in enumerate(
            files,
            start=1,
        ):
            print(
                f"\n    [{index}/{len(files)}] "
                f"{path.name}"
            )

            ds = open_standardized(
                path,
                dataset_name=dataset_name,
            )

            # -------------------------------------------------
            # Spatial subset BEFORE regridding
            # -------------------------------------------------

            print("      Subsetting domain")

            ds = subset_domain(
                ds,
                lat_min=cfg["spatial"]["lat_min"],
                lat_max=cfg["spatial"]["lat_max"],
                lon_min=cfg["spatial"]["lon_min"],
                lon_max=cfg["spatial"]["lon_max"],
            )

            print(
                f"      Domain size: "
                f"{ds.sizes['latitude']} × "
                f"{ds.sizes['longitude']}"
            )

            # -------------------------------------------------
            # Create ONE regridder for this source file
            # -------------------------------------------------

            regridder = create_regridder(
                ds,
                target_grid,
                method=regrid_method,
            )

            time_size = ds.sizes["time"]

            # Smaller chunks = lower peak RAM.
            chunk_size = 10

            chunk_files = []

            for start in range(
                0,
                time_size,
                chunk_size,
            ):
                end = min(
                    start + chunk_size,
                    time_size,
                )

                print(
                    f"      Regridding time "
                    f"{start}:{end} / {time_size}"
                )

                chunk = ds.isel(
                    time=slice(start, end)
                )

                # -------------------------------------------------
                # Regrid using the SAME regridder
                # -------------------------------------------------

                chunk = regrid_dataset(
                    chunk,
                    target_grid,
                    method=regrid_method,
                    regridder=regridder,
                )

                # -------------------------------------------------
                # Harmonize this chunk immediately
                # -------------------------------------------------

                chunk = harmonize_subsurface(
                    chunk,
                    temperature_name=temperature_name,
                    salinity_name=salinity_name,
                )

                # -------------------------------------------------
                # Write chunk to disk immediately
                # -------------------------------------------------

                chunk_path = (
                    temp_root
                    / f"{dataset_name}_"
                      f"{index:04d}_"
                      f"{start:05d}.nc"
                )

                chunk.to_netcdf(
                    chunk_path,
                    engine="netcdf4",
                )

                chunk_files.append(chunk_path)

                # Explicitly release memory
                del chunk

                gc.collect()

            # -------------------------------------------------
            # Release source dataset + regridder
            # -------------------------------------------------

            del regridder
            del ds

            gc.collect()

            # -------------------------------------------------
            # Combine this source file's chunks lazily
            # -------------------------------------------------

            source_ds = xr.open_mfdataset(
                chunk_files,
                combine="by_coords",
                parallel=False,
                chunks="auto",
            )

            # Write a single temporary file for this source
            source_output = (
                temp_root
                / f"{dataset_name}_"
                  f"{index:04d}.nc"
            )

            source_ds.to_netcdf(
                source_output,
                engine="netcdf4",
            )

            source_ds.close()

            del source_ds

            gc.collect()

            processed_files.append(
                source_output
            )

            # Remove individual chunks
            for chunk_file in chunk_files:
                chunk_file.unlink(
                    missing_ok=True
                )

            gc.collect()

        # =====================================================
        # Combine all source files
        # =====================================================

        print(
            "\n  Concatenating subsurface files"
        )

        ds = xr.open_mfdataset(
            processed_files,
            combine="by_coords",
            parallel=False,
            chunks="auto",
        )

        # -----------------------------------------------------
        # Remove duplicate timestamps
        # -----------------------------------------------------

        _, unique_indices = np.unique(
            ds.time.values,
            return_index=True,
        )

        unique_indices = np.sort(
            unique_indices
        )

        if (
            len(unique_indices)
            != ds.sizes["time"]
        ):
            removed = (
                ds.sizes["time"]
                - len(unique_indices)
            )

            print(
                f"  Removing {removed} "
                f"duplicate timestamps"
            )

            ds = ds.isel(
                time=unique_indices
            )

        ds = ds.sortby("time")

        # -----------------------------------------------------
        # Restrict to common temporal range
        # -----------------------------------------------------

        ds = align_daily(
            ds,
            start=cfg["temporal"]["start"],
            end=cfg["temporal"]["end"],
        )

        # -----------------------------------------------------
        # Finalize
        # -----------------------------------------------------

        ds = finalize_dataset(
            ds,
            dataset_name=dataset_name,
            spatial_cfg=cfg["spatial"],
        )

        save_dataset(
            ds,
            output_path,
        )

        ds.close()

    finally:

        # -----------------------------------------------------
        # Always clean temporary files
        # -----------------------------------------------------

        shutil.rmtree(
            temp_root,
            ignore_errors=True,
        )


# ---------------------------------------------------------------------------
# WIND
# ---------------------------------------------------------------------------

def process_wind(
    cfg: dict,
    output_path: Path,
    files: list[Path]
) -> None:
    if not files:
        raise FileNotFoundError(
            "No wind files found."
        )

    target_grid = create_target_grid(
        **cfg["spatial"],
    )

    processed = []

    print(
        f"  Processing {len(files)} wind files"
    )

    for index, path in enumerate(files, start=1):

        print(
            f"\n    [{index}/{len(files)}] "
            f"{path.name}"
        )

        ds = prepare_file(
            path,
            dataset_name="wind",
            cfg=cfg,
            target_grid=target_grid,
            regrid_method=cfg["regrid"]["method"].get(
                "wind",
                "bilinear",
            ),
        )

        print("      Daily averaging")

        ds = daily_mean(ds)

        ds = harmonize_wind(ds)

        ds = derive_wind_speed(ds)

        processed.append(ds)

    print("\n  Concatenating processed wind files")

    ds = xr.concat( processed, dim="time")
    ds = ds.sortby("time")
    ds = deduplicate_time(ds)

    _, unique_indices = __import__("numpy").unique(
        ds["time"].values,
        return_index=True,
    )

    ds = ds.isel(
        time=sorted(unique_indices)
    )

    # Verify daily coverage.
    expected_start = pd.Timestamp(
        cfg["temporal"]["start"]
    )

    expected_end = pd.Timestamp(
        cfg["temporal"]["end"]
    )

    expected_days = len(
        pd.date_range(
            expected_start,
            expected_end,
            freq="1D",
        )
    )

    actual_days = ds.sizes["time"]

    if actual_days != expected_days:
        raise ValueError(
            f"Wind temporal coverage mismatch: "
            f"expected {expected_days} days, "
            f"got {actual_days}."
        )

    ds = finalize_dataset(
        ds,
        dataset_name="wind",
        spatial_cfg=cfg["spatial"],
    )

    save_dataset(
        ds,
        output_path,
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:

    full_cfg = load_config()
    cfg = full_cfg["phase2"]

    output_root = ROOT / cfg["output_root"]

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("OceanEmbed Phase 2")
    print("Domain, Temporal & Resolution Alignment")
    print("=" * 70)

    print(
        f"Domain: "
        f"{cfg['spatial']['lat_min']} -> "
        f"{cfg['spatial']['lat_max']} N, "
        f"{cfg['spatial']['lon_min']} -> "
        f"{cfg['spatial']['lon_max']} E"
    )

    print(
        f"Resolution: "
        f"{cfg['spatial']['resolution']}°"
    )

    print(
        f"Time: "
        f"{cfg['temporal']['start']} -> "
        f"{cfg['temporal']['end']}"
    )

    print(
        f"Output: {output_root}"
    )

    # =======================================================================
    # 1. SST
    # =======================================================================

    files_sst = sorted(
        (ROOT / "data/raw/sst").glob("*.nc")
    )

    files_sst = filter_files_by_time(
        files_sst,
        start=cfg["temporal"]["start"],
        end=cfg["temporal"]["end"],
    )

    sst_output = output_root / "sst.nc"

    if sst_output.exists():
        print("\n[1/7] SST")
        print("  Output already exists — skipping.")
    else:
        print("\n" + "=" * 70)
        print("[1/7] SST")
        print("=" * 70)
        
        process_surface_files(
            files=files_sst,
            dataset_name="sst",
            cfg=cfg,
            output_path=output_root / "sst.nc",
            harmonizer=harmonize_sst,
            temporal_mode="align",
            regrid_method="bilinear",
        )

    # =======================================================================
    # 2. SSS
    # =======================================================================

    print("\n" + "=" * 70)
    print("[2/7] SSS")
    print("=" * 70)

    files_sss = sorted((ROOT / "data/raw/sss").glob("*.nc"))
    files_sss = filter_files_by_time(
        files_sss,
        start=cfg["temporal"]["start"],
        end=cfg["temporal"]["end"],
    )

    sss_output = output_root / "sss.nc"

    if sss_output.exists():
        print("  Output already exists — skipping.")
    else:
        process_surface_files(
            files=files_sss,
            dataset_name="sss",
            cfg=cfg,
            output_path=output_root / "sss.nc",
            harmonizer=harmonize_sss,
            temporal_mode="align",
            regrid_method="bilinear",
        )

    # =======================================================================
    # 3. SSH
    # =======================================================================

    print("\n" + "=" * 70)
    print("[3/7] SSH")
    print("=" * 70)

    files_ssh = sorted((ROOT / "data/raw/ssh").glob("*.nc"))
    files_ssh = filter_files_by_time(
        files_ssh,
        start=cfg["temporal"]["start"],
        end=cfg["temporal"]["end"],
    )


    ssh_output = output_root / "ssh.nc"

    if ssh_output.exists():
        print("  Output already exists — skipping.")
    else:
        process_surface_files(
            files=files_ssh,
            dataset_name="ssh",
            cfg=cfg,
            output_path=output_root / "ssh.nc",
            harmonizer=harmonize_ssh,
            temporal_mode="align",
            regrid_method="bilinear",
        )

    # =======================================================================
    # 4. WIND
    # =======================================================================

    print("\n" + "=" * 70)
    print("[4/7] WIND")
    print("=" * 70)
    files_wind = sorted((ROOT / "data/raw/wind").glob("*.nc"))
    files_wind = filter_files_by_time(
        files_wind,
        start=cfg["temporal"]["start"],
        end=cfg["temporal"]["end"],
    )

    wind_output = output_root / "wind.nc"

    if wind_output.exists():
        print("  Output already exists — skipping.")
    else:
        process_wind(
            cfg,
            output_root / "wind.nc",
            files=files_wind
        )

    # =======================================================================
    # 5. CURRENTS
    # =======================================================================

    print("\n" + "=" * 70)
    print("[5/7] SURFACE CURRENTS")
    print("=" * 70)

    files_curr = sorted((ROOT / "data/raw/currents").glob("*.nc"))
    files_curr = filter_files_by_time(
        files_curr,
        start=cfg["temporal"]["start"],
        end=cfg["temporal"]["end"],
    )
    curr_output = output_root / "currents.nc"

    if curr_output.exists():
        print("  Output already exists — skipping.")
    else:
        process_surface_files(
            files=files_curr,
            dataset_name="currents",
            cfg=cfg,
            output_path=output_root / "currents.nc",
            harmonizer=harmonize_currents,
            temporal_mode="align",
            regrid_method="bilinear",
        )

    # =======================================================================
    # 6. ARMOR3D
    # =======================================================================

    print("\n" + "=" * 70)
    print("[6/7] ARMOR3D")
    print("=" * 70)

    armor3d_output = output_root / "armor3d.nc"

    if armor3d_output.exists():
        print("  Output already exists — skipping.")
    else:
        process_subsurface(
            files=sorted(
                (ROOT / "data/raw/armor3d").glob("*.nc")
            ),
            dataset_name="armor3d",
            temperature_name="to",
            salinity_name="so",
            cfg=cfg,
            output_path=output_root / "armor3d.nc",
        )

    # =======================================================================
    # 7. GLORYS
    # =======================================================================

    print("\n" + "=" * 70)
    print("[7/7] GLORYS")
    print("=" * 70)

    process_subsurface(
        files=sorted(
            (ROOT / "data/raw/glorys").glob("*.nc")
        ),
        dataset_name="glorys",
        temperature_name="thetao",
        salinity_name="so",
        cfg=cfg,
        output_path=output_root / "glorys.nc",
    )

    # =======================================================================
    # DONE
    # =======================================================================

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETE")
    print("=" * 70)

    print("\nGenerated datasets:")

    for path in sorted(output_root.glob("*.nc")):
        print(f"  {path}")


if __name__ == "__main__":
    main()