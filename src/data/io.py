from __future__ import annotations
from pathlib import Path

import xarray as xr

def open_source(path_pattern: str, variable: str, chunks: dict | None = None) -> xr.DataArray:
    # open one or more netcdf files, return single DataArray. Files mustn't expose compatible coords.
    paths = sorted(Path().glob(path_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {path_pattern}")

    datasets = xr.open_mfdataset(
        [str(p) for p in paths], combine="by_coords", chunks=chunks or {}, parallel=True
    )
    if variable not in datasets:
        raise KeyError(
            f"Variable '{variable}' not found.\nAvailable variables: {list(datasets.data_vars)}"
        )
    return datasets[variable]

def normalize_coordinates(da: xr.DataArray) -> xr.DataArray:
    # normalize common coord naming conventions
    rename = {}
    if "latitude" in da.coords:
        rename["latitude"] = "lat"
    if "longitude" in da.coords:
        rename["longitude"] = "lon"
    if "depth" in da.coords:
        rename["depth"] = "depth"
    if "time" not in da.coords:
        raise ValueError("Dataset has no time coordinate.")
    da = da.rename(rename)

    if float(da.lon.max()) > 180:
        lon = ((da.lon + 180)%360) - 180
        da = da.assign_coords(lon=lon).sortby("lon")
    return da
