
from __future__ import annotations

from pathlib import Path
import copernicusmarine
from datetime import date

def describe_datasets(product_id: str):
    catalog = copernicusmarine.describe(
        product_id=product_id,
        disable_progress_bar=True,
    )

    if not catalog.products:
        raise RuntimeError(f"No Copernicus product found: {product_id}")

    return catalog.products[0].datasets


def resolve_daily_dataset(
    product_id: str,
    preferred_tokens: tuple[str, ...] = (),
) -> str:
    datasets = describe_datasets(product_id)

    candidates = [
        ds.dataset_id
        for ds in datasets
        if "P1D" in ds.dataset_id
        and all(token in ds.dataset_id for token in preferred_tokens)
    ]

    if not candidates:
        raise RuntimeError(
            f"No daily dataset found for {product_id}. "
            f"Candidates: {[d.dataset_id for d in datasets]}"
        )

    preferred = [
        ds for ds in candidates
        if "_my_" in ds or "_my" in ds
    ]

    return preferred[0] if preferred else candidates[0]


def resolve_variable(
    dataset_id: str,
    preferred_names: tuple[str, ...],
) -> str:
    catalog = copernicusmarine.describe(
        dataset_id=dataset_id,
        disable_progress_bar=True,
    )

    dataset = catalog.products[0].datasets[0]

    available = []

    for version in dataset.versions:
        for part in version.parts:
            for service in part.services:
                available.extend(
                    variable.short_name
                    for variable in service.variables
                )

    available = list(dict.fromkeys(available))

    for name in preferred_names:
        if name in available:
            return name

    raise RuntimeError(
        f"Could not resolve variable for {dataset_id}.\n"
        f"Available variables: {available}"
    )


def _years(start: str, end: str):
    start_year = int(start[:4])
    end_year = int(end[:4])

    return range(start_year, end_year + 1)


def _year_range(year: int, global_start: str, global_end: str):
    start = max(
        date(year, 1, 1),
        date.fromisoformat(global_start),
    )

    end = min(
        date(year, 12, 31),
        date.fromisoformat(global_end),
    )

    return (
        start.isoformat(),
        end.isoformat(),
    )


def subset_chunk(
    *,
    dataset_id: str,
    variables: list[str],
    output_dir: str,
    filename: str,
    start: str,
    end: str,
    bbox: dict,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> None:

    output_path = Path(output_dir) / filename

    if output_path.exists():
        print(f"SKIP: {output_path} already exists")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(
        f"\nDownloading:\n"
        f"  Dataset : {dataset_id}\n"
        f"  Variables: {variables}\n"
        f"  Time    : {start} -> {end}\n"
        f"  Output  : {output_path}"
    )

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,

        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],

        start_datetime=f"{start}T00:00:00",
        end_datetime=f"{end}T23:59:59",

        minimum_depth=min_depth,
        maximum_depth=max_depth,

        output_directory=output_dir,
        output_filename=filename,

        file_format="netcdf",

        skip_existing=True,

        netcdf_compression_level=4,
    )


def acquire_yearly(
    *,
    dataset_id: str,
    variables: list[str],
    output_dir: str,
    filename_prefix: str,
    start: str,
    end: str,
    bbox: dict,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> None:

    for year in _years(start, end):

        chunk_start, chunk_end = _year_range(
            year,
            start,
            end,
        )

        filename = (
            f"{filename_prefix}_{year}.nc"
        )

        subset_chunk(
            dataset_id=dataset_id,
            variables=variables,
            output_dir=output_dir,
            filename=filename,
            start=chunk_start,
            end=chunk_end,
            bbox=bbox,
            min_depth=min_depth,
            max_depth=max_depth,
        )


def acquire_copernicus(cfg: dict) -> None:

    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]
    bbox = cfg["domain"]

    sources = cfg["copernicus"]

    # ---------------------------------------------------------
    # GLORYS
    # ---------------------------------------------------------
    """
    print("\n=== GLORYS ===")

    glorys = sources["glorys"]

    acquire_yearly(
        dataset_id=glorys["dataset_id"],
        variables=glorys["variables"],
        output_dir=glorys["output_dir"],
        filename_prefix="glorys_ts",
        start=start,
        end=end,
        bbox=bbox,
        min_depth=0,
        max_depth=1000,
    )
    """
    # ---------------------------------------------------------
    # ARMOR3D
    # ---------------------------------------------------------

    print("\n=== ARMOR3D ===")

    armor = sources["armor3d"]

    acquire_yearly(
        dataset_id=armor["dataset_id"],
        variables=armor["variables"],
        output_dir=armor["output_dir"],
        filename_prefix="armor3d_ts",
        start=start,
        end=end,
        bbox=bbox,
        min_depth=0,
        max_depth=1000,
    )

    # ---------------------------------------------------------
    # SSH / SLA
    # ---------------------------------------------------------

    print("\n=== SSH / SLA ===")

    ssh = sources["ssh"]

    ssh_dataset = resolve_daily_dataset(
        ssh["product_id"],
        preferred_tokens=("my",),
    )

    ssh_variable = resolve_variable(
        ssh_dataset,
        tuple(ssh["preferred_variables"]),
    )

    acquire_yearly(
        dataset_id=ssh_dataset,
        variables=[ssh_variable],
        output_dir=ssh["output_dir"],
        filename_prefix="ssh_sla",
        start=start,
        end=end,
        bbox=bbox,
    )
"""
def describe_datasets(product_id: str):
    catalog = copernicusmarine.describe(
        product_id=product_id,
        disable_progress_bar=True,
    )
    if not catalog.products:
        raise RuntimeError(f"No Copernicus product found: {product_id}")
    return catalog.products[0].datasets


def resolve_daily_dataset(
    product_id: str,
    preferred_tokens: tuple[str, ...] = (),
) -> str:
    datasets = describe_datasets(product_id)

    candidates = [
        ds.dataset_id
        for ds in datasets
        if "P1D" in ds.dataset_id
        and all(token in ds.dataset_id for token in preferred_tokens)
    ]

    if not candidates:
        raise RuntimeError(
            f"No daily dataset found for {product_id}. "
            f"Candidates were: {[d.dataset_id for d in datasets]}"
        )

    # Prefer the multi-year/reprocessed dataset when several daily datasets exist.
    preferred = [
        ds for ds in candidates
        if "_my_" in ds or "_my" in ds
    ]
    return preferred[0] if preferred else candidates[0]


def resolve_variable(
    dataset_id: str,
    preferred_names: tuple[str, ...],
) -> str:
    catalog = copernicusmarine.describe(
        dataset_id=dataset_id,
        disable_progress_bar=True,
    )

    dataset = catalog.products[0].datasets[0]

    available: list[str] = []
    for version in dataset.versions:
        for part in version.parts:
            for service in part.services:
                available.extend(
                    variable.short_name
                    for variable in service.variables
                )

    available = list(dict.fromkeys(available))

    for name in preferred_names:
        if name in available:
            return name

    raise RuntimeError(
        f"Could not resolve variable for {dataset_id}.\n"
        f"Available variables: {available}"
    )


def subset(
    *,
    dataset_id: str,
    variables: list[str],
    output_dir: str,
    filename: str,
    start: str,
    end: str,
    bbox: dict,
    min_depth: float | None = None,
    max_depth: float | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    copernicusmarine.subset(
        dataset_id=dataset_id,
        variables=variables,
        minimum_longitude=bbox["lon_min"],
        maximum_longitude=bbox["lon_max"],
        minimum_latitude=bbox["lat_min"],
        maximum_latitude=bbox["lat_max"],
        start_datetime=f"{start}T00:00:00",
        end_datetime=f"{end}T23:59:59",
        minimum_depth=min_depth,
        maximum_depth=max_depth,
        output_directory=output_dir,
        output_filename=filename,
        file_format="netcdf",
        skip_existing=True,
        netcdf_compression_level=4,
    )


def acquire_copernicus(cfg: dict) -> None:
    start = cfg["dates"]["start"]
    end = cfg["dates"]["end"]
    bbox = cfg["domain"]
    sources = cfg["copernicus"]

    glorys = sources["glorys"]
    subset(
        dataset_id=glorys["dataset_id"],
        variables=glorys["variables"],
        output_dir=glorys["output_dir"],
        filename="glorys_ts.nc",
        start=start,
        end=end,
        bbox=bbox,
        min_depth=0,
        max_depth=1000,
    )

    armor = sources["armor3d"]
    subset(
        dataset_id=armor["dataset_id"],
        variables=armor["variables"],
        output_dir=armor["output_dir"],
        filename="armor3d_ts.nc",
        start=start,
        end=end,
        bbox=bbox,
        min_depth=0,
        max_depth=1000,
    )

    ssh = sources["ssh"]
    ssh_dataset = resolve_daily_dataset(
        ssh["product_id"],
        preferred_tokens=("my",),
    )
    ssh_variable = resolve_variable(
        ssh_dataset,
        tuple(ssh["preferred_variables"]),
    )

    subset(
        dataset_id=ssh_dataset,
        variables=[ssh_variable],
        output_dir=ssh["output_dir"],
        filename="ssh_sla.nc",
        start=start,
        end=end,
        bbox=bbox,
    )
"""