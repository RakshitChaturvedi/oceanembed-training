from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW =ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts"

def summarize_dataset(path: Path) -> dict:
    result ={"file":str(path.relative_to(ROOT)), "size_gb":round(path.stat().st_size/1024**3,3)}
    try:
        with xr.open_dataset(path, decode_times=True, chunks={}) as ds:
            result["dimensions"] = {k: int(v) for k,v in ds.sizes.items()}
            result["coordinates"] = list(ds.coords)
            result["varibales"] = {}
            for name, da in ds.data_vars.items():
                info = {
                    "dims": list(da.dims),
                    "shape": [int(x) for x in da.shape],
                    "dtype": str(da.dtype),
                    "units": da.attrs.get("units"),
                    "long_name": da.attrs.get("long_name"),
                }

                try:
                    sample = da.isel({
                        d: 0 
                        for d in da.dims if d not in {"lat", "latitude", "lon", "longitude"}
                    }).values
                    sample = np.asarray(sample)

                    if sample.size:
                        finite = sample[np.isfinite(sample)]
                        
                        if finite.size:
                            info["sample_min"] = float(finite.min())
                            info["sample_max"] = float(finite.max())
                except Exception as exc:
                    info["sample_error"] = repr(exc)

                variables = {}
                for name, var in ds.data_vars.items():
                    variables[name] = {
                        "dimensions": list(var.dims),
                        "shape": list(var.shape),
                        "dtype": str(var.dtype),
                        "units": var.attrs.get("units"),
                        "long_name": var.attrs.get("long_name"),
                    }

            for coord_name in ds.coords:
                coord = ds[coord_name]
                info = {"dims": list(coord.dims), "size": int(coord.size), "dtype": str(coord.dtype)}

                try:
                    values = coord.values

                    if values.size:
                        info["first"] = str(values.flat[0])
                        info["last"] = str(values.flat[-1])

                        if values.size > 1:
                            numeric = np.asarray(values)

                            if np.issubdtype(
                                numeric.dtype,
                                np.number,
                            ):
                                diffs = np.diff(numeric.astype(float))
                                finite = diffs[np.isfinite(diffs)]

                                if finite.size:
                                    info["spacing_min"] = float(
                                        finite.min()
                                    )
                                    info["spacing_max"] = float(
                                        finite.max()
                                    )
                except Exception as exc:
                    info["value_error"] = repr(exc)

                result.setdefault("coordinate_details", {})[
                    coord_name
                ] = info

            result["attrs"] = {
                k: str(v)
                for k, v in ds.attrs.items()
            }

    except Exception as exc:
        result["open_error"] = repr(exc)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source",
        default=None,
        help="Only inspect a source directory, e.g. wind",
    )

    args = parser.parse_args()

    root = RAW / args.source if args.source else RAW

    files = sorted(root.rglob("*.nc"))

    if not files:
        raise SystemExit(f"No NetCDF files found under {root}")

    print(f"Inspecting {len(files)} NetCDF files...")

    records = []

    for index, path in enumerate(files, start=1):
        print(
            f"[{index}/{len(files)}] "
            f"{path.relative_to(ROOT)}"
        )

        records.append(summarize_dataset(path))

    ARTIFACTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_name = (
        f"raw_inspection_{args.source}.json"
        if args.source
        else "raw_inspection.json"
    )

    output = ARTIFACTS / output_name

    with output.open("w") as f:
        json.dump(
            {
                "phase": 2,
                "file_count": len(records),
                "files": records,
            },
            f,
            indent=2,
        )

    print(f"\nWritten: {output}")


if __name__ == "__main__":
    main()

            