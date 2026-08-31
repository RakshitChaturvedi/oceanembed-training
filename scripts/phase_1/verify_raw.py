from __future__ import annotations

import hashlib
import json
from pathlib import Path

import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "artifacts" / "raw_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_nc(path: Path) -> dict:
    try:
        with xr.open_dataset(path) as ds:
            return {
                "variables": list(ds.data_vars),
                "dimensions": {
                    name: int(size)
                    for name, size in ds.sizes.items()
                },
                "coords": list(ds.coords),
                "attrs": {
                    key: str(value)
                    for key, value in ds.attrs.items()
                },
            }
    except Exception as exc:
        return {"open_error": repr(exc)}


def main() -> None:
    files = sorted(RAW.rglob("*.nc"))

    if not files:
        raise SystemExit(f"No NetCDF files found under {RAW}")

    records = []

    for path in files:
        print(path)
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "netcdf": inspect_nc(path),
            }
        )

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST.open("w") as f:
        json.dump(
            {
                "phase": 1,
                "files": records,
            },
            f,
            indent=2,
        )

    print(f"\nManifest written to {MANIFEST}")


if __name__ == "__main__":
    main()