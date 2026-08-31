from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "input_tensor.nc"
)

EXPECTED_SHAPE = (1419, 101, 241, 12)

EXPECTED_CHANNELS = [
    "SST",
    "SSS",
    "SSHA",
    "wind_U",
    "wind_V",
    "current_U",
    "current_V",
    "wind_stress_curl",
    "latitude",
    "longitude",
    "sin_day_of_year",
    "cos_day_of_year",
]


def main() -> None:

    print("=" * 100)
    print("OceanEmbed — Phase 5 Input Tensor Verification")
    print("=" * 100)

    print()
    print("INPUT")
    print(f"  {INPUT}")

    try:
        with xr.open_dataset(INPUT) as ds:

            print()
            print("=" * 100)
            print("DATASET")
            print("=" * 100)

            print(f"Variables : {list(ds.data_vars)}")
            print(f"Dimensions: {dict(ds.sizes)}")

            if "input_tensor" not in ds:
                raise ValueError(
                    "Dataset does not contain 'input_tensor'."
                )

            tensor = ds["input_tensor"]

            # --------------------------------------------------------------
            # Dimensions
            # --------------------------------------------------------------

            expected_dims = (
                "time",
                "latitude",
                "longitude",
                "channel",
            )

            print()
            print("Dimensions:")
            print(f"  Expected : {expected_dims}")
            print(f"  Actual   : {tensor.dims}")

            if tensor.dims != expected_dims:
                raise ValueError(
                    f"Dimension mismatch: {tensor.dims}"
                )

            # --------------------------------------------------------------
            # Shape
            # --------------------------------------------------------------

            print()
            print("Shape:")
            print(f"  Expected : {EXPECTED_SHAPE}")
            print(f"  Actual   : {tensor.shape}")

            if tensor.shape != EXPECTED_SHAPE:
                raise ValueError(
                    f"Shape mismatch: {tensor.shape}"
                )

            # --------------------------------------------------------------
            # Coordinates
            # --------------------------------------------------------------

            for coord in (
                "time",
                "latitude",
                "longitude",
                "channel",
            ):
                if coord not in tensor.coords:
                    raise ValueError(
                        f"Missing coordinate: {coord}"
                    )

                print(f"  {coord:<12}: OK")

            # --------------------------------------------------------------
            # Numerical integrity
            # --------------------------------------------------------------

            print()
            print("=" * 100)
            print("NUMERICAL VALIDATION")
            print("=" * 100)

            values = tensor.values

            nan_count = int(np.isnan(values).sum())
            inf_count = int(np.isinf(values).sum())

            print(f"NaNs : {nan_count:,}")
            print(f"Infs : {inf_count:,}")

            if nan_count != 0:
                raise ValueError(
                    f"Tensor contains {nan_count:,} NaNs."
                )

            if inf_count != 0:
                raise ValueError(
                    f"Tensor contains {inf_count:,} Infs."
                )

            # --------------------------------------------------------------
            # Channel count
            # --------------------------------------------------------------

            channel_count = tensor.sizes["channel"]

            print()
            print("Channel count:")
            print(f"  Expected : {len(EXPECTED_CHANNELS)}")
            print(f"  Actual   : {channel_count}")

            if channel_count != len(EXPECTED_CHANNELS):
                raise ValueError(
                    "Channel count mismatch."
                )

            # --------------------------------------------------------------
            # Metadata
            # --------------------------------------------------------------

            print()
            print("=" * 100)
            print("METADATA")
            print("=" * 100)

            print(
                f"Phase          : "
                f"{ds.attrs.get('oceanembed_phase', 'MISSING')}"
            )

            print(
                f"Normalization  : "
                f"{ds.attrs.get('oceanembed_normalization', 'MISSING')}"
            )

            print(
                f"Channel mapping: "
                f"{ds.attrs.get('oceanembed_channel_mapping', 'MISSING')}"
            )

    except Exception as exc:
        print()
        print(f"FATAL ERROR: {exc}")
        raise

    print()
    print("=" * 100)
    print("RESULT: PASS")
    print("=" * 100)


if __name__ == "__main__":
    main()