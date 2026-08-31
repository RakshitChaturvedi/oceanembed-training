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

CHANNELS = [
    (0, "SST"),
    (1, "SSS"),
    (2, "SSHA"),
    (3, "wind_U"),
    (4, "wind_V"),
    (5, "current_U"),
    (6, "current_V"),
    (7, "wind_stress_curl"),
    (8, "latitude"),
    (9, "longitude"),
    (10, "sin_day_of_year"),
    (11, "cos_day_of_year"),
]


def main() -> None:

    print("=" * 100)
    print("OceanEmbed — Phase 5 Channel Verification")
    print("=" * 100)

    try:
        with xr.open_dataset(INPUT) as ds:

            tensor = ds["input_tensor"]

            print()
            print("=" * 100)
            print("CHANNEL STATISTICS")
            print("=" * 100)

            for index, name in CHANNELS:

                data = tensor.isel(channel=index).values

                nan_count = int(np.isnan(data).sum())
                inf_count = int(np.isinf(data).sum())

                finite = data[np.isfinite(data)]

                if finite.size == 0:
                    raise ValueError(
                        f"{name}: contains no finite values."
                    )

                print()
                print(f"[{index:02d}] {name}")
                print(f"  Shape : {data.shape}")
                print(f"  Min   : {finite.min():.10e}")
                print(f"  Max   : {finite.max():.10e}")
                print(f"  Mean  : {finite.mean():.10e}")
                print(f"  Std   : {finite.std():.10e}")
                print(f"  NaNs  : {nan_count:,}")
                print(f"  Infs  : {inf_count:,}")

                if nan_count != 0 or inf_count != 0:
                    raise ValueError(
                        f"{name}: invalid numerical values."
                    )

            # --------------------------------------------------------------
            # Cyclic temporal features
            # --------------------------------------------------------------

            print()
            print("=" * 100)
            print("TEMPORAL FEATURE VALIDATION")
            print("=" * 100)

            sin_day = tensor.isel(channel=10).values
            cos_day = tensor.isel(channel=11).values

            magnitude = np.sqrt(
                sin_day ** 2 + cos_day ** 2
            )

            error = np.abs(magnitude - 1.0)

            print(
                f"Max |sqrt(sin²+cos²)-1| : "
                f"{error.max():.10e}"
            )

            if error.max() > 1e-5:
                raise ValueError(
                    "Temporal cyclic features are inconsistent."
                )

            # --------------------------------------------------------------
            # Spatial feature validation
            # --------------------------------------------------------------

            print()
            print("=" * 100)
            print("SPATIAL FEATURE VALIDATION")
            print("=" * 100)

            latitude = tensor.isel(channel=8).values
            longitude = tensor.isel(channel=9).values

            lat_reference = latitude[0]
            lon_reference = longitude[0]

            if not np.allclose(
                latitude,
                lat_reference[None, :, :],
            ):
                raise ValueError(
                    "Latitude channel changes across time."
                )

            if not np.allclose(
                longitude,
                lon_reference[None, :, :],
            ):
                raise ValueError(
                    "Longitude channel changes across time."
                )

            print("Latitude  : spatially consistent")
            print("Longitude : spatially consistent")

            # --------------------------------------------------------------
            # Channel ordering
            # --------------------------------------------------------------

            print()
            print("=" * 100)
            print("CHANNEL ORDER")
            print("=" * 100)

            for index, name in CHANNELS:
                print(f"  [{index:02d}] -> {name}")

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