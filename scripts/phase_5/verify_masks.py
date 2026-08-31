from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]

MASK_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "ocean_mask.nc"
)

CURL_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "wind_stress_curl_filled.nc"
)

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "input_tensor.nc"
)


def main() -> None:

    print("=" * 100)
    print("OceanEmbed — Phase 5 Mask Verification")
    print("=" * 100)

    try:
        with (
            xr.open_dataset(MASK_FILE) as mask_ds,
            xr.open_dataset(CURL_FILE) as curl_ds,
            xr.open_dataset(INPUT_FILE) as input_ds,
        ):

            # --------------------------------------------------------------
            # Ocean mask
            # --------------------------------------------------------------

            ocean_mask = mask_ds["ocean_mask"].load()

            if ocean_mask.dims != (
                "latitude",
                "longitude",
            ):
                raise ValueError(
                    f"Unexpected ocean mask dimensions: "
                    f"{ocean_mask.dims}"
                )

            ocean_count = int(ocean_mask.sum().item())
            land_count = int(
                (~ocean_mask).sum().item()
            )

            print()
            print("=" * 100)
            print("OCEAN MASK")
            print("=" * 100)

            print(f"Ocean cells : {ocean_count:,}")
            print(f"Non-ocean   : {land_count:,}")

            # --------------------------------------------------------------
            # Curl validity mask
            # --------------------------------------------------------------

            if "curl_valid_mask" not in curl_ds:
                raise ValueError(
                    "curl_valid_mask missing from curl file."
                )

            curl_valid = (
                curl_ds["curl_valid_mask"].load()
            )

            curl_valid_count = int(
                curl_valid.sum().item()
            )

            curl_invalid_count = int(
                (~curl_valid).sum().item()
            )

            print()
            print("=" * 100)
            print("CURL VALIDITY MASK")
            print("=" * 100)

            print(
                f"Curl-valid cells     : "
                f"{curl_valid_count:,}"
            )

            print(
                f"Curl-invalid cells   : "
                f"{curl_invalid_count:,}"
            )

            # --------------------------------------------------------------
            # Relationship
            # --------------------------------------------------------------

            permanently_unavailable_ocean = (
                ocean_mask & ~curl_valid
            )

            count = int(
                permanently_unavailable_ocean.sum().item()
            )

            print()
            print(
                "Ocean cells without "
                "permanently valid curl:"
            )
            print(f"  {count:,}")

            # This is expected from our previous diagnosis.
            print()
            print(
                "NOTE: These cells are retained in the "
                "common ocean mask but are unavailable "
                "for the derived curl feature."
            )

            # --------------------------------------------------------------
            # Filled curl
            # --------------------------------------------------------------

            curl = curl_ds["wind_stress_curl"].load()

            nan_count = int(
                curl.isnull().sum().item()
            )

            inf_count = int(
                np.isinf(curl.values).sum()
            )

            print()
            print("=" * 100)
            print("FILLED CURL")
            print("=" * 100)

            print(f"NaNs : {nan_count:,}")
            print(f"Infs : {inf_count:,}")

            if nan_count != 0:
                raise ValueError(
                    "Filled curl still contains NaNs."
                )

            if inf_count != 0:
                raise ValueError(
                    "Filled curl contains Infs."
                )

            # --------------------------------------------------------------
            # Input tensor consistency
            # --------------------------------------------------------------

            tensor = input_ds["input_tensor"]

            curl_channel = tensor.isel(
                channel=7
            ).load()

            if not np.allclose(
                curl.values,
                curl_channel.values,
                rtol=1e-6,
                atol=1e-10,
            ):
                raise ValueError(
                    "Curl in input tensor does not match "
                    "wind_stress_curl_filled.nc."
                )

            print()
            print(
                "Curl source -> tensor consistency: PASS"
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