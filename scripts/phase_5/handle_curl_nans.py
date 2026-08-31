from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

CURL_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "phase4"
    / "wind_stress_curl.nc"
)

OCEAN_MASK_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "ocean_mask.nc"
)

OUTPUT = (
    ROOT
    / "data"
    / "processed"
    / "phase5"
    / "wind_stress_curl_filled.nc"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def validate_dimensions(da: xr.DataArray, name: str) -> None:
    expected = ("time", "latitude", "longitude")

    if da.dims != expected:
        raise ValueError(
            f"{name}: unexpected dimensions.\n"
            f"  expected: {expected}\n"
            f"  found:    {da.dims}"
        )


def validate_grid(
    curl: xr.DataArray,
    mask: xr.DataArray,
) -> None:
    for coord in ("latitude", "longitude"):
        if not curl[coord].equals(mask[coord]):
            raise ValueError(
                f"Grid mismatch in coordinate: {coord}"
            )


def main() -> None:

    print("=" * 100)
    print("OceanEmbed — Phase 5 Wind Stress Curl NaN Handling")
    print("=" * 100)

    print()
    print("READ ONLY INPUTS")
    print(f"  Curl : {CURL_INPUT}")
    print(f"  Mask : {OCEAN_MASK_INPUT}")

    # -----------------------------------------------------------------------
    # Load
    # -----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("LOADING INPUTS")
    print("=" * 100)

    try:
        with (
            xr.open_dataset(CURL_INPUT) as curl_ds,
            xr.open_dataset(OCEAN_MASK_INPUT) as mask_ds,
        ):
            if "wind_stress_curl" not in curl_ds:
                raise ValueError(
                    "Input curl dataset does not contain "
                    "'wind_stress_curl'."
                )

            if "ocean_mask" not in mask_ds:
                raise ValueError(
                    "Ocean mask dataset does not contain "
                    "'ocean_mask'."
                )

            curl = curl_ds["wind_stress_curl"].load()
            ocean_mask = mask_ds["ocean_mask"].load()

    except Exception as exc:
        print()
        print(f"FATAL ERROR: {exc}")
        raise

    validate_dimensions(curl, "wind_stress_curl")

    if ocean_mask.dims != ("latitude", "longitude"):
        raise ValueError(
            "ocean_mask: unexpected dimensions.\n"
            f"  expected: ('latitude', 'longitude')\n"
            f"  found:    {ocean_mask.dims}"
        )

    validate_grid(curl, ocean_mask)

    # -----------------------------------------------------------------------
    # Initial diagnostics
    # -----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("INITIAL CURL DIAGNOSTICS")
    print("=" * 100)

    total_nans = int(curl.isnull().sum().item())
    total_infs = int(
        np.isinf(curl.values).sum()
    )

    ocean_3d = ocean_mask.broadcast_like(curl)

    nans_on_ocean = int(
        (curl.isnull() & ocean_3d).sum().item()
    )

    nans_off_ocean = int(
        (curl.isnull() & ~ocean_3d).sum().item()
    )

    print(f"Total NaNs             : {total_nans:,}")
    print(f"NaNs on ocean mask     : {nans_on_ocean:,}")
    print(f"NaNs outside ocean     : {nans_off_ocean:,}")
    print(f"Total Infs             : {total_infs:,}")

    # -----------------------------------------------------------------------
    # Safety check
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Curl validity mask
    # -----------------------------------------------------------------------
    #
    # A cell is considered curl-valid if the Phase 4 curl has at least
    # one valid observation over time.
    #
    # This is deliberately separate from the common ocean mask because
    # spatial differentiation can make some ocean cells unavailable.
    # -----------------------------------------------------------------------

    curl_valid_mask = curl.notnull().any(dim="time")

    curl_valid_count = int(curl_valid_mask.sum().item())
    curl_invalid_count = int((~curl_valid_mask).sum().item())

    print()
    print("Curl validity mask:")
    print(f"  Valid cells       : {curl_valid_count:,}")
    print(f"  Permanently invalid: {curl_invalid_count:,}")


    # -----------------------------------------------------------------------
    # Check relationship with common ocean mask
    # -----------------------------------------------------------------------

    invalid_inside_ocean = (
        (~curl_valid_mask) & ocean_mask
    )

    invalid_inside_ocean_count = int(
        invalid_inside_ocean.sum().item()
    )

    print()
    print("Curl / ocean-mask relationship:")
    print(
        f"  Permanently invalid curl cells inside ocean mask : "
        f"{invalid_inside_ocean_count:,}"
    )


    # -----------------------------------------------------------------------
    # Fill only cells where curl is permanently unavailable
    # -----------------------------------------------------------------------

    curl_filled = curl.fillna(0.0)

    curl_filled.name = "wind_stress_curl"

    # -----------------------------------------------------------------------
    # Fill permanently unavailable cells
    # -----------------------------------------------------------------------

    print()
    print("=" * 100)
    print("ENCODING PERMANENTLY UNAVAILABLE CELLS")
    print("=" * 100)

    missing = curl.isnull()

    filled_count = int(
        (missing & ~ocean_3d).sum().item()
    )

    curl_filled = curl.where(
        ~missing,
        0.0,
    )

    print(
        f"Unavailable NaNs converted to 0 : {filled_count:,}"
    )

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    final_nans = int(
        curl_filled.isnull().sum().item()
    )

    final_infs = int(
        np.isinf(curl_filled.values).sum()
    )

    print()
    print("=" * 100)
    print("FINAL VALIDATION")
    print("=" * 100)

    print(f"Final NaNs : {final_nans:,}")
    print(f"Final Infs : {final_infs:,}")

    if final_nans != 0:
        raise RuntimeError(
            f"Final curl still contains {final_nans:,} NaNs."
        )

    if final_infs != 0:
        raise RuntimeError(
            f"Final curl contains {final_infs:,} Infs."
        )

    # -----------------------------------------------------------------------
    # Metadata
    # -----------------------------------------------------------------------

    curl_filled.name = "wind_stress_curl"

    curl_filled.attrs.update(
        {
            "oceanembed_phase": "phase5",
            "oceanembed_nan_policy": (
                "permanently unavailable cells encoded as 0; "
                "valid ocean curl values preserved"
            ),
            "oceanembed_source": str(CURL_INPUT),
            "oceanembed_ocean_mask": str(OCEAN_MASK_INPUT),
        }
    )

    curl_valid_mask.name = "curl_valid_mask"

    curl_valid_mask.attrs.update(
        {
            "long_name": "wind stress curl validity mask",
            "description": (
                "True where wind stress curl has at least one "
                "valid observation over time in the Phase 4 product"
            ),
            "oceanembed_phase": "phase5",
        }
    )

    output_ds = xr.Dataset(
        {
            "wind_stress_curl": curl_filled,
            "curl_valid_mask": curl_valid_mask,
        }
    )

    output_ds.attrs.update(
        {
            "oceanembed_phase": "phase5",
            "oceanembed_feature": "wind_stress_curl",
            "oceanembed_processing": "NaN handling only",
        }
    )

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print("WRITING")
    print("=" * 100)

    print(f"Output: {OUTPUT}")

    output_ds.to_netcdf(
        OUTPUT,
        mode="w",
    )

    output_ds.close()

    print()
    print("=" * 100)
    print("COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()