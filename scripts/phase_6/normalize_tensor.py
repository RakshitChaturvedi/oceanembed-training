from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from p6_config import (
    INPUT_TENSOR_PATH,
    NORMALIZATION_STATS_PATH,
    OCEAN_MASK_PATH,
    OCEAN_MASK_VALUE,
)


OUTPUT_PATH = Path(
    "data/processed/phase6/input_tensor_normalized.nc"
)

MISSINGNESS_MASK_PATH = (
    INPUT_TENSOR_PATH.parent.parent / "phase5" / "missingness_masks.nc"
)

CURL_PATH = (
    INPUT_TENSOR_PATH.parent.parent / "phase5"
    / "wind_stress_curl_filled.nc"
)

# Tensor channel -> missingness variable
MISSINGNESS_VARIABLES = {
    0: "sst",
    1: "sss",
    2: "ssh",
    5: "current_u",
    6: "current_v",
}


def fail(message: str) -> None:
    raise RuntimeError(
        f"NORMALIZATION FAILED: {message}"
    )


def main() -> None:

    print("=" * 80)
    print("OceanEmbed — Phase 6 Part 3 — Apply Normalization")
    print("=" * 80)

    print("\nINPUT")
    print(f"  tensor       : {INPUT_TENSOR_PATH}")
    print(f"  statistics   : {NORMALIZATION_STATS_PATH}")
    print(f"  ocean mask   : {OCEAN_MASK_PATH}")
    print(f"  missingness  : {MISSINGNESS_MASK_PATH}")
    print(f"  curl mask    : {CURL_PATH}")
    print(f"  output       : {OUTPUT_PATH}")

    # ------------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------------

    for path in (
        INPUT_TENSOR_PATH,
        NORMALIZATION_STATS_PATH,
        OCEAN_MASK_PATH,
        MISSINGNESS_MASK_PATH,
        CURL_PATH,
    ):
        if not path.exists():
            fail(f"Required file not found: {path}")

    # ------------------------------------------------------------------
    # Load statistics
    # ------------------------------------------------------------------

    print("\nLOADING NORMALIZATION STATISTICS")

    with open(
        NORMALIZATION_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    if stats.get("normalization_method") != "z-score":
        fail(
            "Expected normalization method 'z-score', "
            f"got {stats.get('normalization_method')}"
        )

    statistics = stats.get("statistics")

    if not statistics:
        fail("No channel statistics found")

    print("  method : z-score")
    print("  formula: (x - mean) / std")

    # ------------------------------------------------------------------
    # Open datasets
    # ------------------------------------------------------------------

    with (
        xr.open_dataset(INPUT_TENSOR_PATH) as tensor_ds,
        xr.open_dataset(OCEAN_MASK_PATH) as mask_ds,
        xr.open_dataset(MISSINGNESS_MASK_PATH) as missing_ds,
        xr.open_dataset(CURL_PATH) as curl_ds,
    ):

        if "input_tensor" not in tensor_ds:
            fail("Variable 'input_tensor' not found")

        if "ocean_mask" not in mask_ds:
            fail("Variable 'ocean_mask' not found")

        if "curl_valid_mask" not in curl_ds:
            fail("Variable 'curl_valid_mask' not found")

        tensor = tensor_ds["input_tensor"]
        ocean_mask = mask_ds["ocean_mask"]
        curl_valid_mask = curl_ds["curl_valid_mask"]

        # --------------------------------------------------------------
        # Validate tensor
        # --------------------------------------------------------------

        expected_dims = (
            "time",
            "latitude",
            "longitude",
            "channel",
        )

        if tensor.dims != expected_dims:
            fail(
                f"Unexpected tensor dimensions: {tensor.dims}"
            )

        # --------------------------------------------------------------
        # Validate coordinates
        # --------------------------------------------------------------

        if not np.array_equal(
            tensor.latitude.values,
            ocean_mask.latitude.values,
        ):
            fail("Tensor/ocean-mask latitude coordinates differ")

        if not np.array_equal(
            tensor.longitude.values,
            ocean_mask.longitude.values,
        ):
            fail("Tensor/ocean-mask longitude coordinates differ")

        # --------------------------------------------------------------
        # Common ocean mask
        # --------------------------------------------------------------

        ocean = (
            ocean_mask == OCEAN_MASK_VALUE
        )

        ocean_count = int(
            ocean.sum().item()
        )

        if ocean_count != 12490:
            fail(
                f"Expected 12,490 ocean cells, "
                f"got {ocean_count}"
            )

        print("\nOCEAN MASK")
        print(f"  ocean cells : {ocean_count}")

        # --------------------------------------------------------------
        # Create output copy
        # --------------------------------------------------------------

        normalized = tensor.copy(
            deep=True
        )

        # ==============================================================
        # CHANNELS 0-9
        # ==============================================================

        print("\nNORMALIZING CHANNELS")

        for channel_index in range(10):

            channel_name = stats["input"]["channels"][
                channel_index
            ]

            if channel_name not in statistics:
                fail(
                    f"Statistics missing for {channel_name}"
                )

            channel_stats = statistics[channel_name]

            mean = float(
                channel_stats["mean"]
            )

            std = float(
                channel_stats["std"]
            )

            if not np.isfinite(mean):
                fail(
                    f"{channel_name}: non-finite mean"
                )

            if not np.isfinite(std) or std <= 0:
                fail(
                    f"{channel_name}: invalid std={std}"
                )

            # ----------------------------------------------------------
            # Build spatial validity mask
            # ----------------------------------------------------------

            if channel_index in MISSINGNESS_VARIABLES:

                variable_name = MISSINGNESS_VARIABLES[
                    channel_index
                ]

                if variable_name not in missing_ds:
                    fail(
                        f"Missingness variable "
                        f"'{variable_name}' not found"
                    )

                missing = missing_ds[
                    variable_name
                ].astype(bool)

                # Permanently invalid = missing for
                # every timestep.
                permanent_invalid = (
                    missing.all(dim="time")
                )

                if channel_index in (1,2):
                    tensor_always_zero=(tensor.isel(channel=channel_index)==0).all(dim="time")
                    permanent_invalid = (permanent_invalid | tensor_always_zero)

                valid_mask = (
                    ocean
                    & ~permanent_invalid
                )

            elif channel_index == 7:

                # Curl has its own permanent-valid mask.
                valid_mask = (
                    ocean
                    & curl_valid_mask
                )

            else:

                # Wind and geographic channels are valid
                # throughout the common ocean domain.
                valid_mask = ocean

            valid_count = int(
                valid_mask.sum().item()
            )

            print(
                f"  [{channel_index:2d}] "
                f"{channel_name:<20} "
                f"mean={mean:.12g} "
                f"std={std:.12g} "
                f"valid_cells={valid_count}"
            )

            # ----------------------------------------------------------
            # Extract channel
            # ----------------------------------------------------------

            values = tensor.isel(
                channel=channel_index
            )

            # ----------------------------------------------------------
            # Normalize ONLY valid cells.
            #
            # Everything outside valid_mask is explicitly forced
            # to zero.
            # ----------------------------------------------------------

            transformed = (
                values - mean
            ) / std
            valid_mask_3d = valid_mask.broadcast_like(values)

            transformed = xr.where(
                valid_mask_3d,
                transformed,
                0.0,
            )

            # ----------------------------------------------------------
            # Replace channel
            # ----------------------------------------------------------

            normalized.loc[
                {
                    "channel": channel_index
                }
            ] = transformed

        # ==============================================================
        # CHANNELS 10-11
        # ==============================================================

        print("\nUNCHANGED CHANNELS")

        for channel_index in (10, 11):

            channel_name = stats["input"]["channels"][
                channel_index
            ]

            print(
                f"  [{channel_index:2d}] "
                f"{channel_name:<20} "
                "normalization = NONE"
            )

        # --------------------------------------------------------------
        # Explicitly preserve metadata
        # --------------------------------------------------------------

        normalized.attrs.update(
            tensor.attrs
        )

        normalized.attrs[
            "oceanembed_phase"
        ] = "phase6"

        normalized.attrs[
            "oceanembed_normalization"
        ] = "z-score"

        normalized.attrs[
            "oceanembed_normalization_stats"
        ] = str(
            NORMALIZATION_STATS_PATH
        )

        normalized.attrs[
            "oceanembed_land_value"
        ] = 0.0

        normalized.attrs[
            "oceanembed_normalization_contract"
        ] = (
            "Channels 0-9 normalized only on valid "
            "ocean cells; invalid/land cells remain 0.0; "
            "channels 10-11 unchanged."
        )

        # --------------------------------------------------------------
        # Basic output validation
        # --------------------------------------------------------------

        print("\nVALIDATING OUTPUT")

        if normalized.dims != tensor.dims:
            fail(
                "Output dimensions changed"
            )

        if normalized.shape != tensor.shape:
            fail(
                "Output shape changed"
            )

        if not np.array_equal(
            normalized.latitude.values,
            tensor.latitude.values,
        ):
            fail(
                "Output latitude coordinates changed"
            )

        if not np.array_equal(
            normalized.longitude.values,
            tensor.longitude.values,
        ):
            fail(
                "Output longitude coordinates changed"
            )

        if not np.array_equal(
            normalized.time.values,
            tensor.time.values,
        ):
            fail(
                "Output time coordinates changed"
            )

        # --------------------------------------------------------------
        # Verify unchanged channels
        # --------------------------------------------------------------

        for channel_index in (10, 11):

            original = tensor.isel(
                channel=channel_index
            )

            result = normalized.isel(
                channel=channel_index
            )

            if not np.array_equal(
                original.values,
                result.values,
            ):
                fail(
                    f"Unchanged channel {channel_index} "
                    "was modified"
                )

        print(
            "  shape/dimensions : PASS"
        )

        print(
            "  coordinates      : PASS"
        )

        print(
            "  channels 10-11   : PASS"
        )

        # --------------------------------------------------------------
        # Save
        # --------------------------------------------------------------

        print("\nSAVING")

        OUTPUT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalized_ds = normalized.to_dataset(
            name="input_tensor"
        )

        normalized_ds.to_netcdf(
            OUTPUT_PATH
        )

        print(
            f"  output : {OUTPUT_PATH}"
        )

    print("\n" + "=" * 80)
    print("RESULT: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()