from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from p6_config import (
    CHANNELS,
    EXPECTED_SHAPE,
    INPUT_TENSOR_PATH,
    NORMALIZATION_STATS_PATH,
    OCEAN_MASK_PATH,
    OCEAN_MASK_VALUE,
    TRAINING_SIZE,
    TRAINING_SLICE,
)


NORMALIZED_PATH = Path(
    "data/processed/phase6/input_tensor_normalized.nc"
)


MISSINGNESS_MASK_PATH = (
    INPUT_TENSOR_PATH.parent.parent
    / "phase5"
    / "missingness_masks.nc"
)

CURL_PATH = (
    INPUT_TENSOR_PATH.parent.parent
    / "phase5"
    / "wind_stress_curl_filled.nc"
)


MISSINGNESS_VARIABLES = {
    0: "sst",
    1: "sss",
    2: "ssh",
    5: "current_u",
    6: "current_v",
}


# Numerical tolerance for checking mean/std ~= 0/1.
MEAN_TOLERANCE = 1e-5
STD_TOLERANCE = 1e-5


def fail(message: str) -> None:
    raise RuntimeError(
        f"NORMALIZATION VERIFICATION FAILED: {message}"
    )


def isoformat_time(value) -> str:
    return str(
        np.datetime_as_string(
            value,
            unit="s",
        )
    )


def main() -> None:

    print("=" * 80)
    print("OceanEmbed — Phase 6 Part 4 — Verify Normalization")
    print("=" * 80)

    print("\nINPUT")

    print(
        f"  original tensor : "
        f"{INPUT_TENSOR_PATH}"
    )

    print(
        f"  normalized      : "
        f"{NORMALIZED_PATH}"
    )

    print(
        f"  statistics      : "
        f"{NORMALIZATION_STATS_PATH}"
    )

    print(
        f"  ocean mask      : "
        f"{OCEAN_MASK_PATH}"
    )

    print(
        f"  missingness     : "
        f"{MISSINGNESS_MASK_PATH}"
    )

    print(
        f"  curl mask       : "
        f"{CURL_PATH}"
    )

    # ------------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------------

    print("\nFILE VALIDATION")

    for path in (
        INPUT_TENSOR_PATH,
        NORMALIZED_PATH,
        NORMALIZATION_STATS_PATH,
        OCEAN_MASK_PATH,
        MISSINGNESS_MASK_PATH,
        CURL_PATH,
    ):
        if not path.exists():
            fail(
                f"Required file not found: {path}"
            )

    print("  required files : PASS")

    # ------------------------------------------------------------------
    # Load statistics
    # ------------------------------------------------------------------

    with open(
        NORMALIZATION_STATS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    if stats.get("normalization_method") != "z-score":
        fail(
            "Statistics file does not declare z-score normalization"
        )

    statistics = stats.get("statistics")

    if not statistics:
        fail(
            "No channel statistics found"
        )

    # ------------------------------------------------------------------
    # Open datasets
    # ------------------------------------------------------------------

    with (
        xr.open_dataset(INPUT_TENSOR_PATH) as original_ds,
        xr.open_dataset(NORMALIZED_PATH) as normalized_ds,
        xr.open_dataset(OCEAN_MASK_PATH) as mask_ds,
        xr.open_dataset(MISSINGNESS_MASK_PATH) as missing_ds,
        xr.open_dataset(CURL_PATH) as curl_ds,
    ):

        if "input_tensor" not in original_ds:
            fail(
                "Original tensor variable not found"
            )

        if "input_tensor" not in normalized_ds:
            fail(
                "Normalized tensor variable not found"
            )

        if "ocean_mask" not in mask_ds:
            fail(
                "Ocean mask variable not found"
            )

        if "curl_valid_mask" not in curl_ds:
            fail(
                "Curl validity mask not found"
            )

        original = original_ds["input_tensor"]
        normalized = normalized_ds["input_tensor"]

        ocean_mask = mask_ds["ocean_mask"]
        curl_valid_mask = curl_ds["curl_valid_mask"]

        # ==============================================================
        # 1. SHAPE / DIMENSION VALIDATION
        # ==============================================================

        print("\n1. STRUCTURAL VALIDATION")

        if tuple(normalized.shape) != EXPECTED_SHAPE:
            fail(
                f"Expected normalized tensor shape "
                f"{EXPECTED_SHAPE}, got "
                f"{tuple(normalized.shape)}"
            )

        if normalized.dims != (
            "time",
            "latitude",
            "longitude",
            "channel",
        ):
            fail(
                f"Unexpected normalized dimensions: "
                f"{normalized.dims}"
            )

        if normalized.shape != original.shape:
            fail(
                "Normalized tensor shape differs "
                "from original tensor"
            )

        if not np.array_equal(
            normalized.time.values,
            original.time.values,
        ):
            fail(
                "Time coordinates changed"
            )

        if not np.array_equal(
            normalized.latitude.values,
            original.latitude.values,
        ):
            fail(
                "Latitude coordinates changed"
            )

        if not np.array_equal(
            normalized.longitude.values,
            original.longitude.values,
        ):
            fail(
                "Longitude coordinates changed"
            )

        print(
            f"  shape       : {tuple(normalized.shape)}"
        )

        print(
            f"  dimensions  : {normalized.dims}"
        )

        print(
            "  coordinates : PASS"
        )

        print(
            "  structure   : PASS"
        )

        # ==============================================================
        # 2. OCEAN MASK
        # ==============================================================

        print("\n2. OCEAN MASK")

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

        print(
            f"  ocean cells : {ocean_count}"
        )

        print(
            "  validation  : PASS"
        )

        # ==============================================================
        # 3. TRAINING WINDOW
        # ==============================================================

        print("\n3. TRAINING WINDOW")

        original_train = original.isel(
            time=TRAINING_SLICE
        )

        normalized_train = normalized.isel(
            time=TRAINING_SLICE
        )

        if normalized_train.sizes["time"] != TRAINING_SIZE:
            fail(
                f"Expected {TRAINING_SIZE} training "
                f"timesteps, got "
                f"{normalized_train.sizes['time']}"
            )

        print(
            f"  timesteps : {TRAINING_SIZE}"
        )

        print(
            f"  dates     : "
            f"{isoformat_time(original.time.isel(time=0).values)} "
            f"→ "
            f"{isoformat_time(original.time.isel(time=TRAINING_SIZE - 1).values)}"
        )

        print(
            "  validation: PASS"
        )

        # ==============================================================
        # 4. REBUILD VALIDITY MASKS
        # ==============================================================

        print("\n4. VALIDITY MASKS")

        permanent_valid_masks = {}

        for channel_index, variable_name in (
            MISSINGNESS_VARIABLES.items()
        ):

            if variable_name not in missing_ds:
                fail(
                    f"Missingness variable "
                    f"'{variable_name}' not found"
                )

            missing = missing_ds[
                variable_name
            ].astype(bool)

            original_channel = original.isel(channel=channel_index)
            tensor_always_zero = (original_channel==0).all(dim="time")

            permanent_invalid = (
                missing.all(dim="time") | tensor_always_zero
            )

            permanent_valid = (
                ~permanent_invalid
            )

            valid_mask = (
                ocean
                & permanent_valid
            )

            permanent_valid_masks[
                channel_index
            ] = valid_mask

        # Wind channels
        permanent_valid_masks[3] = ocean
        permanent_valid_masks[4] = ocean

        # Curl
        permanent_valid_masks[7] = (
            ocean
            & curl_valid_mask
        )

        # Geographic channels
        permanent_valid_masks[8] = ocean
        permanent_valid_masks[9] = ocean

        for channel_index in range(10):

            count = int(
                permanent_valid_masks[
                    channel_index
                ].sum().item()
            )

            print(
                f"  [{channel_index:2d}] "
                f"{CHANNELS[channel_index]:<20} "
                f"valid cells = {count}"
            )

        print(
            "  validation : PASS"
        )

        # ==============================================================
        # 5. FINITE VALUE VALIDATION
        # ==============================================================

        print("\n5. FINITE VALUE VALIDATION")

        for channel_index in range(12):

            values = normalized.isel(
                channel=channel_index
            ).values

            nonfinite = np.count_nonzero(
                ~np.isfinite(values)
            )

            if nonfinite:
                fail(
                    f"{CHANNELS[channel_index]} contains "
                    f"{nonfinite:,} non-finite values"
                )

        print(
            "  all channels finite : PASS"
        )

        # ==============================================================
        # 6. NORMALIZED CHANNEL STATISTICS
        # ==============================================================

        print("\n6. NORMALIZED TRAINING STATISTICS")

        for channel_index in range(10):

            channel_name = CHANNELS[
                channel_index
            ]

            if channel_name not in statistics:
                fail(
                    f"Statistics missing for "
                    f"{channel_name}"
                )

            expected_mean = float(
                statistics[channel_name]["mean"]
            )

            expected_std = float(
                statistics[channel_name]["std"]
            )

            valid_mask = (
                permanent_valid_masks[
                    channel_index
                ]
            )

            values = normalized_train.isel(
                channel=channel_index
            )

            masked = values.where(
                valid_mask
            )

            values_np = masked.values

            valid_values = values_np[
                np.isfinite(values_np)
            ].astype(
                np.float64,
                copy=False,
            )

            if valid_values.size == 0:
                fail(
                    f"{channel_name}: no valid "
                    "training values"
                )

            mean = float(
                np.mean(valid_values)
            )

            std = float(
                np.std(
                    valid_values,
                    ddof=0,
                )
            )

            mean_error = abs(mean)
            std_error = abs(std - 1.0)

            print(
                f"\n  [{channel_index:2d}] "
                f"{channel_name}"
            )

            print(
                f"      normalized mean : "
                f"{mean:.12g}"
            )

            print(
                f"      normalized std  : "
                f"{std:.12g}"
            )

            print(
                f"      mean error      : "
                f"{mean_error:.12g}"
            )

            print(
                f"      std error       : "
                f"{std_error:.12g}"
            )

            if mean_error > MEAN_TOLERANCE:
                fail(
                    f"{channel_name}: normalized "
                    f"mean={mean} exceeds tolerance "
                    f"{MEAN_TOLERANCE}"
                )

            if std_error > STD_TOLERANCE:
                fail(
                    f"{channel_name}: normalized "
                    f"std={std} exceeds tolerance "
                    f"{STD_TOLERANCE}"
                )

            # Verify the normalization actually corresponds
            # to the recorded statistics.
            original_values = (
                original_train
                .isel(channel=channel_index)
                .where(valid_mask)
                .values
            )

            original_values = (
                original_values[
                    np.isfinite(original_values)
                ]
                .astype(
                    np.float64,
                    copy=False,
                )
            )

            expected_normalized = (
                original_values
                - expected_mean
            ) / expected_std

            actual_normalized = (
                valid_values
            )

            if not np.allclose(
                actual_normalized,
                expected_normalized,
                rtol=1e-6,
                atol=1e-6,
            ):
                fail(
                    f"{channel_name}: output does not "
                    "match recorded z-score statistics"
                )

            print(
                "      z-score formula : PASS"
            )

        print(
            "\n  normalized channels : PASS"
        )

        # ==============================================================
        # 7. LAND CELLS MUST REMAIN ZERO
        # ==============================================================

        print("\n7. LAND CELL PRESERVATION")

        land = ~ocean

        land_count = int(
            land.sum().item()
        )

        print(
            f"  land cells : {land_count}"
        )

        for channel_index in range(10):

            values = normalized.isel(
                channel=channel_index
            ).values

            land_values = values[:, land.values]

            if not np.all(
                land_values == 0.0
            ):
                fail(
                    f"{CHANNELS[channel_index]}: "
                    "land cells are not exactly 0.0"
                )

        print(
            "  channels 0-9 : exactly 0.0 : PASS"
        )

        # ==============================================================
        # 8. PERMANENTLY INVALID CELLS MUST REMAIN ZERO
        # ==============================================================

        print(
            "\n8. PERMANENTLY INVALID CELL PRESERVATION"
        )

        for channel_index in range(10):

            valid_mask = (
                permanent_valid_masks[
                    channel_index
                ]
            )

            invalid = ~valid_mask

            values = normalized.isel(
                channel=channel_index
            ).values

            invalid_values = values[:, invalid.values]

            if not np.all(
                invalid_values == 0.0
            ):
                fail(
                    f"{CHANNELS[channel_index]}: "
                    "permanently invalid cells "
                    "are not exactly 0.0"
                )

        print(
            "  invalid cells : exactly 0.0 : PASS"
        )

        # ==============================================================
        # 9. UNCHANGED CHANNELS
        # ==============================================================

        print("\n9. UNCHANGED CHANNELS")

        for channel_index in (10, 11):

            channel_name = CHANNELS[
                channel_index
            ]

            original_values = (
                original.isel(
                    channel=channel_index
                ).values
            )

            normalized_values = (
                normalized.isel(
                    channel=channel_index
                ).values
            )

            if not np.array_equal(
                original_values,
                normalized_values,
            ):
                fail(
                    f"{channel_name}: channel "
                    "was modified"
                )

            print(
                f"  [{channel_index:2d}] "
                f"{channel_name:<20} "
                "UNCHANGED : PASS"
            )

        # ==============================================================
        # 10. OUTPUT METADATA
        # ==============================================================

        print("\n10. OUTPUT METADATA")

        normalization_method = (
            normalized.attrs.get(
                "oceanembed_normalization"
            )
        )

        if normalization_method != "z-score":
            fail(
                "Output metadata does not declare "
                "z-score normalization"
            )

        land_value = (
            normalized.attrs.get(
                "oceanembed_land_value"
            )
        )

        if land_value != 0.0:
            fail(
                f"Expected land value metadata "
                f"0.0, got {land_value}"
            )

        print(
            "  normalization : z-score"
        )

        print(
            "  land value    : 0.0"
        )

        print(
            "  metadata      : PASS"
        )

        # ==============================================================
        # FINAL RESULT
        # ==============================================================

        print("\n" + "=" * 80)
        print("RESULT: PASS")
        print("=" * 80)

        print(
            "\nPhase 6 Part 4 verification confirms:"
        )

        print(
            "  ✓ channels 0-9 are correctly normalized"
        )

        print(
            "  ✓ training-only statistics are respected"
        )

        print(
            "  ✓ land cells remain exactly 0.0"
        )

        print(
            "  ✓ permanently invalid cells remain exactly 0.0"
        )

        print(
            "  ✓ channels 10-11 remain unchanged"
        )

        print(
            "  ✓ tensor structure and coordinates are unchanged"
        )

        print(
            "  ✓ no NaN/Inf values are present"
        )

        print(
            "  ✓ normalization formula matches recorded statistics"
        )


if __name__ == "__main__":
    main()