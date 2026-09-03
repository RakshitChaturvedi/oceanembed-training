from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from p6_config import (
    CHANNELS,
    EXPECTED_SHAPE,
    INPUT_TENSOR_PATH,
    NORMALIZATION_METHOD,
    NORMALIZATION_STATS_PATH,
    OCEAN_MASK_PATH,
    OCEAN_MASK_VALUE,
    STD_MINIMUM,
    TRAINING_SIZE,
    TRAINING_SLICE,
)


MISSINGNESS_MASK_PATH = (
    INPUT_TENSOR_PATH.parent / "missingness_masks.nc"
)

CURL_PATH = (
    INPUT_TENSOR_PATH.parent / "wind_stress_curl_filled.nc"
)


# Channel -> variable in missingness_masks.nc
MISSINGNESS_VARIABLES = {
    0: "sst",
    1: "sss",
    2: "ssh",
    5: "current_u",
    6: "current_v",
}


def fail(message: str) -> None:
    raise RuntimeError(
        f"STATISTICS COMPUTATION FAILED: {message}"
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
    print("OceanEmbed — Phase 6 Training Statistics")
    print("=" * 80)

    print("\nINPUT")
    print(f"  tensor       : {INPUT_TENSOR_PATH}")
    print(f"  ocean mask   : {OCEAN_MASK_PATH}")
    print(f"  missingness  : {MISSINGNESS_MASK_PATH}")
    print(f"  curl file    : {CURL_PATH}")

    # ------------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------------

    for path in (
        INPUT_TENSOR_PATH,
        OCEAN_MASK_PATH,
        MISSINGNESS_MASK_PATH,
        CURL_PATH,
    ):
        if not path.exists():
            fail(f"Required file not found: {path}")

    NORMALIZATION_STATS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Open datasets
    # ------------------------------------------------------------------

    with (
        xr.open_dataset(INPUT_TENSOR_PATH) as tensor_ds,
        xr.open_dataset(OCEAN_MASK_PATH) as mask_ds,
        xr.open_dataset(MISSINGNESS_MASK_PATH) as missing_ds,
        xr.open_dataset(CURL_PATH) as curl_ds,
    ):
        # --------------------------------------------------------------
        # Basic variables
        # --------------------------------------------------------------

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
        # Tensor validation
        # --------------------------------------------------------------

        print("\nTENSOR")

        if tuple(tensor.shape) != EXPECTED_SHAPE:
            fail(
                f"Expected tensor shape {EXPECTED_SHAPE}, "
                f"got {tuple(tensor.shape)}"
            )

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

        print(f"  shape      : {tuple(tensor.shape)}")
        print(f"  dimensions : {tensor.dims}")
        print("  validation : PASS")

        # --------------------------------------------------------------
        # Ocean mask validation
        # --------------------------------------------------------------

        print("\nOCEAN MASK")

        expected_mask_dims = (
            "latitude",
            "longitude",
        )

        expected_mask_shape = (
            EXPECTED_SHAPE[1],
            EXPECTED_SHAPE[2],
        )

        if ocean_mask.dims != expected_mask_dims:
            fail(
                f"Unexpected ocean mask dimensions: "
                f"{ocean_mask.dims}"
            )

        if tuple(ocean_mask.shape) != expected_mask_shape:
            fail(
                f"Unexpected ocean mask shape: "
                f"{tuple(ocean_mask.shape)}"
            )

        if not np.array_equal(
            tensor.latitude.values,
            ocean_mask.latitude.values,
        ):
            fail("Tensor/mask latitude coordinates differ")

        if not np.array_equal(
            tensor.longitude.values,
            ocean_mask.longitude.values,
        ):
            fail("Tensor/mask longitude coordinates differ")

        ocean = ocean_mask == OCEAN_MASK_VALUE

        ocean_cell_count = int(
            ocean.sum().item()
        )

        if ocean_cell_count != 12490:
            fail(
                f"Expected 12,490 common-ocean cells, "
                f"got {ocean_cell_count}"
            )

        print(
            f"  common ocean cells : "
            f"{ocean_cell_count}"
        )
        print("  validation         : PASS")

        # --------------------------------------------------------------
        # Curl mask validation
        # --------------------------------------------------------------

        print("\nCURL VALIDITY MASK")

        if curl_valid_mask.dims != expected_mask_dims:
            fail(
                "curl_valid_mask has unexpected dimensions"
            )

        if tuple(curl_valid_mask.shape) != expected_mask_shape:
            fail(
                "curl_valid_mask has unexpected shape"
            )

        if not np.array_equal(
            tensor.latitude.values,
            curl_valid_mask.latitude.values,
        ):
            fail(
                "Tensor/curl mask latitude coordinates differ"
            )

        if not np.array_equal(
            tensor.longitude.values,
            curl_valid_mask.longitude.values,
        ):
            fail(
                "Tensor/curl mask longitude coordinates differ"
            )

        curl_valid_cell_count = int(
            curl_valid_mask.sum().item()
        )

        if curl_valid_cell_count != 10530:
            fail(
                f"Expected 10,530 curl-valid cells, "
                f"got {curl_valid_cell_count}"
            )

        print(
            f"  curl-valid cells : "
            f"{curl_valid_cell_count}"
        )
        print("  validation       : PASS")

        # --------------------------------------------------------------
        # Training period
        # --------------------------------------------------------------

        print("\nTRAINING WINDOW")

        train = tensor.isel(
            time=TRAINING_SLICE
        )

        if train.sizes["time"] != TRAINING_SIZE:
            fail(
                f"Expected {TRAINING_SIZE} training timesteps, "
                f"got {train.sizes['time']}"
            )

        print(
            f"  timesteps : {train.sizes['time']}"
        )
        print(
            f"  indices   : [0, {TRAINING_SIZE})"
        )
        print(
            f"  dates     : "
            f"{isoformat_time(tensor.time.isel(time=0).values)} "
            f"→ "
            f"{isoformat_time(tensor.time.isel(time=TRAINING_SIZE - 1).values)}"
        )

        # --------------------------------------------------------------
        # Verify all missingness masks
        # --------------------------------------------------------------

        print("\nPERMANENT VALIDITY MASKS")

        permanent_valid_masks = {}

        for channel_index, variable_name in MISSINGNESS_VARIABLES.items():

            if variable_name not in missing_ds:
                fail(
                    f"Missingness variable '{variable_name}' "
                    "not found"
                )

            missing = missing_ds[variable_name]

            if missing.dims != (
                "time",
                "latitude",
                "longitude",
            ):
                fail(
                    f"{variable_name} has unexpected dimensions: "
                    f"{missing.dims}"
                )

            if tuple(missing.shape) != (
                EXPECTED_SHAPE[0],
                EXPECTED_SHAPE[1],
                EXPECTED_SHAPE[2],
            ):
                fail(
                    f"{variable_name} has unexpected shape"
                )

            # IMPORTANT:
            #
            # missing == 1 means the ORIGINAL observation was missing.
            #
            # A spatial cell is permanently unavailable only if it
            # was missing for the ENTIRE time period.
            #
            # Therefore:
            #
            #     permanent_invalid = missing.all(time)
            #     permanent_valid   = NOT permanent_invalid
            #
            tensor_always_zero = (tensor.isel(channel=channel_index)==0).all(dim="time")
            permanent_invalid = (
                missing.astype(bool)
                .all(dim="time") | tensor_always_zero
            )

            permanent_valid = ~permanent_invalid

            # We only care about the common ocean domain.
            channel_ocean_valid = (
                ocean & permanent_valid
            )

            count = int(
                channel_ocean_valid.sum().item()
            )

            permanent_valid_masks[channel_index] = (
                channel_ocean_valid
            )

            print(
                f"  [{channel_index:2d}] "
                f"{CHANNELS[channel_index]:<20} "
                f"valid spatial cells = {count}"
            )

        # --------------------------------------------------------------
        # Latitude/longitude masks
        # --------------------------------------------------------------

        permanent_valid_masks[3] = ocean
        permanent_valid_masks[4] = ocean

        permanent_valid_masks[8] = ocean
        permanent_valid_masks[9] = ocean

        print(
            f"  [ 8] latitude             "
            f"valid spatial cells = {ocean_cell_count}"
        )

        print(
            f"  [ 9] longitude            "
            f"valid spatial cells = {ocean_cell_count}"
        )

        # --------------------------------------------------------------
        # Curl mask
        # --------------------------------------------------------------

        curl_stat_mask = (
            ocean & curl_valid_mask
        )

        curl_count = int(
            curl_stat_mask.sum().item()
        )

        if curl_count != 10530:
            fail(
                f"Expected 10,530 valid curl cells "
                f"inside common ocean mask, got {curl_count}"
            )

        permanent_valid_masks[7] = curl_stat_mask

        print(
            f"  [ 7] wind_stress_curl     "
            f"valid spatial cells = {curl_count}"
        )

        # --------------------------------------------------------------
        # Compute statistics
        # --------------------------------------------------------------

        print("\nCOMPUTING STATISTICS")

        print(
            "  temporal population : first 993 days"
        )
        print(
            "  spatial population  : common ocean"
        )
        print(
            "  permanent invalid   : excluded"
        )
        print(
            "  intermittent gaps   : included after Phase 5 filling"
        )
        print(
            "  normalized channels : 0-9"
        )

        statistics = {}

        for channel_index in range(10):

            channel_name = CHANNELS[channel_index]

            print(
                f"\n  [{channel_index:2d}] "
                f"{channel_name}"
            )

            values = train.isel(
                channel=channel_index
            )

            spatial_mask = permanent_valid_masks[
                channel_index
            ]

            # Broadcast the 2-D spatial mask across training time.
            masked = values.where(
                spatial_mask
            )

            values_np = masked.values

            valid_values = values_np[np.isfinite(values_np)]

            if channel_index == 1:
                valid_values = valid_values[valid_values > 0.0]

            if valid_values.size == 0:
                fail(
                    f"{channel_name}: no finite values "
                    "after validity masking"
                )

            if channel_index == 1:
                expected_count = int(
                    np.count_nonzero(
                        np.isfinite(values_np) & (values_np>0.0)
                    )
                )
            else:
                expected_count = (
                    TRAINING_SIZE
                    * int(spatial_mask.sum().item())
                )

            if valid_values.size != expected_count:
                fail(
                    f"{channel_name}: expected "
                    f"{expected_count:,} valid values, "
                    f"found {valid_values.size:,}"
                )

            valid_values = valid_values.astype(
                np.float64,
                copy=False,
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

            minimum = float(
                np.min(valid_values)
            )

            maximum = float(
                np.max(valid_values)
            )

            if not np.isfinite(mean):
                fail(
                    f"{channel_name}: mean is not finite"
                )

            if not np.isfinite(std):
                fail(
                    f"{channel_name}: std is not finite"
                )

            if std <= STD_MINIMUM:
                fail(
                    f"{channel_name}: std={std} "
                    f"is too small"
                )

            statistics[channel_name] = {
                "index": channel_index,
                "mean": mean,
                "std": std,
                "valid_spatial_cells": int(
                    spatial_mask.sum().item()
                ),
                "valid_count": int(
                    valid_values.size
                ),
                "min": minimum,
                "max": maximum,
            }

            print(
                f"      spatial cells : "
                f"{int(spatial_mask.sum().item()):,}"
            )
            print(
                f"      valid values  : "
                f"{valid_values.size:,}"
            )
            print(
                f"      mean          : "
                f"{mean:.12g}"
            )
            print(
                f"      std           : "
                f"{std:.12g}"
            )
            print(
                f"      min           : "
                f"{minimum:.12g}"
            )
            print(
                f"      max           : "
                f"{maximum:.12g}"
            )

            del values_np
            del valid_values

        # --------------------------------------------------------------
        # Verify unchanged channels
        # --------------------------------------------------------------

        print("\nUNCHANGED CHANNELS")
        print("  [10] sin_day_of_year")
        print("  [11] cos_day_of_year")
        print("  normalization : NONE")

        # --------------------------------------------------------------
        # Build artifact
        # --------------------------------------------------------------

        output = {
            "phase": 6,
            "artifact": "normalization_stats",
            "normalization_method": NORMALIZATION_METHOD,
            "formula": "(x - mean) / std",

            "input": {
                "path": str(INPUT_TENSOR_PATH),
                "shape": list(EXPECTED_SHAPE),
                "channels": CHANNELS,
            },

            "ocean_mask": {
                "path": str(OCEAN_MASK_PATH),
                "mask_value": OCEAN_MASK_VALUE,
                "ocean_cell_count": ocean_cell_count,
            },

            "missingness_masks": {
                "path": str(MISSINGNESS_MASK_PATH),
                "encoding": {
                    "0": "original observation present",
                    "1": "original observation missing",
                },
                "permanent_invalid_definition": (
                    "missingness_mask == 1 for every timestep"
                ),
                "intermittent_filled_values": (
                    "included in normalization population"
                ),
            },

            "curl_validity": {
                "path": str(CURL_PATH),
                "variable": "curl_valid_mask",
                "true": (
                    "at least one valid curl observation "
                    "over the full time period"
                ),
                "false": (
                    "curl permanently unavailable"
                ),
                "valid_spatial_cells": curl_count,
            },

            "training_window": {
                "start_index": 0,
                "end_index_exclusive": TRAINING_SIZE,
                "sample_count": TRAINING_SIZE,
                "start_time": isoformat_time(
                    tensor.time.isel(time=0).values
                ),
                "end_time": isoformat_time(
                    tensor.time.isel(
                        time=TRAINING_SIZE - 1
                    ).values
                ),
            },

            "normalized_channel_indices": list(range(10)),

            "unchanged_channel_indices": [10, 11],

            "unchanged_channels": {
                "10": "sin_day_of_year",
                "11": "cos_day_of_year",
            },

            "statistics": statistics,

            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        # --------------------------------------------------------------
        # Write artifact
        # --------------------------------------------------------------

        with open(
            NORMALIZATION_STATS_PATH,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                output,
                f,
                indent=2,
                allow_nan=False,
            )

    print("\nOUTPUT")
    print(
        f"  {NORMALIZATION_STATS_PATH}"
    )

    print("\n" + "=" * 80)
    print("RESULT: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()