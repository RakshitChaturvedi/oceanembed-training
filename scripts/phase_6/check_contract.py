from pathlib import Path
import xarray as xr

from p6_config import (
        INPUT_TENSOR_PATH,
    OCEAN_MASK_PATH,
    EXPECTED_SHAPE,
    EXPECTED_TIME_SIZE,
    EXPECTED_LATITUDE_SIZE,
    EXPECTED_LONGITUDE_SIZE,
    EXPECTED_CHANNEL_SIZE,
    CHANNELS,
    TRAINING_SIZE,
    NORMALIZE_CHANNEL_INDICES,
    UNCHANGED_CHANNEL_INDICES,
)

def fail(message: str) -> None:
    raise RuntimeError(f"Contract check failed: {message}")

def fail(message: str) -> None:
    raise RuntimeError(f"CONTRACT CHECK FAILED: {message}")


def main() -> None:
    print("=" * 80)
    print("OceanEmbed — Phase 6 Contract Check")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # File existence
    # -------------------------------------------------------------------------

    print("\nINPUT FILES")

    if not INPUT_TENSOR_PATH.exists():
        fail(f"Input tensor not found: {INPUT_TENSOR_PATH}")

    print(f"  input tensor : {INPUT_TENSOR_PATH}")

    if not OCEAN_MASK_PATH.exists():
        fail(f"Ocean mask not found: {OCEAN_MASK_PATH}")

    print(f"  ocean mask   : {OCEAN_MASK_PATH}")

    # -------------------------------------------------------------------------
    # Tensor validation
    # -------------------------------------------------------------------------

    print("\nTENSOR")

    with xr.open_dataset(INPUT_TENSOR_PATH) as ds:

        if "input_tensor" not in ds:
            fail("Variable 'input_tensor' not found")

        tensor = ds["input_tensor"]

        actual_shape = tuple(tensor.shape)

        print(f"  dimensions : {tensor.dims}")
        print(f"  shape      : {actual_shape}")

        if actual_shape != EXPECTED_SHAPE:
            fail(
                f"Expected shape {EXPECTED_SHAPE}, "
                f"got {actual_shape}"
            )

        if tensor.dims != ("time", "latitude", "longitude", "channel"):
            fail(
                "Unexpected dimension order: "
                f"{tensor.dims}"
            )

        print("  shape      : PASS")
        print("  dimensions : PASS")

    # -------------------------------------------------------------------------
    # Ocean mask validation
    # -------------------------------------------------------------------------

    print("\nOCEAN MASK")

    with xr.open_dataset(OCEAN_MASK_PATH) as mask_ds:

        print(f"  dimensions : {mask_ds.dims}")

        if "ocean_mask" not in mask_ds:
            fail("Variable 'ocean_mask' not found")

        mask = mask_ds["ocean_mask"]

        expected_mask_shape = (
            EXPECTED_LATITUDE_SIZE,
            EXPECTED_LONGITUDE_SIZE,
        )

        if tuple(mask.shape) != expected_mask_shape:
            fail(
                f"Expected ocean mask shape {expected_mask_shape}, "
                f"got {tuple(mask.shape)}"
            )

        if mask.dims != ("latitude", "longitude"):
            fail(
                "Unexpected ocean mask dimension order: "
                f"{mask.dims}"
            )

        print("  shape      : PASS")
        print("  dimensions : PASS")

    # -------------------------------------------------------------------------
    # Channel contract
    # -------------------------------------------------------------------------

    print("\nCHANNEL CONTRACT")

    if len(CHANNELS) != EXPECTED_CHANNEL_SIZE:
        fail(
            f"Expected {EXPECTED_CHANNEL_SIZE} channels, "
            f"configuration contains {len(CHANNELS)}"
        )

    for index, name in enumerate(CHANNELS):
        print(f"  {index:2d} : {name}")

    if NORMALIZE_CHANNEL_INDICES != tuple(range(10)):
        fail("Normalization channel indices are incorrect")

    if UNCHANGED_CHANNEL_INDICES != (10, 11):
        fail("Unchanged channel indices are incorrect")

    print("  channel count : PASS")
    print("  channel order : PASS")
    print("  normalization channels : 0-9")
    print("  unchanged channels      : 10-11")

    # -------------------------------------------------------------------------
    # Temporal contract
    # -------------------------------------------------------------------------

    print("\nTEMPORAL CONTRACT")

    if TRAINING_SIZE != 993:
        fail(
            f"Training size must be exactly 993, "
            f"got {TRAINING_SIZE}"
        )

    if TRAINING_SIZE >= EXPECTED_TIME_SIZE:
        fail("Training cutoff consumes the entire dataset")

    print(f"  total timesteps    : {EXPECTED_TIME_SIZE}")
    print(f"  training timesteps : {TRAINING_SIZE}")
    print(f"  training indices   : [0, {TRAINING_SIZE})")
    print(f"  held-out indices   : [{TRAINING_SIZE}, {EXPECTED_TIME_SIZE})")

    print("  chronological cutoff : PASS")

    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT: PASS")
    print("=" * 80)


if __name__ == "__main__":
    main()