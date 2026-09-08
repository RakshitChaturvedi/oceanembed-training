from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# Paths
# =============================================================================

SOURCE_TENSOR = Path(
    "data/processed/phase6/input_tensor_normalized.nc"
)

SPLIT_CONFIG = Path(
    "data/processed/phase8/split_config.json"
)

OUTPUT_DIR = Path(
    "data/processed/phase8"
)

OUTPUT_FILE = OUTPUT_DIR / "weekly_windows.json"


# =============================================================================
# Contract
# =============================================================================

WINDOW_DAYS = 7
WEDNESDAY = 2  # Python datetime.weekday(): Monday=0 ... Sunday=6


# =============================================================================
# Validation helpers
# =============================================================================

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed

    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")
        failed += 1


# =============================================================================
# Header
# =============================================================================

print("=" * 90)
print("OceanEmbed — Phase 8 Part 2")
print("Weekly Evaluation / Operational Window Generation")
print("=" * 90)

print(
    """
Purpose:

  Generate 7-day, non-overlapping Wednesday-aligned windows
  for validation, test, and operational evaluation.

Policy:

  - Training remains daily.
  - Validation uses complete Wednesday→Tuesday weeks.
  - Test uses complete Wednesday→Tuesday weeks.
  - Operational serving uses the same weekly contract.
  - Windows are derived from the actual tensor calendar.
  - Partial weeks are excluded.
  - No tensor data is copied.
"""
)


# =============================================================================
# [1] Required artifacts
# =============================================================================

print("\n" + "=" * 90)
print("[1] REQUIRED ARTIFACTS")
print("=" * 90)

check(
    "source tensor exists",
    SOURCE_TENSOR.exists(),
    str(SOURCE_TENSOR),
)

check(
    "split config exists",
    SPLIT_CONFIG.exists(),
    str(SPLIT_CONFIG),
)

if failed:
    raise SystemExit(1)


# =============================================================================
# [2] Load split configuration
# =============================================================================

print("\n" + "=" * 90)
print("[2] SPLIT CONFIGURATION")
print("=" * 90)

with SPLIT_CONFIG.open("r", encoding="utf-8") as f:
    split_config = json.load(f)

check(
    "split config is dictionary",
    isinstance(split_config, dict),
)

if not isinstance(split_config, dict):
    raise RuntimeError("Invalid split configuration.")

check(
    "phase is 8",
    split_config.get("phase") == 8,
)

check(
    "split method is leave-one-year-out",
    split_config.get("split_method") == "leave-one-year-out",
)

check(
    "total days is 1419",
    split_config.get("total_days") == 1419,
)

folds = split_config.get("folds")

check(
    "folds is a list",
    isinstance(folds, list),
)

if not isinstance(folds, list):
    raise RuntimeError("split_config['folds'] must be a list.")

check(
    "four folds present",
    len(folds) == 4,
)


# =============================================================================
# [3] Open tensor
# =============================================================================

print("\n" + "=" * 90)
print("[3] SOURCE TENSOR")
print("=" * 90)

ds = xr.open_dataset(SOURCE_TENSOR)

print(f"Dimensions: {dict(ds.sizes)}")
print(f"Variables : {list(ds.data_vars)}")

check(
    "time dimension exists",
    "time" in ds.sizes,
)

check(
    "time count is 1419",
    ds.sizes.get("time") == 1419,
)

check(
    "time coordinate exists",
    "time" in ds.coords,
)

if failed:
    ds.close()
    raise SystemExit(1)


# =============================================================================
# [4] Tensor calendar
# =============================================================================

print("\n" + "=" * 90)
print("[4] TENSOR CALENDAR")
print("=" * 90)

time_values = ds["time"].values

check(
    "1419 timestamps loaded",
    len(time_values) == 1419,
)

# Convert only the coordinate, never the tensor itself.
timestamps = np.asarray(time_values)

print(f"First timestamp: {timestamps[0]}")
print(f"Last timestamp : {timestamps[-1]}")

# Verify strictly increasing time.
time_diffs = np.diff(timestamps)

check(
    "timestamps strictly increasing",
    np.all(time_diffs > np.timedelta64(0, "D")),
)

# Verify daily cadence.
check(
    "timestamps have daily cadence",
    np.all(time_diffs == np.timedelta64(1, "D")),
)


# =============================================================================
# [5] Calendar → Python dates
# =============================================================================

print("\n" + "=" * 90)
print("[5] CALENDAR ALIGNMENT")
print("=" * 90)

# Convert datetime64[D] safely for weekday calculations.
dates = timestamps.astype("datetime64[D]")

first_date = dates[0]

# NumPy weekday:
# 1970-01-01 was Thursday (weekday=3).
# Use modulo arithmetic to determine weekday.
epoch = np.datetime64("1970-01-01", "D")

weekday_values = (
    (dates - epoch).astype("timedelta64[D]").astype(np.int64) + 3
) % 7

print(f"First date: {first_date}")
print(f"First weekday index: {weekday_values[0]}")
print("Weekday convention: Monday=0 ... Wednesday=2 ... Sunday=6")

check(
    "first tensor date is 2021-01-01",
    str(first_date) == "2021-01-01",
)

# Find every Wednesday in the actual tensor.
wednesday_indices = np.where(
    weekday_values == WEDNESDAY
)[0]

print(f"Wednesday timestamps available: {len(wednesday_indices)}")

check(
    "Wednesday timestamps exist",
    len(wednesday_indices) > 0,
)


# =============================================================================
# [6] Window generator
# =============================================================================

print("\n" + "=" * 90)
print("[6] WINDOW GENERATION")
print("=" * 90)


def generate_windows(
    interval_start: int,
    interval_end: int,
) -> list[dict]:

    windows = []

    # Candidate start must:
    #
    #   1. be Wednesday
    #   2. leave 7 complete days inside the interval
    #
    for start in wednesday_indices:

        start = int(start)
        end = start + WINDOW_DAYS

        # Entire window must fit inside [interval_start, interval_end).
        if start < interval_start:
            continue

        if end > interval_end:
            continue

        # Sanity check: exactly seven timestamps.
        if end - start != WINDOW_DAYS:
            continue

        window_dates = dates[start:end]

        # Explicit calendar check.
        if len(window_dates) != WINDOW_DAYS:
            continue

        if window_dates[-1] != window_dates[0] + np.timedelta64(
            WINDOW_DAYS - 1,
            "D",
        ):
            continue

        windows.append(
            {
                "start_index": start,
                "end_index": end,
                "length_days": WINDOW_DAYS,
                "start_date": str(window_dates[0]),
                "end_date": str(window_dates[-1]),
            }
        )

    return windows


# =============================================================================
# [7] Generate fold windows
# =============================================================================

print("\n" + "=" * 90)
print("[7] FOLD WINDOW GENERATION")
print("=" * 90)

fold_outputs = []

for fold in folds:

    fold_id = fold["fold_id"]
    test_year = fold["test_year"]

    validation_start, validation_end = fold["validation_indices"]
    test_start, test_end = fold["test_indices"]

    print("\n" + "-" * 90)
    print(f"Fold {fold_id} — Test year {test_year}")
    print("-" * 90)

    validation_windows = generate_windows(
        validation_start,
        validation_end,
    )

    test_windows = generate_windows(
        test_start,
        test_end,
    )

    print(
        f"Validation windows: {len(validation_windows)}"
    )

    print(
        f"Test windows      : {len(test_windows)}"
    )

    # -------------------------------------------------------------------------
    # Validate validation windows
    # -------------------------------------------------------------------------

    for i, window in enumerate(validation_windows):

        start = window["start_index"]
        end = window["end_index"]

        check(
            f"fold {fold_id} validation window {i} length",
            end - start == WINDOW_DAYS,
        )

        check(
            f"fold {fold_id} validation window {i} Wednesday",
            weekday_values[start] == WEDNESDAY,
        )

        check(
            f"fold {fold_id} validation window {i} inside boundary",
            start >= validation_start
            and end <= validation_end,
        )

    # -------------------------------------------------------------------------
    # Validate test windows
    # -------------------------------------------------------------------------

    for i, window in enumerate(test_windows):

        start = window["start_index"]
        end = window["end_index"]

        check(
            f"fold {fold_id} test window {i} length",
            end - start == WINDOW_DAYS,
        )

        check(
            f"fold {fold_id} test window {i} Wednesday",
            weekday_values[start] == WEDNESDAY,
        )

        check(
            f"fold {fold_id} test window {i} inside boundary",
            start >= test_start
            and end <= test_end,
        )

    # -------------------------------------------------------------------------
    # Validate non-overlap
    # -------------------------------------------------------------------------

    for name, windows in (
        ("validation", validation_windows),
        ("test", test_windows),
    ):

        for previous, current in zip(
            windows,
            windows[1:],
        ):

            check(
                f"fold {fold_id} {name} windows non-overlapping",
                previous["end_index"]
                <= current["start_index"],
            )

            check(
                f"fold {fold_id} {name} windows 7-day spaced",
                current["start_index"]
                - previous["start_index"]
                == WINDOW_DAYS,
            )

    fold_outputs.append(
        {
            "fold_id": fold_id,
            "test_year": test_year,
            "validation_windows": validation_windows,
            "test_windows": test_windows,
        }
    )


# =============================================================================
# [8] Operational contract
# =============================================================================

print("\n" + "=" * 90)
print("[8] OPERATIONAL CONTRACT")
print("=" * 90)

operational_contract = {
    "window_length_days": WINDOW_DAYS,
    "alignment": "Wednesday-to-Tuesday",
    "start_weekday": "Wednesday",
    "end_weekday": "Tuesday",
    "stride_days": WINDOW_DAYS,
    "overlap": False,
    "training_sampling": {
        "cadence": "daily",
        "stride_days": 1,
        "overlap_allowed": True,
    },
    "evaluation_sampling": {
        "cadence": "weekly",
        "stride_days": WINDOW_DAYS,
        "overlap_allowed": False,
    },
    "serving_sampling": {
        "cadence": "weekly",
        "stride_days": WINDOW_DAYS,
        "overlap_allowed": False,
    },
}

print(
    json.dumps(
        operational_contract,
        indent=2,
    )
)


# =============================================================================
# [9] Global validation
# =============================================================================

print("\n" + "=" * 90)
print("[9] GLOBAL VALIDATION")
print("=" * 90)

# Verify test years are distinct.
test_years = [
    fold["test_year"]
    for fold in fold_outputs
]

check(
    "four distinct test years",
    sorted(test_years) == [2021, 2022, 2023, 2024],
)

# Verify every generated window is exactly 7 days.
all_windows = []

for fold in fold_outputs:

    all_windows.extend(
        fold["validation_windows"]
    )

    all_windows.extend(
        fold["test_windows"]
    )

check(
    "all generated windows are 7 days",
    all(
        w["end_index"] - w["start_index"]
        == WINDOW_DAYS
        for w in all_windows
    ),
)

check(
    "all generated windows start Wednesday",
    all(
        weekday_values[w["start_index"]]
        == WEDNESDAY
        for w in all_windows
    ),
)

# Verify every window's date span.
check(
    "all generated windows have correct date span",
    all(
        (
            np.datetime64(w["end_date"])
            - np.datetime64(w["start_date"])
        )
        == np.timedelta64(6, "D")
        for w in all_windows
    ),
)


# =============================================================================
# [10] Output
# =============================================================================

print("\n" + "=" * 90)
print("[10] OUTPUT")
print("=" * 90)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

artifact = {
    "phase": 8,
    "part": 2,
    "artifact": "weekly evaluation and operational windows",
    "source_tensor": str(SOURCE_TENSOR),
    "window_length_days": WINDOW_DAYS,
    "alignment": "Wednesday-to-Tuesday",
    "stride_days": WINDOW_DAYS,
    "overlap": False,
    "training_policy": {
        "cadence": "daily",
        "stride_days": 1,
        "overlap_allowed": True,
        "window_length_days": 7,
    },
    "evaluation_policy": {
        "cadence": "weekly",
        "alignment": "Wednesday-to-Tuesday",
        "stride_days": 7,
        "overlap_allowed": False,
    },
    "operational_policy": {
        "cadence": "weekly",
        "alignment": "Wednesday-to-Tuesday",
        "stride_days": 7,
        "overlap_allowed": False,
    },
    "folds": fold_outputs,
}

with OUTPUT_FILE.open(
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        artifact,
        f,
        indent=2,
    )

print(f"Written: {OUTPUT_FILE}")


# =============================================================================
# [11] Summary
# =============================================================================

print("\n" + "=" * 90)
print("[11] SUMMARY")
print("=" * 90)

for fold in fold_outputs:

    print(
        f"Fold {fold['fold_id']} "
        f"(test {fold['test_year']}): "
        f"validation={len(fold['validation_windows'])}, "
        f"test={len(fold['test_windows'])}"
    )

print(
    f"\nTotal evaluation windows: {len(all_windows)}"
)

ds.close()


# =============================================================================
# Final result
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 8 — PART 2 RESULT")
print("=" * 90)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:

    print(
        """
PART 2 WEEKLY WINDOW GENERATION: PASS

Validation and test windows were generated from the actual
tensor calendar using complete Wednesday-to-Tuesday weeks.

No tensor copies were created.
"""
    )

else:

    print(
        """
PART 2 WEEKLY WINDOW GENERATION: FAIL

Do not proceed to Phase 8 final validation.
"""
    )

print("=" * 90)

raise SystemExit(
    0 if failed == 0 else 1
)