"""
OceanEmbed — Phase 8 Part 3
Final train/validation/test split validation.

Validates:
  - Phase 6 normalized tensor
  - LOYO fold structure
  - train/validation/test disjointness
  - recorded date boundaries
  - weekly evaluation/operational windows
  - 7-day duration
  - Wednesday alignment
  - non-overlap
  - containment inside split boundaries
  - ARGO isolation policy
  - absence of physical split tensors

No source artifact is modified.
No complete tensor is loaded into memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_TENSOR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase6"
    / "input_tensor_normalized.nc"
)

SPLIT_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase8"
    / "split_config.json"
)

# Change this only if your Part 2 script used a different filename.
WEEKLY_CONFIG = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase8"
    / "weekly_windows.json"
)

PHASE8_DIR = PROJECT_ROOT / "data" / "processed" / "phase8"


# =============================================================================
# EXPECTED GLOBAL VALUES
# =============================================================================

EXPECTED_TOTAL_DAYS = 1419
EXPECTED_FOLDS = 4
WEEK_DAYS = 7

EXPECTED_YEARS = [2021, 2022, 2023, 2024]


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

PASS = 0
FAIL = 0


def check(condition: bool, label: str) -> bool:
    global PASS, FAIL

    if condition:
        print(f"  PASS  {label}")
        PASS += 1
        return True

    print(f"  FAIL  {label}")
    FAIL += 1
    return False


def fail_fatal(message: str) -> None:
    print(f"\nFATAL: {message}")
    sys.exit(1)


def load_json(path: Path) -> dict:
    if not path.exists():
        fail_fatal(f"Missing configuration: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        fail_fatal(f"Could not read {path}: {exc}")


def normalize_timestamp(value) -> np.datetime64:
    return np.datetime64(value, "ns")


def timestamp_string(value) -> str:
    return str(normalize_timestamp(value))


def date_range_from_indices(
    time_values: np.ndarray,
    start: int,
    end: int,
):
    if start < 0 or end > len(time_values) or start >= end:
        return None, None

    return (
        normalize_timestamp(time_values[start]),
        normalize_timestamp(time_values[end - 1]),
    )


def indices_to_set(ranges):
    result = set()

    for start, end in ranges:
        result.update(range(start, end))

    return result


# =============================================================================
# HEADER
# =============================================================================

print("=" * 90)
print("OceanEmbed — Phase 8 Final Split Validation")
print("=" * 90)

print()
print("Purpose:")
print("  Independently validate the Phase 8 chronological LOYO split")
print("  and weekly evaluation/operational sampling configuration.")

print()
print("Rules:")
print("  - No source artifact is modified.")
print("  - No complete tensor is loaded.")
print("  - Training uses chronological yearly folds.")
print("  - Validation and test windows are non-overlapping weekly blocks.")
print("  - Weekly blocks must be Wednesday → Tuesday.")
print("  - ARGO remains completely isolated/deferred.")


# =============================================================================
# [1] REQUIRED ARTIFACTS
# =============================================================================

print("\n" + "=" * 90)
print("[1] REQUIRED ARTIFACTS")
print("=" * 90)

check(
    SOURCE_TENSOR.exists(),
    "Phase 6 normalized tensor exists",
)

check(
    SPLIT_CONFIG.exists(),
    "split_config.json exists",
)

check(
    WEEKLY_CONFIG.exists(),
    "weekly window configuration exists",
)


if FAIL:
    fail_fatal("Required artifact missing.")


# =============================================================================
# [2] SOURCE TENSOR CALENDAR
# =============================================================================

print("\n" + "=" * 90)
print("[2] SOURCE TENSOR CALENDAR")
print("=" * 90)

ds = xr.open_dataset(SOURCE_TENSOR)

try:
    check(
        "time" in ds.coords or "time" in ds.variables,
        "time coordinate exists",
    )

    if "time" not in ds:
        fail_fatal("Source tensor has no time coordinate.")

    time_values = np.asarray(ds["time"].values)

    check(
        len(time_values) == EXPECTED_TOTAL_DAYS,
        f"total timestamps = {EXPECTED_TOTAL_DAYS}",
    )

    if len(time_values) != EXPECTED_TOTAL_DAYS:
        fail_fatal(
            f"Expected {EXPECTED_TOTAL_DAYS} timestamps, "
            f"found {len(time_values)}."
        )

    first_time = normalize_timestamp(time_values[0])
    last_time = normalize_timestamp(time_values[-1])

    print(f"  First timestamp : {first_time}")
    print(f"  Last timestamp  : {last_time}")

    # Confirm daily cadence.
    deltas = np.diff(time_values.astype("datetime64[D]")).astype(int)

    check(
        np.all(deltas == 1),
        "tensor calendar has daily cadence",
    )

finally:
    ds.close()


# =============================================================================
# [3] LOAD SPLIT CONFIG
# =============================================================================

print("\n" + "=" * 90)
print("[3] SPLIT CONFIGURATION")
print("=" * 90)

split_config = load_json(SPLIT_CONFIG)

check(
    split_config.get("phase") == 8,
    "phase = 8",
)

check(
    split_config.get("part") == 1,
    "split configuration part = 1",
)

check(
    split_config.get("split_method") == "leave-one-year-out",
    "split method = leave-one-year-out",
)

check(
    split_config.get("sampling_cadence") == "daily",
    "sampling cadence = daily",
)

check(
    split_config.get("random_shuffle") is False,
    "random shuffle disabled",
)

check(
    split_config.get("total_days") == EXPECTED_TOTAL_DAYS,
    "configured total_days = 1419",
)

folds = split_config.get("folds", [])

check(
    len(folds) == EXPECTED_FOLDS,
    "exactly 4 folds configured",
)


# =============================================================================
# [4] LOYO FOLD VALIDATION
# =============================================================================

print("\n" + "=" * 90)
print("[4] LEAVE-ONE-YEAR-OUT FOLDS")
print("=" * 90)

for fold in folds:

    fold_id = fold.get("fold_id")
    test_year = fold.get("test_year")

    print()
    print(f"  Fold {fold_id} — Test year {test_year}")

    check(
        test_year in EXPECTED_YEARS,
        f"Fold {fold_id} test year is valid",
    )

    test_indices = fold.get("test_indices")

    check(
        isinstance(test_indices, list) and len(test_indices) == 2,
        f"Fold {fold_id} test indices are [start, end]",
    )

    if not isinstance(test_indices, list) or len(test_indices) != 2:
        continue

    test_start, test_end = test_indices

    check(
        0 <= test_start < test_end <= EXPECTED_TOTAL_DAYS,
        f"Fold {fold_id} test indices are in bounds",
    )

    train_ranges = fold.get("train_indices", [])

    train_set = indices_to_set(train_ranges)
    test_set = set(range(test_start, test_end))

    validation_indices = fold.get("validation_indices")

    check(
        isinstance(validation_indices, list)
        and len(validation_indices) == 2,
        f"Fold {fold_id} validation indices are [start, end]",
    )

    if not isinstance(validation_indices, list) or len(validation_indices) != 2:
        continue

    val_start, val_end = validation_indices
    val_set = set(range(val_start, val_end))

    # -------------------------------------------------------------------------
    # Bounds
    # -------------------------------------------------------------------------

    check(
        all(
            isinstance(x, int)
            for x in [test_start, test_end, val_start, val_end]
        ),
        f"Fold {fold_id} split indices are integers",
    )

    check(
        0 <= val_start < val_end <= EXPECTED_TOTAL_DAYS,
        f"Fold {fold_id} validation indices are in bounds",
    )

    # -------------------------------------------------------------------------
    # Disjointness
    # -------------------------------------------------------------------------

    check(
        train_set.isdisjoint(val_set),
        f"Fold {fold_id} train ∩ validation = ∅",
    )

    check(
        train_set.isdisjoint(test_set),
        f"Fold {fold_id} train ∩ test = ∅",
    )

    check(
        val_set.isdisjoint(test_set),
        f"Fold {fold_id} validation ∩ test = ∅",
    )

    # -------------------------------------------------------------------------
    # Counts
    # -------------------------------------------------------------------------

    configured_counts = fold.get("counts", {})

    check(
        len(train_set) == configured_counts.get("train_days"),
        f"Fold {fold_id} train day count",
    )

    check(
        len(val_set) == configured_counts.get("validation_days"),
        f"Fold {fold_id} validation day count",
    )

    check(
        len(test_set) == configured_counts.get("test_days"),
        f"Fold {fold_id} test day count",
    )

    # -------------------------------------------------------------------------
    # Test-year calendar
    # -------------------------------------------------------------------------

    test_first, test_last = date_range_from_indices(
        time_values,
        test_start,
        test_end,
    )

    expected_test_year = np.datetime64(f"{test_year}-01-01", "D")

    check(
        test_first.astype("datetime64[Y]") == expected_test_year.astype("datetime64[Y]"),
        f"Fold {fold_id} test starts in {test_year}",
    )

    check(
        test_last.astype("datetime64[Y]") == expected_test_year.astype("datetime64[Y]"),
        f"Fold {fold_id} test ends in {test_year}",
    )

    print(
        f"    test : [{test_start}:{test_end}] "
        f"{test_first} -> {test_last}"
    )

    # -------------------------------------------------------------------------
    # Validation date metadata
    # -------------------------------------------------------------------------

    val_first, val_last = date_range_from_indices(
        time_values,
        val_start,
        val_end,
    )

    dates = fold.get("dates", {})

    if "validation_start" in dates:
        check(
            normalize_timestamp(dates["validation_start"]) == val_first,
            f"Fold {fold_id} validation start date",
        )

    if "validation_end" in dates:
        check(
            normalize_timestamp(dates["validation_end"]) == val_last,
            f"Fold {fold_id} validation end date",
        )

    if "test_start" in dates:
        check(
            normalize_timestamp(dates["test_start"]) == test_first,
            f"Fold {fold_id} test start date",
        )

    if "test_end" in dates:
        check(
            normalize_timestamp(dates["test_end"]) == test_last,
            f"Fold {fold_id} test end date",
        )


# =============================================================================
# [5] WEEKLY CONFIGURATION
# =============================================================================

print("\n" + "=" * 90)
print("[5] WEEKLY EVALUATION / OPERATIONAL WINDOWS")
print("=" * 90)

weekly_config = load_json(WEEKLY_CONFIG)

check(
    weekly_config.get("phase") == 8,
    "weekly configuration phase = 8",
)

weekly_folds = weekly_config.get("folds")

if weekly_folds is None:
    # Support a simpler configuration where validation/test windows
    # are stored directly.
    weekly_folds = weekly_config.get("weekly_windows", [])

check(
    isinstance(weekly_folds, list),
    "weekly fold configuration is a list",
)

if not isinstance(weekly_folds, list):
    weekly_folds = []


# =============================================================================
# WEEK WINDOW PARSER
# =============================================================================

def get_windows(fold_cfg, key):
    """
    Accept common representations:

      "validation_windows": [[start, end], ...]
      "test_windows": [[start, end], ...]

    or:

      [{"start_index": x, "end_index": y}, ...]

    or:

      [{"indices": [x, y]}, ...]
    """

    raw = fold_cfg.get(key, [])

    windows = []

    for item in raw:

        if isinstance(item, list) and len(item) == 2:
            windows.append((int(item[0]), int(item[1])))

        elif isinstance(item, dict):

            if "indices" in item:
                start, end = item["indices"]
                windows.append((int(start), int(end)))

            elif "start_index" in item and "end_index" in item:
                windows.append(
                    (
                        int(item["start_index"]),
                        int(item["end_index"]),
                    )
                )

    return windows


def validate_week_windows(
    label,
    windows,
    split_start,
    split_end,
):
    print(f"\n  {label}")

    if not windows:
        check(False, f"{label}: windows exist")
        return

    check(
        all((end - start) == WEEK_DAYS for start, end in windows),
        f"{label}: every window is exactly 7 days",
    )

    check(
        all(
            split_start <= start < end <= split_end
            for start, end in windows
        ),
        f"{label}: every window is inside split boundary",
    )

    # Wednesday alignment.
    aligned = True

    for start, end in windows:
        date = normalize_timestamp(time_values[start])

        # NumPy datetime64 -> Python-compatible weekday calculation.
        day_index = int(
            (date.astype("datetime64[D]") - np.datetime64("1970-01-01", "D"))
            .astype(int)
        )

        # 1970-01-01 was Thursday.
        # Wednesday = 2 when Monday = 0.
        weekday = (day_index + 3) % 7

        if weekday != 2:
            aligned = False
            break

    check(
        aligned,
        f"{label}: Wednesday-aligned",
    )

    # Non-overlap.
    ordered = sorted(windows)

    non_overlapping = True

    for previous, current in zip(ordered, ordered[1:]):
        if previous[1] > current[0]:
            non_overlapping = False
            break

    check(
        non_overlapping,
        f"{label}: windows do not overlap",
    )

    # Strict 7-day stepping.
    consecutive = True

    for previous, current in zip(ordered, ordered[1:]):
        if current[0] != previous[1]:
            consecutive = False
            break

    # This is informational rather than a hard requirement.
    print(
        f"    windows : {len(windows)}"
    )

    if windows:
        first_start, first_end = ordered[0]
        last_start, last_end = ordered[-1]

        print(
            f"    first   : [{first_start}:{first_end}] "
            f"{time_values[first_start]} -> "
            f"{time_values[first_end - 1]}"
        )

        print(
            f"    last    : [{last_start}:{last_end}] "
            f"{time_values[last_start]} -> "
            f"{time_values[last_end - 1]}"
        )


# =============================================================================
# MATCH WEEKLY FOLDS TO LOYO FOLDS
# =============================================================================

for fold in folds:

    fold_id = fold["fold_id"]

    matching = [
        wf
        for wf in weekly_folds
        if wf.get("fold_id") == fold_id
    ]

    check(
        len(matching) == 1,
        f"Fold {fold_id} weekly configuration exists",
    )

    if len(matching) != 1:
        continue

    weekly_fold = matching[0]

    val_start, val_end = fold["validation_indices"]
    test_start, test_end = fold["test_indices"]

    validation_windows = get_windows(
        weekly_fold,
        "validation_windows",
    )

    test_windows = get_windows(
        weekly_fold,
        "test_windows",
    )

    validate_week_windows(
        f"Fold {fold_id} validation windows",
        validation_windows,
        val_start,
        val_end,
    )

    validate_week_windows(
        f"Fold {fold_id} test windows",
        test_windows,
        test_start,
        test_end,
    )

# =============================================================================
# [6] OPERATIONAL CONTRACT
# =============================================================================

print("\n" + "=" * 90)
print("[6] OPERATIONAL CONTRACT")
print("=" * 90)

operational = weekly_config.get("operational_policy", {})

check(
    isinstance(operational, dict),
    "operational contract exists",
)

if isinstance(operational, dict):

    # "weekly" means a 7-day operational window
    check(
        operational.get("cadence") == "weekly",
        "operational window = 7 days",
    )

    # Wednesday-to-Tuesday means Wednesday-aligned
    check(
        operational.get("alignment") == "Wednesday-to-Tuesday",
        "operational alignment = Wednesday",
    )

    check(
        operational.get("stride_days") == 7,
        "operational stride = 7 days",
    )

    check(
        operational.get("overlap_allowed") is False,
        "operational windows are non-overlapping",
    )
# =============================================================================
# [7] ARGO ISOLATION
# =============================================================================

print("\n" + "=" * 90)
print("[7] ARGO ISOLATION")
print("=" * 90)

argo_policy = split_config.get("argo_policy", {})

check(
    argo_policy.get("status") == "deferred",
    "ARGO status = deferred",
)

check(
    argo_policy.get("isolation") == "test-window-only",
    "ARGO isolation = test-window-only",
)

check(
    argo_policy.get("allowed_for_training") is False,
    "ARGO excluded from training",
)

check(
    argo_policy.get("allowed_for_validation") is False,
    "ARGO excluded from validation",
)

check(
    argo_policy.get("allowed_for_hyperparameter_tuning") is False,
    "ARGO excluded from hyperparameter tuning",
)

check(
    argo_policy.get("allowed_for_model_selection") is False,
    "ARGO excluded from model selection",
)

check(
    argo_policy.get("purpose") == "final independent truth check",
    "ARGO purpose = final independent truth check",
)


# =============================================================================
# [8] PHYSICAL TENSOR DUPLICATION CHECK
# =============================================================================

print("\n" + "=" * 90)
print("[8] ARTIFACT INTEGRITY")
print("=" * 90)

# These are intentionally checked only as warnings in the output.
# The canonical Phase 8 design requires configuration-only splitting.

physical_split_names = [
    "train_tensor.nc",
    "val_tensor.nc",
    "validation_tensor.nc",
    "test_tensor.nc",
]

physical_split_paths = [
    PHASE8_DIR / name
    for name in physical_split_names
]

physical_copies = [
    path
    for path in physical_split_paths
    if path.exists()
]

check(
    len(physical_copies) == 0,
    "no physical train/validation/test tensor copies",
)

if physical_copies:
    for path in physical_copies:
        print(f"    WARNING: {path}")


# =============================================================================
# [9] GLOBAL FOLD COVERAGE
# =============================================================================

print("\n" + "=" * 90)
print("[9] GLOBAL FOLD COVERAGE")
print("=" * 90)

seen_test_years = []

for fold in folds:
    seen_test_years.append(fold.get("test_year"))

check(
    sorted(seen_test_years) == EXPECTED_YEARS,
    "all four calendar years appear exactly once as test years",
)

check(
    len(set(seen_test_years)) == len(seen_test_years),
    "no test year is duplicated",
)


# =============================================================================
# [10] FINAL RESULT
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 8 — PART 3 RESULT")
print("=" * 90)

print(f"PASS: {PASS}")
print(f"FAIL: {FAIL}")

if FAIL == 0:
    print()
    print("PART 3 SPLIT VALIDATION: PASS")
    print()
    print(
        "The Phase 8 chronological LOYO folds, weekly evaluation/"
        "operational windows, calendar boundaries, ARGO isolation policy, "
        "and configuration-only artifact strategy passed validation."
    )
else:
    print()
    print("PART 3 SPLIT VALIDATION: FAIL")
    print()
    print(
        "One or more Phase 8 invariants failed. "
        "Do not finalize Phase 8 until the failures are resolved."
    )
    sys.exit(1)