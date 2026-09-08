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

OUTPUT_DIR = Path(
    "data/processed/phase8"
)

OUTPUT_FILE = OUTPUT_DIR / "split_config.json"


# =============================================================================
# Split contract
# =============================================================================

EXPECTED_TOTAL_DAYS = 1419

TEST_YEARS = [2021, 2022, 2023, 2024]

VALIDATION_DAYS = 30


# =============================================================================
# Validation
# =============================================================================

passed = 0
failed = 0


def check(
    name: str,
    condition: bool,
    detail: str = "",
) -> None:
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
print("OceanEmbed — Phase 8 Part 1: LOYO Split Configuration")
print("=" * 90)

print(
    """
Purpose:
  Construct chronological leave-one-year-out folds from the
  Phase 6 normalized tensor.

Policy:
  - Daily samples.
  - No random shuffling.
  - One calendar year is held out as test for each fold.
  - Validation consists of the final 30 days of the training
    interval/block according to the Phase 8 contract.
  - No physical tensor copies are created.
  - Split configuration stores index ranges only.
"""
)


# =============================================================================
# [1] Required artifact
# =============================================================================

print("\n" + "=" * 90)
print("[1] REQUIRED ARTIFACT")
print("=" * 90)

check(
    "normalized tensor exists",
    SOURCE_TENSOR.exists(),
    str(SOURCE_TENSOR),
)

if failed:
    raise SystemExit(1)


# =============================================================================
# [2] Open tensor and inspect calendar
# =============================================================================

print("\n" + "=" * 90)
print("[2] SOURCE TENSOR / CALENDAR")
print("=" * 90)

ds = xr.open_dataset(SOURCE_TENSOR)

check(
    "time coordinate exists",
    "time" in ds.coords,
)

if "time" not in ds.coords:
    ds.close()
    raise RuntimeError("Source tensor has no time coordinate.")

time = ds["time"].values

print(f"Total timestamps: {len(time)}")
print(f"First timestamp : {time[0]}")
print(f"Last timestamp  : {time[-1]}")

check(
    "total days = 1419",
    len(time) == EXPECTED_TOTAL_DAYS,
    f"actual={len(time)}",
)


# =============================================================================
# [3] Derive calendar years from actual timestamps
# =============================================================================

print("\n" + "=" * 90)
print("[3] CALENDAR YEAR BOUNDARIES")
print("=" * 90)

# Convert each timestamp to a calendar year.
years = np.array(
    [
        t.astype("datetime64[Y]").astype(int) + 1970
        for t in time
    ],
    dtype=np.int32,
)

unique_years = np.unique(years)

print(f"Years present: {unique_years.tolist()}")

check(
    "expected years present",
    np.array_equal(
        unique_years,
        np.array(TEST_YEARS),
    ),
    f"actual={unique_years.tolist()}",
)


# =============================================================================
# Derive exact index ranges from calendar
# =============================================================================

year_ranges: dict[int, tuple[int, int]] = {}

for year in TEST_YEARS:
    indices = np.where(years == year)[0]

    check(
        f"{year} exists",
        len(indices) > 0,
    )

    if len(indices) == 0:
        continue

    start = int(indices[0])
    end = int(indices[-1]) + 1

    year_ranges[year] = (start, end)

    print(
        f"{year}: "
        f"[{start}, {end}) "
        f"length={end - start} "
        f"{time[start]} -> {time[end - 1]}"
    )


# =============================================================================
# [4] Validate expected calendar boundaries
# =============================================================================

print("\n" + "=" * 90)
print("[4] CALENDAR VALIDATION")
print("=" * 90)

expected_ranges = {
    2021: (0, 365),
    2022: (365, 730),
    2023: (730, 1095),
    2024: (1095, 1419),
}

for year in TEST_YEARS:
    actual = year_ranges[year]
    expected = expected_ranges[year]

    check(
        f"{year} index range",
        actual == expected,
        f"expected={expected}, actual={actual}",
    )


# =============================================================================
# [5] Construct LOYO folds
# =============================================================================

print("\n" + "=" * 90)
print("[5] CONSTRUCT LOYO FOLDS")
print("=" * 90)

folds = []

for fold_id, test_year in enumerate(TEST_YEARS, start=1):

    test_start, test_end = year_ranges[test_year]

    # -------------------------------------------------------------------------
    # All data except the held-out test year is initially training data.
    # -------------------------------------------------------------------------

    training_blocks = []

    if test_start > 0:
        training_blocks.append(
            [0, test_start]
        )

    if test_end < EXPECTED_TOTAL_DAYS:
        training_blocks.append(
            [test_end, EXPECTED_TOTAL_DAYS]
        )

    # -------------------------------------------------------------------------
    # Validation policy:
    #
    # Take the final 30 days of the LAST chronological training block.
    #
    # Fold 1 therefore uses the last 30 available training days:
    #   [1389, 1419)
    #
    # Fold 2:
    #   [335, 365)
    #
    # Fold 3:
    #   [700, 730)
    #
    # Fold 4:
    #   [1065, 1095)
    # -------------------------------------------------------------------------

    validation_block_index = len(training_blocks) - 1

    validation_block = training_blocks[
        validation_block_index
    ]

    validation_start = (
        validation_block[1] - VALIDATION_DAYS
    )

    validation_end = validation_block[1]

    # Replace the final training block with its non-validation portion.
    training_blocks = training_blocks.copy()

    original_start = training_blocks[
        validation_block_index
    ][0]

    if validation_start > original_start:
        training_blocks[
            validation_block_index
        ] = [
            original_start,
            validation_start,
        ]
    else:
        training_blocks.pop(
            validation_block_index
        )

    # -------------------------------------------------------------------------
    # Date helpers
    # -------------------------------------------------------------------------

    test_start_date = str(time[test_start])
    test_end_date = str(time[test_end - 1])

    validation_start_date = str(time[validation_start])
    validation_end_date = str(time[validation_end - 1])

    train_start_date = str(
        time[training_blocks[0][0]]
    ) if training_blocks else None

    train_end_date = str(
        time[training_blocks[-1][1] - 1]
    ) if training_blocks else None

    # -------------------------------------------------------------------------
    # Calculate counts
    # -------------------------------------------------------------------------

    train_days = sum(
        end - start
        for start, end in training_blocks
    )

    validation_days = (
        validation_end - validation_start
    )

    test_days = (
        test_end - test_start
    )

    print(f"\nFold {fold_id} — Test year {test_year}")

    print(
        f"  Train      : {training_blocks}"
        f" ({train_days} days)"
    )

    print(
        f"  Validation : "
        f"[{validation_start}, {validation_end})"
        f" ({validation_days} days)"
    )

    print(
        f"  Test       : "
        f"[{test_start}, {test_end})"
        f" ({test_days} days)"
    )

    fold = {
        "fold_id": fold_id,
        "test_year": test_year,

        "train_indices": training_blocks,

        "validation_indices": [
            validation_start,
            validation_end,
        ],

        "test_indices": [
            test_start,
            test_end,
        ],

        "counts": {
            "train_days": train_days,
            "validation_days": validation_days,
            "test_days": test_days,
        },

        "dates": {
            "validation_start": validation_start_date,
            "validation_end": validation_end_date,
            "test_start": test_start_date,
            "test_end": test_end_date,
        },
    }

    folds.append(fold)


# =============================================================================
# [6] Fold validation
# =============================================================================

print("\n" + "=" * 90)
print("[6] FOLD VALIDATION")
print("=" * 90)


def expand_ranges(
    ranges: list[list[int]],
) -> set[int]:
    indices: set[int] = set()

    for start, end in ranges:
        indices.update(range(start, end))

    return indices


for fold in folds:

    fold_id = fold["fold_id"]

    train_set = expand_ranges(
        fold["train_indices"]
    )

    validation_start, validation_end = (
        fold["validation_indices"]
    )

    test_start, test_end = (
        fold["test_indices"]
    )

    validation_set = set(
        range(
            validation_start,
            validation_end,
        )
    )

    test_set = set(
        range(
            test_start,
            test_end,
        )
    )

    # -------------------------------------------------------------
    # Bounds
    # -------------------------------------------------------------

    check(
        f"Fold {fold_id} train bounds",
        all(
            0 <= i < EXPECTED_TOTAL_DAYS
            for i in train_set
        ),
    )

    check(
        f"Fold {fold_id} validation bounds",
        all(
            0 <= i < EXPECTED_TOTAL_DAYS
            for i in validation_set
        ),
    )

    check(
        f"Fold {fold_id} test bounds",
        all(
            0 <= i < EXPECTED_TOTAL_DAYS
            for i in test_set
        ),
    )

    # -------------------------------------------------------------
    # Expected counts
    # -------------------------------------------------------------

    check(
        f"Fold {fold_id} validation = 30 days",
        len(validation_set) == 30,
        f"actual={len(validation_set)}",
    )

    # -------------------------------------------------------------
    # Disjointness
    # -------------------------------------------------------------

    check(
        f"Fold {fold_id} train ∩ validation = empty",
        train_set.isdisjoint(validation_set),
    )

    check(
        f"Fold {fold_id} train ∩ test = empty",
        train_set.isdisjoint(test_set),
    )

    check(
        f"Fold {fold_id} validation ∩ test = empty",
        validation_set.isdisjoint(test_set),
    )

    # -------------------------------------------------------------
    # Test exactly equals held-out year
    # -------------------------------------------------------------

    expected_test_start, expected_test_end = (
        year_ranges[fold["test_year"]]
    )

    check(
        f"Fold {fold_id} test equals "
        f"{fold['test_year']} calendar year",
        (test_start, test_end)
        == (
            expected_test_start,
            expected_test_end,
        ),
    )

    # -------------------------------------------------------------
    # Training + validation + test cover entire dataset
    # -------------------------------------------------------------

    combined = (
        train_set
        | validation_set
        | test_set
    )

    check(
        f"Fold {fold_id} covers all 1419 days",
        len(combined) == EXPECTED_TOTAL_DAYS,
        f"covered={len(combined)}",
    )


# =============================================================================
# [7] Global fold validation
# =============================================================================

print("\n" + "=" * 90)
print("[7] GLOBAL FOLD VALIDATION")
print("=" * 90)

check(
    "exactly 4 folds",
    len(folds) == 4,
)

check(
    "test years are unique",
    len(
        {
            fold["test_year"]
            for fold in folds
        }
    ) == 4,
)

check(
    "test years are 2021-2024",
    [
        fold["test_year"]
        for fold in folds
    ] == TEST_YEARS,
)


# =============================================================================
# [8] Write JSON
# =============================================================================

print("\n" + "=" * 90)
print("[8] WRITE SPLIT CONFIGURATION")
print("=" * 90)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

artifact = {
    "phase": 8,
    "part": 1,
    "artifact": "leave-one-year-out folds",

    "source_tensor": str(SOURCE_TENSOR),

    "split_method": "leave-one-year-out",

    "sampling_cadence": "daily",

    "random_shuffle": False,

    "total_days": EXPECTED_TOTAL_DAYS,

    "validation_policy": {
        "method": (
            "last 30 days of the final "
            "chronological training block"
        ),
        "length_days": VALIDATION_DAYS,
    },

    "folds": folds,

    "argo_policy": {
        "status": "deferred",
        "isolation": "test-window-only",
        "allowed_for_training": False,
        "allowed_for_validation": False,
        "allowed_for_hyperparameter_tuning": False,
        "allowed_for_model_selection": False,
        "purpose": "final independent truth check",
    },
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
# [9] Reload and verify artifact
# =============================================================================

print("\n" + "=" * 90)
print("[9] ARTIFACT VERIFICATION")
print("=" * 90)

with OUTPUT_FILE.open(
    "r",
    encoding="utf-8",
) as f:
    saved = json.load(f)

check(
    "output artifact exists",
    OUTPUT_FILE.exists(),
)

check(
    "saved phase = 8",
    saved["phase"] == 8,
)

check(
    "saved fold count = 4",
    len(saved["folds"]) == 4,
)

check(
    "saved total days = 1419",
    saved["total_days"] == 1419,
)

check(
    "saved sampling cadence = daily",
    saved["sampling_cadence"] == "daily",
)

check(
    "saved random shuffle = false",
    saved["random_shuffle"] is False,
)


# =============================================================================
# Cleanup
# =============================================================================

ds.close()


# =============================================================================
# Final result
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 8 — PART 1 RESULT")
print("=" * 90)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:
    print(
        """
PART 1 LOYO SPLIT CONFIGURATION: PASS

Four chronological leave-one-year-out folds were constructed
from the actual Phase 6 tensor calendar.

No tensor copies were created.
The split configuration contains index ranges only.
"""
    )
else:
    print(
        """
PART 1 LOYO SPLIT CONFIGURATION: FAIL

Do not proceed to the next Phase 8 part.
"""
    )

print("=" * 90)

raise SystemExit(
    0 if failed == 0 else 1
)