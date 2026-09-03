from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


# =============================================================================
# Paths
# =============================================================================

MAIN_GLORYS = Path("data/processed/phase2/glorys.nc")
DEEP_GLORYS = Path("data/processed/phase2/glorys_1062.nc")

OUTPUT = Path("data/processed/phase7/depth_mapping.json")


# =============================================================================
# Requested model depths
# =============================================================================

TARGET_DEPTHS = [
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700,
    1000,
]


# =============================================================================
# Tolerances
# =============================================================================
# The deep artifact is explicitly the 1062.44 m source level.
DEEP_EXPECTED_DEPTH = 1062.44
DEEP_TOLERANCE = 0.01


# =============================================================================
# Helpers
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


def nearest_depth(
    available_depths: np.ndarray,
    requested_depth: float,
) -> tuple[float, float]:
    """
    Return (nearest_source_depth, absolute_difference).
    """
    index = int(
        np.argmin(np.abs(available_depths - requested_depth))
    )

    source_depth = float(available_depths[index])
    difference = abs(source_depth - requested_depth)

    return source_depth, difference


# =============================================================================
# Header
# =============================================================================

print("=" * 90)
print("OceanEmbed — Phase 7 Depth Mapping")
print("=" * 90)

print(
    """
Purpose:
  Establish the mapping between the 15 model target depths and the
  available GLORYS source depths.

Rules:
  - First 14 target depths use nearest available GLORYS native level.
  - 1000 m target uses the separately acquired 1062.44 m GLORYS level.
  - No temperature/salinity data is loaded.
  - Only depth coordinates are inspected.
"""
)


# =============================================================================
# Input checks
# =============================================================================

print("=" * 90)
print("[1] INPUT ARTIFACTS")
print("=" * 90)

check(
    "MAIN GLORYS exists",
    MAIN_GLORYS.exists(),
    str(MAIN_GLORYS),
)

check(
    "DEEP GLORYS exists",
    DEEP_GLORYS.exists(),
    str(DEEP_GLORYS),
)

if failed:
    print("\nCannot continue.")
    raise SystemExit(1)


# =============================================================================
# Open datasets lazily
# =============================================================================

main = xr.open_dataset(MAIN_GLORYS)
deep = xr.open_dataset(DEEP_GLORYS)


# =============================================================================
# Validate depth coordinates
# =============================================================================

print("\n" + "=" * 90)
print("[2] MAIN GLORYS DEPTHS")
print("=" * 90)

check(
    "MAIN depth coordinate exists",
    "depth" in main.coords,
)

if "depth" not in main.coords:
    main.close()
    deep.close()
    raise SystemExit(1)

main_depths = np.asarray(main["depth"].values, dtype=float)

print(f"Number of source levels: {len(main_depths)}")
print("Available depths:")
print(main_depths)


print("\n" + "=" * 90)
print("[3] DEEP GLORYS DEPTH")
print("=" * 90)

check(
    "DEEP depth coordinate exists",
    "depth" in deep.coords,
)

check(
    "DEEP contains exactly one depth",
    deep.sizes.get("depth") == 1,
    f"actual={deep.sizes.get('depth')}",
)

if "depth" not in deep.coords or deep.sizes.get("depth") != 1:
    main.close()
    deep.close()
    raise SystemExit(1)

deep_depth = float(deep["depth"].values[0])

print(f"Deep source depth: {deep_depth:.8f} m")

check(
    "DEEP depth is 1062.44 m",
    np.isclose(
        deep_depth,
        DEEP_EXPECTED_DEPTH,
        atol=DEEP_TOLERANCE,
    ),
    (
        f"expected≈{DEEP_EXPECTED_DEPTH}, "
        f"actual={deep_depth}"
    ),
)


# =============================================================================
# Construct mapping
# =============================================================================

print("\n" + "=" * 90)
print("[4] TARGET DEPTH MAPPING")
print("=" * 90)

mapping = []

print(
    f"{'TARGET':>10} {'SOURCE':>15} {'ERROR':>15} {'SOURCE FILE':>20}"
)
print("-" * 65)


for target_depth in TARGET_DEPTHS:

    # -------------------------------------------------------------------------
    # Special 1000 m target
    # -------------------------------------------------------------------------

    if target_depth == 1000:

        source_depth = deep_depth
        difference = abs(source_depth - target_depth)

        source_file = str(DEEP_GLORYS)
        source_type = "deep_glorys"

        print(
            f"{target_depth:>10.1f} "
            f"{source_depth:>15.6f} "
            f"{difference:>15.6f} "
            f"{source_type:>20}"
        )

        check(
            "1000 m target uses deep GLORYS artifact",
            np.isclose(
                source_depth,
                DEEP_EXPECTED_DEPTH,
                atol=DEEP_TOLERANCE,
            ),
        )

    # -------------------------------------------------------------------------
    # First 14 target depths
    # -------------------------------------------------------------------------

    else:

        source_depth, difference = nearest_depth(
            main_depths,
            target_depth,
        )

        source_file = str(MAIN_GLORYS)
        source_type = "main_glorys"

        print(
            f"{target_depth:>10.1f} "
            f"{source_depth:>15.6f} "
            f"{difference:>15.6f} "
            f"{source_type:>20}"
        )

        check(
            f"{target_depth} m has valid nearest GLORYS level",
            source_depth in main_depths,
        )

        if difference > 50.0:
            print(
                f"WARN {target_depth}m vertical mismatch is large, ({difference:.6f}m)"
            )

    mapping.append(
        {
            "target_depth_m": int(target_depth),
            "source_depth_m": source_depth,
            "absolute_depth_difference_m": difference,
            "source_file": source_file,
            "source_type": source_type,
        }
    )


# =============================================================================
# Mapping order validation
# =============================================================================

print("\n" + "=" * 90)
print("[5] MAPPING ORDER")
print("=" * 90)

mapped_targets = [
    item["target_depth_m"]
    for item in mapping
]

check(
    "15 target depths present",
    len(mapping) == 15,
    f"actual={len(mapping)}",
)

check(
    "target depths exactly match model contract",
    mapped_targets == TARGET_DEPTHS,
    f"actual={mapped_targets}",
)

check(
    "target depths strictly increasing",
    all(
        b > a
        for a, b in zip(mapped_targets, mapped_targets[1:])
    ),
)


# =============================================================================
# Explicit 1000 m validation
# =============================================================================

print("\n" + "=" * 90)
print("[6] 1000 m SPECIAL CASE")
print("=" * 90)

deep_mapping = mapping[-1]

print("Model target:")
print(f"  {deep_mapping['target_depth_m']} m")

print("GLORYS source:")
print(f"  {deep_mapping['source_depth_m']:.8f} m")

print("Source artifact:")
print(f"  {deep_mapping['source_file']}")

check(
    "1000 m maps to 1062.44 m",
    np.isclose(
        deep_mapping["source_depth_m"],
        DEEP_EXPECTED_DEPTH,
        atol=DEEP_TOLERANCE,
    ),
)

check(
    "1000 m uses deep_glorys source",
    deep_mapping["source_type"] == "deep_glorys",
)


# =============================================================================
# Write JSON artifact
# =============================================================================

print("\n" + "=" * 90)
print("[7] WRITE DEPTH MAPPING")
print("=" * 90)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

artifact = {
    "phase": 7,
    "artifact": "depth_mapping",
    "target_depths_m": TARGET_DEPTHS,
    "mapping": mapping,
    "special_cases": {
        "1000m": {
            "target_depth_m": 1000,
            "source_depth_m": deep_depth,
            "source_file": str(DEEP_GLORYS),
            "reason": (
                "No suitable 1000 m level exists in the main GLORYS "
                "artifact; separately acquired GLORYS level at 1062.44 m "
                "is used as the 1000 m model target."
            ),
        }
    },
}

with OUTPUT.open("w", encoding="utf-8") as f:
    json.dump(artifact, f, indent=2)

print(f"Output: {OUTPUT}")

check(
    "depth mapping JSON written",
    OUTPUT.exists(),
)


# =============================================================================
# Cleanup
# =============================================================================

main.close()
deep.close()


# =============================================================================
# Final result
# =============================================================================

print("\n" + "=" * 90)
print("PHASE 7 — PART 2 RESULT")
print("=" * 90)

print(f"PASS: {passed}")
print(f"FAIL: {failed}")

if failed == 0:
    print(
        "\nPART 2 DEPTH MAPPING: PASS\n"
        "The 15 model target depths have been formally mapped "
        "to GLORYS source levels."
    )
else:
    print(
        "\nPART 2 DEPTH MAPPING: FAIL\n"
        "Do not proceed to target extraction."
    )

print("=" * 90)

raise SystemExit(0 if failed == 0 else 1)
