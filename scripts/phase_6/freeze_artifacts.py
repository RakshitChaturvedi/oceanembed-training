from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from p6_config import (
    INPUT_TENSOR_PATH,
    NORMALIZATION_STATS_PATH,
    OCEAN_MASK_PATH,
)


# ==============================================================
# PHASE 5 AUXILIARY ARTIFACTS
# ==============================================================

PHASE5_DIR = INPUT_TENSOR_PATH.parent

MISSINGNESS_MASK_PATH = (
    PHASE5_DIR / "missingness_masks.nc"
)

CURL_PATH = (
    PHASE5_DIR / "wind_stress_curl_filled.nc"
)


# ==============================================================
# PHASE 6 ARTIFACTS
# ==============================================================

PHASE6_DIR = Path("data/processed/phase6")

NORMALIZED_TENSOR_PATH = (
    PHASE6_DIR / "input_tensor_normalized.nc"
)

REPORT_PATH = (
    PHASE6_DIR / "phase6_report.txt"
)

MANIFEST_PATH = (
    PHASE6_DIR / "phase6_manifest.json"
)


# ==============================================================
# HELPERS
# ==============================================================

def fail(message: str) -> None:
    raise RuntimeError(
        f"PHASE 6 FREEZE FAILED: {message}"
    )


def sha256_file(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            sha256.update(chunk)

    return sha256.hexdigest()


def file_info(path: Path) -> dict:
    if not path.exists():
        fail(f"Required artifact not found: {path}")

    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


# ==============================================================
# MAIN
# ==============================================================

def main() -> None:

    print("=" * 80)
    print("OceanEmbed — Phase 6 — Freeze Artifacts")
    print("=" * 80)

    # ----------------------------------------------------------
    # Required artifacts
    # ----------------------------------------------------------

    required_artifacts = {
        "input_tensor": INPUT_TENSOR_PATH,
        "normalization_stats": NORMALIZATION_STATS_PATH,
        "normalized_tensor": NORMALIZED_TENSOR_PATH,
        "ocean_mask": OCEAN_MASK_PATH,
        "missingness_masks": MISSINGNESS_MASK_PATH,
        "curl_filled": CURL_PATH,
    }

    print("\nARTIFACT VALIDATION")

    for name, path in required_artifacts.items():

        if not path.exists():
            fail(
                f"Required artifact not found: {path}"
            )

        print(
            f"  {name:<22} : PASS"
        )

    # ----------------------------------------------------------
    # Load normalization statistics
    # ----------------------------------------------------------

    print("\nNORMALIZATION STATISTICS")

    with NORMALIZATION_STATS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        stats = json.load(f)

    if stats.get("phase") != 6:
        fail(
            "normalization_stats.json does not belong to Phase 6"
        )

    if stats.get("normalization_method") != "z-score":
        fail(
            "Expected z-score normalization"
        )

    print("  phase                : 6")
    print("  method               : z-score")
    print("  formula              : (x - mean) / std")

    # ----------------------------------------------------------
    # Verify expected channel contract
    # ----------------------------------------------------------

    expected_normalized = list(range(10))
    expected_unchanged = [10, 11]

    actual_normalized = stats.get(
        "normalized_channel_indices"
    )

    actual_unchanged = stats.get(
        "unchanged_channel_indices"
    )

    if actual_normalized != expected_normalized:
        fail(
            "Normalized channel contract does not match "
            "expected channels 0-9"
        )

    if actual_unchanged != expected_unchanged:
        fail(
            "Unchanged channel contract does not match "
            "expected channels 10-11"
        )

    print(
        "  normalized channels  : 0-9"
    )

    print(
        "  unchanged channels   : 10-11"
    )

    # ----------------------------------------------------------
    # Training window
    # ----------------------------------------------------------

    training_window = stats.get(
        "training_window"
    )

    if not training_window:
        fail(
            "training_window missing from statistics"
        )

    print("\nTRAINING WINDOW")

    print(
        f"  start index          : "
        f"{training_window['start_index']}"
    )

    print(
        f"  end index exclusive  : "
        f"{training_window['end_index_exclusive']}"
    )

    print(
        f"  sample count         : "
        f"{training_window['sample_count']}"
    )

    print(
        f"  start time           : "
        f"{training_window['start_time']}"
    )

    print(
        f"  end time             : "
        f"{training_window['end_time']}"
    )

    # ----------------------------------------------------------
    # Artifact checksums
    # ----------------------------------------------------------

    print("\nCOMPUTING SHA-256 CHECKSUMS")

    artifact_manifest = {}

    for name, path in required_artifacts.items():

        info = file_info(path)

        artifact_manifest[name] = info

        print(
            f"  {name:<22} : {info['sha256']}"
        )

    # ----------------------------------------------------------
    # Freeze metadata
    # ----------------------------------------------------------

    frozen_at = datetime.now(
        timezone.utc
    ).isoformat()

    manifest = {
        "project": "OceanEmbed",
        "phase": 6,
        "status": "FROZEN",
        "frozen_at_utc": frozen_at,

        "normalization": {
            "method": stats[
                "normalization_method"
            ],
            "formula": stats[
                "formula"
            ],
            "normalized_channel_indices": (
                expected_normalized
            ),
            "unchanged_channel_indices": (
                expected_unchanged
            ),
        },

        "training_window": training_window,

        "artifacts": artifact_manifest,

        "land_cell_contract": {
            "value": 0.0,
            "channels": "0-9",
        },

        "permanent_invalid_cell_contract": {
            "value": 0.0,
            "channels": "0-9",
        },

        "channels_10_11_contract": {
            "normalization": "NONE",
            "channels": {
                "10": "sin_day_of_year",
                "11": "cos_day_of_year",
            },
        },

        "status_statement": (
            "Phase 6 normalization artifacts are frozen. "
            "Channels 0-9 use training-only z-score "
            "statistics. Land and permanently invalid "
            "cells remain exactly 0.0. Channels 10-11 "
            "remain unchanged."
        ),
    }

    # ----------------------------------------------------------
    # Write manifest
    # ----------------------------------------------------------

    PHASE6_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nWRITING MANIFEST")

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )
        f.write("\n")

    print(
        f"  manifest : {MANIFEST_PATH}"
    )

    # ----------------------------------------------------------
    # Write human-readable report
    # ----------------------------------------------------------

    report_lines = [
        "=" * 80,
        "OceanEmbed — PHASE 6 FREEZE REPORT",
        "=" * 80,
        "",
        "STATUS",
        "  FROZEN",
        "",
        f"Frozen at UTC : {frozen_at}",
        "",
        "NORMALIZATION",
        "  method   : z-score",
        "  formula  : (x - mean) / std",
        "  channels : 0-9",
        "",
        "UNCHANGED CHANNELS",
        "  [10] sin_day_of_year",
        "  [11] cos_day_of_year",
        "",
        "TRAINING WINDOW",
        f"  start index         : "
        f"{training_window['start_index']}",
        f"  end index exclusive : "
        f"{training_window['end_index_exclusive']}",
        f"  sample count        : "
        f"{training_window['sample_count']}",
        f"  start time          : "
        f"{training_window['start_time']}",
        f"  end time            : "
        f"{training_window['end_time']}",
        "",
        "PRESERVATION CONTRACT",
        "  land cells                  : exactly 0.0",
        "  permanently invalid cells  : exactly 0.0",
        "  channels 10-11             : unchanged",
        "",
        "FROZEN ARTIFACTS",
    ]

    for name, info in artifact_manifest.items():

        report_lines.extend(
            [
                "",
                f"  {name}",
                f"    path     : {info['path']}",
                f"    size     : {info['size_bytes']} bytes",
                f"    sha256   : {info['sha256']}",
            ]
        )

    report_lines.extend(
        [
            "",
            "FINAL STATEMENT",
            "  Phase 6 Part 4 verification passed.",
            "  Phase 6 normalization artifacts are now frozen.",
            "",
            "=" * 80,
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print(
        f"  report   : {REPORT_PATH}"
    )

    # ----------------------------------------------------------
    # Final output
    # ----------------------------------------------------------

    print("\n" + "=" * 80)
    print("RESULT: PASS")
    print("=" * 80)

    print(
        "\nPhase 6 artifacts are now FROZEN."
    )

    print(
        f"  manifest : {MANIFEST_PATH}"
    )

    print(
        f"  report   : {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()