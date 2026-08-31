from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = [
    "verify_tensor.py",
    "verify_channels.py",
    "verify_masks.py",
]


def main() -> None:

    print("=" * 100)
    print("OceanEmbed — Phase 5 FINAL VERIFICATION")
    print("=" * 100)

    for script in SCRIPTS:

        path = ROOT / "scripts" / "phase_5" / script

        print()
        print("=" * 100)
        print(f"RUNNING: {script}")
        print("=" * 100)

        result = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
        )

        if result.returncode != 0:
            print()
            print("=" * 100)
            print(f"PHASE 5 VERIFICATION FAILED: {script}")
            print("=" * 100)
            raise SystemExit(result.returncode)

    print()
    print("=" * 100)
    print("PHASE 5 VERIFICATION COMPLETE")
    print("=" * 100)
    print()
    print("All verification gates passed.")
    print()
    print("Phase 5 input tensor is ready for normalization.")
    print("=" * 100)


if __name__ == "__main__":
    main()