from pathlib import Path
import sys


# --------------------------------------------------------------
# Make project src/ importable when running the script directly.
# --------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src"),
)


from src.data.wind_stress_curl.wind_stress_curl import ( 
    process_wind_stress_curl,
)


# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------

WIND_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase2"
    / "wind.nc"
)

SST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase2"
    / "sst.nc"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "phase4"
    / "wind_stress_curl.nc"
)


def main() -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("OCEANEMBED PHASE 4 — WIND STRESS CURL")
    print("=" * 80)

    print()
    print(f"Wind : {WIND_PATH}")
    print(f"SST  : {SST_PATH}")
    print(f"Out  : {OUTPUT_PATH}")
    print()

    if not WIND_PATH.exists():
        raise FileNotFoundError(
            f"Wind file not found: {WIND_PATH}"
        )

    if not SST_PATH.exists():
        raise FileNotFoundError(
            f"SST file not found: {SST_PATH}"
        )

    process_wind_stress_curl(
        wind_path=str(WIND_PATH),
        sst_path=str(SST_PATH),
        output_path=str(OUTPUT_PATH),
    )

    print()
    print("=" * 80)
    print("PHASE 4 COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()