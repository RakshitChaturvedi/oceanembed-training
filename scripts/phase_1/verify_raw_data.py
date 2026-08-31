from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
import re
import statistics


ROOT = Path("data/raw")

START = date(2021, 1, 1)
END = date(2024, 12, 31)

DATASETS = {
    "sst": {
        "pattern": r"(\d{8})",
        "expected_daily": True,
    },
    "wind": {
        "pattern": r"(\d{8})",
        "expected_daily": True,
    },
    "currents": {
        "pattern": r"(\d{8})",
        "expected_daily": True,
    },
    "sss": {
        "pattern": r"(\d{4}-\d{2}-\d{2})",
        "expected_daily": False,
    },
}


def extract_date(filename: str, pattern: str) -> date | None:
    match = re.search(pattern, filename)

    if not match:
        return None

    value = match.group(1)

    try:
        if len(value) == 8:
            return date(
                int(value[:4]),
                int(value[4:6]),
                int(value[6:]),
            )

        return date.fromisoformat(value)

    except ValueError:
        return None


def expected_dates() -> set[date]:
    dates = set()
    current = START

    while current <= END:
        dates.add(current)
        current += timedelta(days=1)

    return dates


def size_outliers(files: list[Path]) -> list[tuple[Path, float, float]]:
    if len(files) < 5:
        return []

    sizes = [f.stat().st_size for f in files]

    median = statistics.median(sizes)

    # Robust threshold:
    # anything <50% or >150% of median
    lower = median * 0.50
    upper = median * 1.50

    outliers = []

    for file, size in zip(files, sizes):
        if size < lower or size > upper:
            ratio = size / median
            outliers.append((file, size, ratio))

    return sorted(outliers, key=lambda x: x[1])


def human_size(size: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    for unit in units:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size:.2f} PB"


def check_dataset(name: str, config: dict) -> None:
    directory = ROOT / name

    print("\n" + "=" * 70)
    print(f"{name.upper()}")
    print("=" * 70)

    if not directory.exists():
        print("DIRECTORY MISSING")
        return

    files = sorted(directory.glob("*.nc"))

    print(f"Files found : {len(files)}")

    # ---------------------------------------------------------
    # Missing dates
    # ---------------------------------------------------------

    if config["expected_daily"]:
        expected = expected_dates()
        found = set()

        for file in files:
            d = extract_date(file.name, config["pattern"])

            if d is not None:
                found.add(d)

        missing = sorted(expected - found)
        unexpected = sorted(found - expected)

        print(f"Expected    : {len(expected)} daily files")
        print(f"Detected    : {len(found)} dates")
        print(f"Missing     : {len(missing)}")
        print(f"Unexpected  : {len(unexpected)}")

        if missing:
            print("\nMISSING DATES:")
            for d in missing:
                print(f"  {d}")

        if unexpected:
            print("\nDATES OUTSIDE REQUESTED RANGE:")
            for d in unexpected:
                print(f"  {d}")

    # ---------------------------------------------------------
    # File sizes
    # ---------------------------------------------------------

    if files:
        sizes = [f.stat().st_size for f in files]

        print("\nSIZE STATISTICS:")
        print(f"  Minimum : {human_size(min(sizes))}")
        print(f"  Median  : {human_size(statistics.median(sizes))}")
        print(f"  Maximum : {human_size(max(sizes))}")
        print(f"  Total   : {human_size(sum(sizes))}")

        outliers = size_outliers(files)

        print(f"\nSIZE OUTLIERS: {len(outliers)}")

        if outliers:
            for file, size, ratio in outliers:
                print(
                    f"  {file.name}"
                    f"  | {human_size(size)}"
                    f"  | {ratio:.2f}x median"
                )
        else:
            print("  None")


def main() -> None:
    print("=== OceanEmbed Raw Data Verification ===")
    print(f"Range: {START} -> {END}")

    for name, config in DATASETS.items():
        check_dataset(name, config)

    print("\n" + "=" * 70)
    print("Verification complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()