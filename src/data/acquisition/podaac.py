from __future__ import annotations

import re
import subprocess
from datetime import date, timedelta
from pathlib import Path


DATE_RE = re.compile(r"(20\d{6})")


def extract_date(path: Path) -> date | None:
    """
    Extract YYYYMMDD from a PO.DAAC filename.
    """

    match = DATE_RE.search(path.name)

    if not match:
        return None

    try:
        return date(
            int(match.group(1)[0:4]),
            int(match.group(1)[4:6]),
            int(match.group(1)[6:8]),
        )
    except ValueError:
        return None


def existing_dates(output_dir: Path) -> set[date]:
    """
    Find dates already downloaded.
    """

    dates = set()

    for path in output_dir.glob("*.nc"):
        d = extract_date(path)

        if d is not None:
            dates.add(d)

    return dates


def missing_ranges(
    start: date,
    end: date,
    existing: set[date],
) -> list[tuple[date, date]]:
    """
    Convert missing dates into contiguous ranges.
    """

    missing = []

    current = start

    while current <= end:

        if current not in existing:
            missing.append(current)

        current += timedelta(days=1)

    if not missing:
        return []

    ranges = []

    range_start = missing[0]
    previous = missing[0]

    for current in missing[1:]:

        if current == previous + timedelta(days=1):
            previous = current
        else:
            ranges.append((range_start, previous))
            range_start = current
            previous = current

    ranges.append((range_start, previous))

    return ranges


def download_collection(
    *,
    collection: str,
    output_dir: str,
    start_date: str,
    end_date: str,
    bbox: dict,
) -> None:

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    existing = existing_dates(output)

    print(f"\n=== PO.DAAC: {collection} ===")
    print(f"Existing files: {len(existing)}")

    ranges = missing_ranges(
        start,
        end,
        existing,
    )

    if not ranges:
        print("Everything already downloaded.")
        return

    print("Missing ranges:")

    for rstart, rend in ranges:
        print(f"  {rstart} -> {rend}")

    for rstart, rend in ranges:

        command = [
            "podaac-data-downloader",
            "-c",
            collection,
            "-d",
            str(output),
            "--start-date",
            f"{rstart}T00:00:00Z",
            "--end-date",
            f"{rend}T23:59:59Z",
            (
                f"-b={bbox['lon_min']},{bbox['lat_min']},"
                f"{bbox['lon_max']},{bbox['lat_max']}"
            ),
        ]

        print("\n$", " ".join(command))

        subprocess.run(
            command,
            check=True,
        )