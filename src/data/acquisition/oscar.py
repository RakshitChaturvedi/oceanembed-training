from __future__ import annotations

import re
import subprocess
from pathlib import Path
from datetime import date, timedelta

DATE_RE = re.compile(r"(\d{8})")
def existing_dates(output_dir: Path) -> set[date]:
    """
    Extract dates from existing OSCAR NetCDF filenames.
    """
    dates: set[date] = set()

    for path in output_dir.glob("*.nc"):
        match = DATE_RE.search(path.name)

        if not match:
            continue

        try:
            dates.add(
                date.fromisoformat(
                    f"{match.group(1)[:4]}-"
                    f"{match.group(1)[4:6]}-"
                    f"{match.group(1)[6:]}"
                )
            )
        except ValueError:
            pass

    return dates

def missing_ranges(
    start: date,
    end: date,
    existing: set[date],
) -> list[tuple[date, date]]:
    """
    Convert missing individual dates into contiguous ranges.
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
            continue

        ranges.append((range_start, previous))

        range_start = current
        previous = current

    ranges.append((range_start, previous))

    return ranges

def download_oscar_range(
    *,
    collection: str,
    output_dir: Path,
    start_date: date,
    end_date: date,
    bbox: dict,
) -> None:

    command = [
        "podaac-data-downloader",
        "-c",
        collection,
        "-d",
        str(output_dir),
        "--start-date",
        f"{start_date}T00:00:00Z",
        "--end-date",
        f"{end_date}T23:59:59Z",
        (
            f"-b={bbox['lon_min']},{bbox['lat_min']},"
            f"{bbox['lon_max']},{bbox['lat_max']}"
        ),
    ]

    print(
        f"\nDownloading OSCAR: "
        f"{start_date} -> {end_date}"
    )

    subprocess.run(command, check=True)


def acquire_oscar(cfg: dict) -> None:

    source = cfg["oscar"]

    output_dir = Path(source["output_dir"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    configured_start = date.fromisoformat(
        cfg["dates"]["start"]
    )

    configured_end = date.fromisoformat(
        cfg["dates"]["end"]
    )

    existing = existing_dates(output_dir)

    print("\n=== OSCAR SURFACE CURRENTS ===")
    print(f"Output: {output_dir}")
    print(f"Existing files/dates detected: {len(existing)}")

    ranges = missing_ranges(
        configured_start,
        configured_end,
        existing,
    )

    if not ranges:
        print("OSCAR: all requested dates already exist.")
        return

    print("Missing ranges:")

    for start, end in ranges:
        print(f"  {start} -> {end}")

    for start, end in ranges:
        download_oscar_range(
            collection=source["collection"],
            output_dir=output_dir,
            start_date=start,
            end_date=end,
            bbox=cfg["domain"],
        )