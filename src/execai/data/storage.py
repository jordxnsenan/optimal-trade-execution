import csv

from pathlib import Path
from execai.data.validation import validate_bars

CSV_FIELDS = ("t", "o", "h", "l", "c", "v")


def save_bars_to_csv(bars: list[dict], output_path: Path) -> None:
    """saves validated market bars to CSV file"""

    validate_bars(bars)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)

        writer.writeheader()

        for bar in bars:
            row = {field: bar[field] for field in CSV_FIELDS}

            writer.writerow(row)