import csv

from datetime import date
from pathlib import Path

from execai.data.session_validation import validate_trading_session
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


def load_bars_from_csv(input_path: Path) -> list[dict]:
    """loads market bars from saved CSV file"""

    with input_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError("CSV file has unexpected columns.")

        try:
            bars = [
                {
                    "t": row["t"],
                    "o": float(row["o"]),
                    "h": float(row["h"]),
                    "l": float(row["l"]),
                    "c": float(row["c"]),
                    "v": int(row["v"]),
                }
                for row in reader
            ]
        except (TypeError, ValueError) as error:
            raise ValueError("CSV file contains invalid market data.") from error

    validate_bars(bars)

    return bars


def find_completed_sessions(
    checkpoint_directory: Path,
    start_date: date,
    end_date: date,
) -> set[date]:
    """finds previously saved trading sessions that pass validation"""

    completed_dates = set()

    if not checkpoint_directory.exists():
        return completed_dates

    for csv_path in sorted(checkpoint_directory.glob("*.csv")):
        try:
            trading_date = date.fromisoformat(csv_path.stem)
        except ValueError as error:
            raise ValueError(
                f"Unexpected checkpoint filename: {csv_path.name}"
            ) from error

        if not start_date <= trading_date <= end_date:
            continue

        try:
            bars = load_bars_from_csv(csv_path)
            validate_trading_session(bars, trading_date)
        except (ValueError, OSError) as error:
            raise ValueError(
                f"Invalid checkpoint for {trading_date}: {error}"
            ) from error

        completed_dates.add(trading_date)

    return completed_dates