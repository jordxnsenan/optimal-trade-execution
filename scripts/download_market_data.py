import argparse
import os

from datetime import date
from pathlib import Path
from dotenv import load_dotenv

from execai.data.ingestion import download_spy_sessions
from execai.data.storage import save_bars_to_csv

RAW_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    parser = argparse.ArgumentParser(
        description="Download historical five-minute SPY market data."
    )

    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="First trading date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="Last trading date in YYYY-MM-DD format.",
    )

    arguments = parser.parse_args()

    if arguments.start > arguments.end:
        parser.error("Start date must not be after end date.")

    load_dotenv()

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        parser.error("Missing Alpaca credentials. Check your .env file.")

    filename = f"SPY_5min_SIP_raw_{arguments.start}_{arguments.end}.csv"

    output_path = RAW_DATA_DIRECTORY / filename

    if output_path.exists():
        parser.exit(1, f"File already exists: {output_path}\n")

    print(f"Downloading SPY data from {arguments.start} to {arguments.end}...")

    try:
        bars = download_spy_sessions(
            arguments.start,
            arguments.end,
            api_key,
            secret_key,
        )

        save_bars_to_csv(bars, output_path)

    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Download failed: {error}\n")

    print(f"Successfully downloaded {len(bars)} market bars.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()