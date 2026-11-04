import argparse
import json
import os

from datetime import date
from dotenv import load_dotenv
from pathlib import Path

from execai.data.ingestion import iter_spy_sessions
from execai.data.storage import (
    find_completed_sessions,
    save_session_checkpoint,
)

CHECKPOINT_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "SPY_5min_SIP"
)


def download_and_save_sessions(
    start_date: date,
    end_date: date,
    api_key: str,
    secret_key: str,
    checkpoint_directory: Path,
) -> dict:
    """downloads trading sessions and save each completed session"""

    if start_date > end_date:
        raise ValueError("Start date must not be after end date.")

    completed_dates = find_completed_sessions(
        checkpoint_directory,
        start_date,
        end_date,
    )

    downloaded_dates = []
    rejected_sessions = {}
    download_error = None

    print(f"Found {len(completed_dates)} existing valid checkpoints.")

    try:
        for result in iter_spy_sessions(
            start_date,
            end_date,
            api_key,
            secret_key,
            skip_dates=completed_dates,
        ):
            trading_date = result.trading_date

            if result.validation_error is not None:
                rejected_sessions[str(trading_date)] = result.validation_error
                print(f"Rejected {trading_date}: {result.validation_error}")
                continue

            save_session_checkpoint(
                result.bars,
                trading_date,
                checkpoint_directory,
            )

            downloaded_dates.append(str(trading_date))

            print(f"Saved {trading_date}: " f"{len(result.bars)} bars")

    except (RuntimeError, ValueError, OSError) as error:
        download_error = str(error)

    report = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "provider": "Alpaca",
        "symbol": "SPY",
        "timeframe": "5Min",
        "feed": "sip",
        "adjustment": "raw",
        "existing_sessions": sorted(
            str(trading_date) for trading_date in completed_dates
        ),
        "downloaded_sessions": downloaded_dates,
        "rejected_sessions": rejected_sessions,
        "download_error": download_error,
    }

    report_directory = checkpoint_directory / "reports"
    report_directory.mkdir(parents=True, exist_ok=True)

    report_path = report_directory / f"{start_date}_{end_date}.json"

    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nDownload Summary")
    print("----------------")
    print(f"Previously completed: {len(completed_dates)}")
    print(f"Newly downloaded: {len(downloaded_dates)}")
    print(f"Rejected sessions: {len(rejected_sessions)}")
    print(f"Report saved to: {report_path}")

    if download_error is not None:
        raise RuntimeError(f"Download stopped: {download_error}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Download historical SPY market data.")

    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="First date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        required=True,
        help="Last date in YYYY-MM-DD format.",
    )

    arguments = parser.parse_args()

    if arguments.start > arguments.end:
        parser.error("Start date must not be after end date.")

    load_dotenv()

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        parser.error("Missing Alpaca credentials. Check your .env file.")

    print(f"Downloading SPY data from " f"{arguments.start} to {arguments.end}...")

    try:
        download_and_save_sessions(
            arguments.start,
            arguments.end,
            api_key,
            secret_key,
            CHECKPOINT_DIRECTORY,
        )
    except (RuntimeError, ValueError, OSError) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
