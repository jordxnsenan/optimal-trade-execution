from datetime import date
from pathlib import Path

from execai.data.session_validation import validate_trading_session
from execai.data.storage import load_bars_from_csv
from execai.data.splits import DEFAULT_SPLITS

import pandas_market_calendars as market_calendars


def load_spy_sessions(
    checkpoint_directory: Path,
    start_date: date,
    end_date: date,
) -> dict[date, list[dict]]:
    """loads complete historical SPY trading sessions from local CSV files"""

    if start_date > end_date:
        raise ValueError("Start date must not be after end date.")

    nyse_calendar = market_calendars.get_calendar("NYSE")

    schedule = nyse_calendar.schedule(
        start_date=start_date,
        end_date=end_date,
    )

    if schedule.empty:
        raise ValueError("No trading sessions found in the requested date range.")

    sessions = {}

    for trading_day in schedule.index:
        trading_date = trading_day.date()
        checkpoint_path = checkpoint_directory / f"{trading_date}.csv"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Missing market data checkpoint for {trading_date}."
            )

        bars = load_bars_from_csv(checkpoint_path)
        validate_trading_session(bars, trading_date)

        sessions[trading_date] = bars

    return sessions


def load_dataset_split(
    checkpoint_directory: Path,
    split_name: str,
) -> dict[date, list[dict]]:
    """loads historical market data for a predefined dataset split"""

    date_range = DEFAULT_SPLITS.get_range(split_name)

    return load_spy_sessions(
        checkpoint_directory,
        date_range.start_date,
        date_range.end_date,
    )