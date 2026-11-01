from execai.data.validation import validate_bars
from datetime import date, datetime, timedelta, timezone

import pandas_market_calendars as market_calendars


BAR_INTERVAL_MINUTES = 5


def validate_trading_session(
    bars: list[dict],
    trading_date: date,
) -> None:
    """checks that all expected five-minute trading bars are present"""

    nyse_calendar = market_calendars.get_calendar("NYSE")

    schedule = nyse_calendar.schedule(
        start_date=trading_date,
        end_date=trading_date,
    )

    # ensures that closed trading days dont contain regular-session bars
    if schedule.empty:
        if bars:
            raise ValueError("Market bars found on a closed trading day.")

        return

    validate_bars(bars)

    session_open = schedule.iloc[0]["market_open"].to_pydatetime()
    session_close = schedule.iloc[0]["market_close"].to_pydatetime()

    expected_timestamps = set()
    current_time = session_open

    while current_time < session_close:
        expected_timestamps.add(current_time)
        current_time += timedelta(minutes=BAR_INTERVAL_MINUTES)

    actual_timestamps = set()

    for bar in bars:
        timestamp = datetime.fromisoformat(
            bar["t"].replace("Z", "+00:00")
        ).astimezone(timezone.utc)

        actual_timestamps.add(timestamp)

    missing_timestamps = expected_timestamps - actual_timestamps
    unexpected_timestamps = actual_timestamps - expected_timestamps

    if missing_timestamps:
        first_missing = min(missing_timestamps)

        raise ValueError(
            f"Missing five-minute market bar at {first_missing}."
        )

    if unexpected_timestamps:
        first_unexpected = min(unexpected_timestamps)

        raise ValueError(
            f"Unexpected market bar at {first_unexpected}."
        )