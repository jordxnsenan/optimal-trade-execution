import pandas_market_calendars as market_calendars

from datetime import date, timedelta
from execai.data.alpaca import fetch_spy_bars
from execai.data.session_validation import validate_trading_session


def download_spy_sessions(
    start_date: date,
    end_date: date,
    api_key: str,
    secret_key: str,
) -> list[dict]:
    """downloads and validates SPY bars for a range of trading days"""

    if start_date > end_date:
        raise ValueError("Start date must not be after end date.")

    if not api_key or not secret_key:
        raise ValueError("Alpaca API credentials are required.")

    nyse_calendar = market_calendars.get_calendar("NYSE")

    schedule = nyse_calendar.schedule(
        start_date=start_date,
        end_date=end_date,
    )

    if schedule.empty:
        raise ValueError("No trading sessions found in the requested date range.")

    all_bars = []

    for trading_day, market_hours in schedule.iterrows():
        session_start = market_hours["market_open"].to_pydatetime()

        session_end = market_hours["market_close"].to_pydatetime() - timedelta(
            seconds=1
        )

        bars = fetch_spy_bars(
            session_start,
            session_end,
            api_key,
            secret_key,
        )

        validate_trading_session(bars, trading_day.date())

        all_bars.extend(bars)

    return all_bars
