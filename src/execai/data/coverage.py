import pandas_market_calendars as market_calendars

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from execai.data.session_validation import BAR_INTERVAL_MINUTES
from execai.data.storage import find_completed_sessions


@dataclass(frozen=True)
class DataCoverageReport:
    """summarizes historical market data coverage"""

    expected_sessions: int
    completed_sessions: int
    expected_bars: int
    completed_bars: int
    missing_dates: list[date]

    @property
    def coverage_percent(self) -> float:
        return self.completed_sessions / self.expected_sessions * 100


def audit_market_data(
    checkpoint_directory: Path,
    start_date: date,
    end_date: date,
) -> DataCoverageReport:
    """checks saved market data against NYSE trading calendar"""

    if start_date > end_date:
        raise ValueError("Start date must not be after end date.")

    nyse_calendar = market_calendars.get_calendar("NYSE")

    schedule = nyse_calendar.schedule(
        start_date=start_date,
        end_date=end_date,
    )

    if schedule.empty:
        raise ValueError("No trading sessions found in the date range.")

    completed_dates = find_completed_sessions(
        checkpoint_directory,
        start_date,
        end_date,
    )

    expected_bars = 0
    completed_bars = 0
    missing_dates = []

    bar_duration = timedelta(minutes=BAR_INTERVAL_MINUTES)

    for trading_day, market_hours in schedule.iterrows():
        trading_date = trading_day.date()

        session_duration = market_hours["market_close"] - market_hours["market_open"]

        bars_in_session = int(
            session_duration.total_seconds() / bar_duration.total_seconds()
        )

        expected_bars += bars_in_session

        if trading_date in completed_dates:
            completed_bars += bars_in_session
        else:
            missing_dates.append(trading_date)

    return DataCoverageReport(
        expected_sessions=len(schedule),
        completed_sessions=len(completed_dates),
        expected_bars=expected_bars,
        completed_bars=completed_bars,
        missing_dates=missing_dates,
    )