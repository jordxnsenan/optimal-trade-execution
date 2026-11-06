import pandas_market_calendars as market_calendars

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta


from execai.data.alpaca import fetch_spy_bars
from execai.data.session_validation import validate_trading_session


@dataclass
class SessionDownloadReport:
    """stores downloaded market data and session validation results"""

    bars: list[dict]
    completed_sessions: list[date]
    rejected_sessions: dict[date, str]


@dataclass
class TradingSessionResult:
    """stores the result of downloading one trading session"""

    trading_date: date
    bars: list[dict]
    validation_error: str | None = None


def iter_spy_sessions(
    start_date: date,
    end_date: date,
    api_key: str,
    secret_key: str,
    skip_dates: set[date] | None = None,
) -> Iterator[TradingSessionResult]:
    """downlaods and validates one trading session at a time"""

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

    for trading_day, market_hours in schedule.iterrows():
        trading_date = trading_day.date()

        if skip_dates is not None and trading_date in skip_dates:
            continue

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

        try:
            validate_trading_session(bars, trading_date)
        except ValueError as error:
            yield TradingSessionResult(
                trading_date=trading_date,
                bars=[],
                validation_error=str(error),
            )
            continue

        yield TradingSessionResult(
            trading_date=trading_date,
            bars=bars,
        )


def download_spy_sessions_with_report(
    start_date: date,
    end_date: date,
    api_key: str,
    secret_key: str,
) -> SessionDownloadReport:
    """downloads sessions and collects their validation results"""

    report = SessionDownloadReport(
        bars=[],
        completed_sessions=[],
        rejected_sessions={},
    )

    for result in iter_spy_sessions(
        start_date,
        end_date,
        api_key,
        secret_key,
    ):
        if result.validation_error is not None:
            report.rejected_sessions[result.trading_date] = result.validation_error
            continue

        report.bars.extend(result.bars)
        report.completed_sessions.append(result.trading_date)

    return report


def download_spy_sessions(
    start_date: date,
    end_date: date,
    api_key: str,
    secret_key: str,
) -> list[dict]:
    """downloads and validates SPY bars for valid trading days"""

    report = download_spy_sessions_with_report(
        start_date,
        end_date,
        api_key,
        secret_key,
    )

    if report.rejected_sessions:
        failed_date, reason = next(iter(report.rejected_sessions.items()))

        raise ValueError(f"Trading session {failed_date} was rejected: {reason}")

    return report.bars