import argparse
import os

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from execai.data.alpaca import fetch_spy_bars


MARKET_TIMEZONE = ZoneInfo("America/New_York")
EXPECTED_BAR_COUNT = 78


def get_regular_session_bounds(trading_date: date) -> tuple[datetime, datetime]:
    """returns the start and end of a regular trading session"""

    session_start = datetime.combine(
        trading_date,
        time(9, 30),
        tzinfo=MARKET_TIMEZONE,
    )

    session_end = datetime.combine(
        trading_date,
        time(16, 0),
        tzinfo=MARKET_TIMEZONE,
    ) - timedelta(seconds=1)

    return session_start, session_end


def main():
    parser = argparse.ArgumentParser(
        description="Verify historical SPY data access through Alpaca."
    )

    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        default=date(2025, 10, 15),
        help="Trading date in YYYY-MM-DD format.",
    )

    arguments = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("Missing Alpaca credentials. Check your .env file.")
        return

    start_time, end_time = get_regular_session_bounds(arguments.date)

    try:
        bars = fetch_spy_bars(
            start_time,
            end_time,
            api_key,
            secret_key,
        )
    except (RuntimeError, ValueError) as error:
        print(f"Data request failed: {error}")
        return

    if not bars:
        print(f"No historical bars returned for {arguments.date}.")
        return

    required_fields = {"t", "o", "h", "l", "c", "v"}

    if any(
        not isinstance(bar, dict)
        or not required_fields.issubset(bar)
        for bar in bars
    ):
        print("Historical data is missing required OHLCV fields.")
        return

    first_bar = bars[0]
    last_bar = bars[-1]

    print("\nHistorical SPY Data Check")
    print("---------------------------")
    print(f"Date: {arguments.date}")
    print("Feed: SIP")
    print(f"Bars returned: {len(bars)}")
    print(f"First timestamp: {first_bar['t']}")
    print(f"Last timestamp: {last_bar['t']}")
    print(f"First closing price: ${first_bar['c']}")
    print(f"First volume: {first_bar['v']:,}")

    if len(bars) == EXPECTED_BAR_COUNT:
        print("Expected regular-session bar count received.")
    else:
        print("Bar count requires further investigation.")


if __name__ == "__main__":
    main()