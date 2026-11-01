import argparse
import json
import os

from dotenv import load_dotenv
from zoneinfo import ZoneInfo
from datetime import date, datetime, time, timedelta, timezone

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
EXPECTED_BAR_COUNT = 78


def format_utc_timestamp(timestamp: datetime) -> str:
    """converts timestamp to the UTC format expected by Alpaca"""

    utc_timestamp = timestamp.astimezone(timezone.utc)
    return utc_timestamp.isoformat().replace("+00:00", "Z")


def build_query_parameters(trading_date: date) -> dict:
    """builds a historical data request for one regular trading session"""

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

    return {
        "timeframe": "5Min",
        "start": format_utc_timestamp(session_start),
        "end": format_utc_timestamp(session_end),
        "feed": "sip",
        "adjustment": "raw",
        "limit": 1000,
    }


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

    query_parameters = build_query_parameters(arguments.date)
    request_url = f"{API_URL}?{urlencode(query_parameters)}"

    request = Request(
        request_url,
        headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            data = json.load(response)

    except HTTPError as error:
        print(f"Alpaca request failed (HTTP {error.code}).")
        print(error.read().decode("utf-8", errors="replace"))
        return

    except URLError as error:
        print(f"Could not connect to Alpaca: {error.reason}")
        return

    bars = data.get("bars", [])

    if not bars:
        print(f"No historical bars returned for {arguments.date}.")
        return

    required_fields = {"t", "o", "h", "l", "c", "v"}

    if any(not required_fields.issubset(bar) for bar in bars):
        print("Historical data is missing required OHLCV fields.")
        return

    first_bar = bars[0]
    last_bar = bars[-1]

    print("\nHistorical SPY Data Check")
    print("---------------------------")
    print(f"Date: {arguments.date}")
    print(f"Feed: {query_parameters['feed']}")
    print(f"Bars returned: {len(bars)}")
    print(f"First timestamp: {first_bar['t']}")
    print(f"Last timestamp: {last_bar['t']}")
    print(f"First closing price: ${first_bar['c']}")
    print(f"First volume: {first_bar['v']:,}")

    if data.get("next_page_token"):
        print("Additional data pages are available.")

    if len(bars) == EXPECTED_BAR_COUNT:
        print("Expected regular-session bar count received.")
    else:
        print("Bar count requires further investigation.")


if __name__ == "__main__":
    main()