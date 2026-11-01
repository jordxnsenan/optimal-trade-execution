import json
import os

from dotenv import load_dotenv

from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"

# reg trading session Oct 15 2025
QUERY_PARAMETERS = {
    "timeframe": "5Min",
    "start": "2025-10-15T13:30:00Z",
    "end": "2025-10-15T19:59:59Z",
    "feed": "sip",
    "adjustment": "raw",
    "limit": 1000,
}


def main():
    load_dotenv()

    api_key = os.getenv("APCA_API_KEY_ID")
    secret_key = os.getenv("APCA_API_SECRET_KEY")

    if not api_key or not secret_key:
        print("Missing Alpaca credentials. Check your .env file.")
        
        return

    request_url = f"{API_URL}?{urlencode(QUERY_PARAMETERS)}"

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
        print("No historical bars were returned.")
        return

    first_bar = bars[0]
    last_bar = bars[-1]

    required_fields = ("t", "o", "h", "l", "c", "v")

    if any(field not in first_bar for field in required_fields):
        print("The returned data is missing required OHLCV fields.")
        return

    print("\nHistorical SPY Data Check")
    print("---------------------------")
    print(f"Date: 2025-10-15")
    print(f"Requested feed: {QUERY_PARAMETERS['feed']}")
    print(f"Bars returned: {len(bars)}")
    print(f"First timestamp: {first_bar['t']}")
    print(f"Last timestamp: {last_bar['t']}")
    print(f"First closing price: ${first_bar['c']}")
    print(f"First volume: {first_bar['v']:,}")
    print(f"More pages available: {bool(data.get('next_page_token'))}")

    if len(bars) == 78:
        print("Received the expected number of regular-session bars.")
    else:
        print("Bar count needs further investigation.")


if __name__ == "__main__":
    main()