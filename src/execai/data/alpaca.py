import json

from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


SPY_BARS_URL = "https://data.alpaca.markets/v2/stocks/SPY/bars"
PAGE_LIMIT = 1000
REQUEST_TIMEOUT_SECONDS = 30


def _format_utc_timestamp(timestamp: datetime) -> str:
    """formats a timezone-aware timestamp for the Alpaca API"""

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone.")

    utc_timestamp = timestamp.astimezone(timezone.utc)
    return utc_timestamp.isoformat().replace("+00:00", "Z")


def fetch_spy_bars(
    start_time: datetime,
    end_time: datetime,
    api_key: str,
    secret_key: str,
) -> list[dict]:
    """retrieves five-minute SPY bars including all response pages"""

    start_timestamp = _format_utc_timestamp(start_time)
    end_timestamp = _format_utc_timestamp(end_time)

    if start_time >= end_time:
        raise ValueError("Start time must be before end time.")

    if not api_key or not secret_key:
        raise ValueError("Alpaca API credentials are required.")

    query_parameters = {
        "timeframe": "5Min",
        "start": start_timestamp,
        "end": end_timestamp,
        "feed": "sip",
        "adjustment": "raw",
        "sort": "asc",
        "limit": PAGE_LIMIT,
    }

    all_bars = []
    seen_page_tokens = set()
    page_token = None

    while True:
        if page_token is not None:
            query_parameters["page_token"] = page_token

        request_url = f"{SPY_BARS_URL}?{urlencode(query_parameters)}"

        request = Request(
            request_url,
            headers={
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
            },
        )

        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response_data = json.load(response)

        except HTTPError as error:
            raise RuntimeError(
                f"Alpaca request failed (HTTP {error.code})."
            ) from error

        except URLError as error:
            raise RuntimeError(
                "Could not connect to Alpaca."
            ) from error

        if not isinstance(response_data, dict) or not isinstance(
            response_data.get("bars"), list
        ):
            raise ValueError("Alpaca returned an unexpected response.")

        all_bars.extend(response_data["bars"])
        page_token = response_data.get("next_page_token")

        if page_token is None:
            return all_bars

        if not isinstance(page_token, str) or not page_token:
            raise ValueError("Alpaca returned an invalid page token.")

        if page_token in seen_page_tokens:
            raise RuntimeError("Alpaca returned a repeated page token.")

        seen_page_tokens.add(page_token)