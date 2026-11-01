import math

from datetime import datetime


REQUIRED_FIELDS = ("t", "o", "h", "l", "c", "v")
PRICE_FIELDS = ("o", "h", "l", "c")


def validate_bars(bars: list[dict]) -> None:
    """validates historical market bars and their chronological order"""

    if not bars:
        raise ValueError("Market data must contain at least one bar.")

    previous_timestamp = None

    for index, bar in enumerate(bars):
        bar_number = index + 1

        if not isinstance(bar, dict):
            raise ValueError(f"Bar {bar_number} must be a dictionary.")

        missing_fields = [
            field for field in REQUIRED_FIELDS
            if field not in bar
        ]

        if missing_fields:
            raise ValueError(
                f"Bar {bar_number} is missing fields: "
                f"{', '.join(missing_fields)}."
            )

        # validates timestamp
        timestamp_text = bar["t"]

        if not isinstance(timestamp_text, str):
            raise ValueError(f"Bar {bar_number} has an invalid timestamp.")

        try:
            timestamp = datetime.fromisoformat(
                timestamp_text.replace("Z", "+00:00")
            )
        except ValueError as error:
            raise ValueError(
                f"Bar {bar_number} has an invalid timestamp."
            ) from error

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError(
                f"Bar {bar_number} must have a timezone-aware timestamp."
            )

        if previous_timestamp is not None:
            if timestamp <= previous_timestamp:
                raise ValueError(
                    "Market bar timestamps must be strictly increasing."
                )

        previous_timestamp = timestamp

        # validates OHLC prices
        for field in PRICE_FIELDS:
            price = bar[field]

            if (
                isinstance(price, bool)
                or not isinstance(price, (int, float))
                or not math.isfinite(price)
                or price <= 0
            ):
                raise ValueError(
                    f"Bar {bar_number} has an invalid price in '{field}'."
                )

        open_price = bar["o"]
        high_price = bar["h"]
        low_price = bar["l"]
        close_price = bar["c"]

        if not (
            low_price <= open_price <= high_price
            and low_price <= close_price <= high_price
        ):
            raise ValueError(
                f"Bar {bar_number} has inconsistent OHLC prices."
            )

        # validates trading volume
        volume = bar["v"]

        if type(volume) is not int or volume < 0:
            raise ValueError(
                f"Bar {bar_number} has invalid trading volume."
            )