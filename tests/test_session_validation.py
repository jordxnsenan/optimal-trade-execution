import unittest

from datetime import date, datetime, timedelta, timezone
from execai.data.session_validation import validate_trading_session


class TradingSessionTests(unittest.TestCase):

    def make_bars(
        self,
        session_start: datetime,
        num_bars: int,
    ) -> list[dict]:
        """generates valid five-minute bars"""

        bars = []

        for bar_index in range(num_bars):
            timestamp = session_start + timedelta(minutes=5 * bar_index)

            bars.append(
                {
                    "t": timestamp.isoformat().replace("+00:00", "Z"),
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 150_000,
                }
            )

        return bars

    def test_complete_regular_session(self):
        session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 78)

        validate_trading_session(bars, date(2025, 10, 15))

    def test_complete_session_during_standard_time(self):
        session_start = datetime(2023, 2, 15, 14, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 78)

        validate_trading_session(bars, date(2023, 2, 15))

    def test_missing_market_bar(self):
        session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 78)
        bars.pop(10)

        with self.assertRaisesRegex(ValueError, "Missing"):
            validate_trading_session(bars, date(2025, 10, 15))

    def test_unexpected_market_bar(self):
        session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 79)

        with self.assertRaisesRegex(ValueError, "Unexpected"):
            validate_trading_session(bars, date(2025, 10, 15))

    def test_complete_early_close_session(self):
        session_start = datetime(2025, 11, 28, 14, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 42)

        validate_trading_session(bars, date(2025, 11, 28))

    def test_closed_trading_day_allows_empty_data(self):
        validate_trading_session([], date(2025, 12, 25))

    def test_closed_trading_day_rejects_bars(self):
        session_start = datetime(2025, 12, 25, 14, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 1)

        with self.assertRaisesRegex(ValueError, "closed trading day"):
            validate_trading_session(bars, date(2025, 12, 25))

    def test_open_trading_day_requires_bars(self):
        with self.assertRaises(ValueError):
            validate_trading_session([], date(2025, 10, 15))


if __name__ == "__main__":
    unittest.main()
