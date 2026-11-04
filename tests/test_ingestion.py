import unittest

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch
from execai.data.ingestion import download_spy_sessions


class DataIngestionTests(unittest.TestCase):

    def make_bars(
        self,
        session_start: datetime,
        num_bars: int,
    ) -> list[dict]:
        """creates a complete sequence of valid five-minute market bars"""

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

    @patch("execai.data.ingestion.fetch_spy_bars")
    def test_downloads_multiple_trading_sessions(self, mock_fetch):
        first_session = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

        second_session = datetime(2025, 10, 16, 13, 30, tzinfo=timezone.utc)

        mock_fetch.side_effect = [
            self.make_bars(first_session, 78),
            self.make_bars(second_session, 78),
        ]

        bars = download_spy_sessions(
            date(2025, 10, 15),
            date(2025, 10, 16),
            "test-key",
            "test-secret",
        )

        self.assertEqual(len(bars), 156)
        self.assertEqual(mock_fetch.call_count, 2)

        first_request = mock_fetch.call_args_list[0]

        self.assertEqual(
            first_request.args[0],
            first_session,
        )

        self.assertEqual(
            first_request.args[1],
            datetime(2025, 10, 15, 19, 59, 59, tzinfo=timezone.utc),
        )

    @patch("execai.data.ingestion.fetch_spy_bars")
    def test_handles_early_closing_session(self, mock_fetch):
        session_start = datetime(2025, 11, 28, 14, 30, tzinfo=timezone.utc)

        mock_fetch.return_value = self.make_bars(session_start, 42)

        bars = download_spy_sessions(
            date(2025, 11, 28),
            date(2025, 11, 28),
            "test-key",
            "test-secret",
        )

        self.assertEqual(len(bars), 42)

        requested_end = mock_fetch.call_args.args[1]

        self.assertEqual(
            requested_end,
            datetime(2025, 11, 28, 17, 59, 59, tzinfo=timezone.utc),
        )

    @patch("execai.data.ingestion.fetch_spy_bars")
    def test_rejects_incomplete_session(self, mock_fetch):
        session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

        bars = self.make_bars(session_start, 78)
        bars.pop(10)

        mock_fetch.return_value = bars

        with self.assertRaisesRegex(ValueError, "Missing"):
            download_spy_sessions(
                date(2025, 10, 15),
                date(2025, 10, 15),
                "test-key",
                "test-secret",
            )

    @patch("execai.data.ingestion.fetch_spy_bars")
    def test_rejects_date_range_without_trading_days(self, mock_fetch):
        with self.assertRaisesRegex(ValueError, "No trading sessions"):
            download_spy_sessions(
                date(2025, 10, 18),
                date(2025, 10, 19),
                "test-key",
                "test-secret",
            )

        mock_fetch.assert_not_called()

    def test_rejects_invalid_date_range(self):
        with self.assertRaises(ValueError):
            download_spy_sessions(
                date(2025, 10, 16),
                date(2025, 10, 15),
                "test-key",
                "test-secret",
            )

    def test_requires_api_credentials(self):
        with self.assertRaises(ValueError):
            download_spy_sessions(
                date(2025, 10, 15),
                date(2025, 10, 15),
                "",
                "",
            )


if __name__ == "__main__":
    unittest.main()