import json
import unittest

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from execai.data.ingestion import TradingSessionResult
from execai.data.storage import save_session_checkpoint
from scripts.download_market_data import download_and_save_sessions


class MarketDataDownloadTests(unittest.TestCase):

    def setUp(self):
        self.trading_date = date(2025, 10, 15)

    def make_bars(self):
        """Create a complete trading session for testing."""

        session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

        bars = []

        for bar_index in range(78):
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

    @patch("scripts.download_market_data.iter_spy_sessions")
    def test_saves_downloaded_session(self, mock_sessions):
        bars = self.make_bars()

        mock_sessions.return_value = iter(
            [
                TradingSessionResult(
                    trading_date=self.trading_date,
                    bars=bars,
                )
            ]
        )

        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            report = download_and_save_sessions(
                self.trading_date,
                self.trading_date,
                "test-key",
                "test-secret",
                checkpoint_directory,
            )

            checkpoint_path = checkpoint_directory / "2025-10-15.csv"

            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(
                report["downloaded_sessions"],
                ["2025-10-15"],
            )

    @patch("scripts.download_market_data.iter_spy_sessions")
    def test_skips_existing_checkpoint(self, mock_sessions):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            save_session_checkpoint(
                self.make_bars(),
                self.trading_date,
                checkpoint_directory,
            )

            mock_sessions.return_value = iter([])

            report = download_and_save_sessions(
                self.trading_date,
                self.trading_date,
                "test-key",
                "test-secret",
                checkpoint_directory,
            )

            self.assertEqual(
                report["existing_sessions"],
                ["2025-10-15"],
            )

            self.assertEqual(report["downloaded_sessions"], [])

            self.assertEqual(
                mock_sessions.call_args.kwargs["skip_dates"],
                {self.trading_date},
            )

    @patch("scripts.download_market_data.iter_spy_sessions")
    def test_records_rejected_session(self, mock_sessions):
        mock_sessions.return_value = iter(
            [
                TradingSessionResult(
                    trading_date=self.trading_date,
                    bars=[],
                    validation_error="Missing five-minute bar.",
                )
            ]
        )

        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            report = download_and_save_sessions(
                self.trading_date,
                self.trading_date,
                "test-key",
                "test-secret",
                checkpoint_directory,
            )

            self.assertIn(
                "2025-10-15",
                report["rejected_sessions"],
            )

            self.assertFalse((checkpoint_directory / "2025-10-15.csv").exists())

    @patch("scripts.download_market_data.iter_spy_sessions")
    def test_preserves_checkpoint_after_api_failure(self, mock_sessions):
        def failing_download(*args, **kwargs):
            yield TradingSessionResult(
                trading_date=self.trading_date,
                bars=self.make_bars(),
            )

            raise RuntimeError("Alpaca connection failed.")

        mock_sessions.side_effect = failing_download

        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            with self.assertRaisesRegex(RuntimeError, "connection failed"):
                download_and_save_sessions(
                    self.trading_date,
                    date(2025, 10, 16),
                    "test-key",
                    "test-secret",
                    checkpoint_directory,
                )

            checkpoint_path = checkpoint_directory / "2025-10-15.csv"

            self.assertTrue(checkpoint_path.exists())

            report_path = (
                checkpoint_directory / "reports" / "2025-10-15_2025-10-16.json"
            )

            saved_report = json.loads(report_path.read_text(encoding="utf-8"))

            self.assertEqual(
                saved_report["downloaded_sessions"],
                ["2025-10-15"],
            )

            self.assertIsNotNone(saved_report["download_error"])


if __name__ == "__main__":
    unittest.main()
