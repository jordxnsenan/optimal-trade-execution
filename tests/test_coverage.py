import unittest

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from execai.data.coverage import audit_market_data
from execai.data.storage import (
    save_bars_to_csv,
    save_session_checkpoint,
)


class DataCoverageTests(unittest.TestCase):

    def make_bars(
        self,
        session_start: datetime,
        num_bars: int,
    ) -> list[dict]:
        """Create valid five-minute market bars."""

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

    def test_missing_dataset_reports_zero_coverage(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            report = audit_market_data(
                checkpoint_directory,
                date(2025, 10, 15),
                date(2025, 10, 15),
            )

            self.assertEqual(report.expected_sessions, 1)
            self.assertEqual(report.completed_sessions, 0)
            self.assertEqual(report.expected_bars, 78)
            self.assertEqual(report.completed_bars, 0)
            self.assertEqual(report.coverage_percent, 0.0)
            self.assertEqual(
                report.missing_dates,
                [date(2025, 10, 15)],
            )

    def test_reports_missing_trading_session(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

            save_session_checkpoint(
                self.make_bars(session_start, 78),
                date(2025, 10, 15),
                checkpoint_directory,
            )

            report = audit_market_data(
                checkpoint_directory,
                date(2025, 10, 15),
                date(2025, 10, 16),
            )

            self.assertEqual(report.expected_sessions, 2)
            self.assertEqual(report.completed_sessions, 1)
            self.assertEqual(report.expected_bars, 156)
            self.assertEqual(report.completed_bars, 78)
            self.assertEqual(report.coverage_percent, 50.0)
            self.assertEqual(
                report.missing_dates,
                [date(2025, 10, 16)],
            )

    def test_complete_early_close_session(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            trading_date = date(2025, 11, 28)

            session_start = datetime(2025, 11, 28, 14, 30, tzinfo=timezone.utc)

            save_session_checkpoint(
                self.make_bars(session_start, 42),
                trading_date,
                checkpoint_directory,
            )

            report = audit_market_data(
                checkpoint_directory,
                trading_date,
                trading_date,
            )

            self.assertEqual(report.expected_bars, 42)
            self.assertEqual(report.completed_bars, 42)
            self.assertEqual(report.coverage_percent, 100.0)

    def test_rejects_incomplete_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            session_start = datetime(2025, 10, 15, 13, 30, tzinfo=timezone.utc)

            bars = self.make_bars(session_start, 78)
            bars.pop(10)

            save_bars_to_csv(
                bars,
                checkpoint_directory / "2025-10-15.csv",
            )

            with self.assertRaisesRegex(ValueError, "Invalid checkpoint"):
                audit_market_data(
                    checkpoint_directory,
                    date(2025, 10, 15),
                    date(2025, 10, 15),
                )

    def test_rejects_invalid_date_range(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                audit_market_data(
                    Path(directory),
                    date(2025, 10, 16),
                    date(2025, 10, 15),
                )


if __name__ == "__main__":
    unittest.main()
