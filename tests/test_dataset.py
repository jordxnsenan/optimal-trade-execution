import unittest

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from execai.data.dataset import load_dataset_split, load_spy_sessions
from execai.data.storage import (
    save_bars_to_csv,
    save_session_checkpoint,
)


class HistoricalDatasetTests(unittest.TestCase):

    def make_bars(
        self,
        trading_date: date,
        starting_hour: int,
        num_bars: int,
    ) -> list[dict]:
        """generates valid market bars for one trading session"""

        session_start = datetime(
            trading_date.year,
            trading_date.month,
            trading_date.day,
            starting_hour,
            30,
            tzinfo=timezone.utc,
        )

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

    def test_loads_sessions_in_chronological_order(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            first_date = date(2025, 10, 15)
            second_date = date(2025, 10, 16)

            first_bars = self.make_bars(first_date, 13, 78)
            second_bars = self.make_bars(second_date, 13, 78)

            # saves sessions in reverse order
            save_session_checkpoint(
                second_bars,
                second_date,
                checkpoint_directory,
            )

            save_session_checkpoint(
                first_bars,
                first_date,
                checkpoint_directory,
            )

            sessions = load_spy_sessions(
                checkpoint_directory,
                first_date,
                second_date,
            )

            self.assertEqual(
                list(sessions.keys()),
                [first_date, second_date],
            )

            self.assertEqual(len(sessions[first_date]), 78)
            self.assertEqual(len(sessions[second_date]), 78)
            self.assertEqual(sessions[first_date], first_bars)

    def test_rejects_missing_trading_session(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            trading_date = date(2025, 10, 15)

            save_session_checkpoint(
                self.make_bars(trading_date, 13, 78),
                trading_date,
                checkpoint_directory,
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Missing market data checkpoint",
            ):
                load_spy_sessions(
                    checkpoint_directory,
                    date(2025, 10, 15),
                    date(2025, 10, 16),
                )

    def test_rejects_incomplete_trading_session(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            trading_date = date(2025, 10, 15)

            bars = self.make_bars(trading_date, 13, 78)
            bars.pop(10)

            save_bars_to_csv(
                bars,
                checkpoint_directory / "2025-10-15.csv",
            )

            with self.assertRaisesRegex(ValueError, "Missing"):
                load_spy_sessions(
                    checkpoint_directory,
                    trading_date,
                    trading_date,
                )

    def test_loads_early_closing_session(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)

            trading_date = date(2025, 11, 28)

            bars = self.make_bars(trading_date, 14, 42)

            save_session_checkpoint(
                bars,
                trading_date,
                checkpoint_directory,
            )

            sessions = load_spy_sessions(
                checkpoint_directory,
                trading_date,
                trading_date,
            )

            self.assertEqual(len(sessions[trading_date]), 42)

    def test_rejects_date_range_without_trading_days(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError,
                "No trading sessions",
            ):
                load_spy_sessions(
                    Path(directory),
                    date(2025, 10, 18),
                    date(2025, 10, 19),
                )

    def test_rejects_invalid_date_range(self):
        with TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_spy_sessions(
                    Path(directory),
                    date(2025, 10, 16),
                    date(2025, 10, 15),
                )

    @patch("execai.data.dataset.load_spy_sessions")
    def test_loads_training_partition(self, mock_load):
        checkpoint_directory = Path("data/raw/SPY_5min_SIP")

        expected_sessions = {date(2023, 1, 3): []}

        mock_load.return_value = expected_sessions

        sessions = load_dataset_split(
            checkpoint_directory,
            "training",
        )

        mock_load.assert_called_once_with(
            checkpoint_directory,
            date(2023, 1, 1),
            date(2024, 12, 31),
        )

        self.assertEqual(sessions, expected_sessions)

    @patch("execai.data.dataset.load_spy_sessions")
    def test_loads_validation_and_test_partitions(self, mock_load):
        checkpoint_directory = Path("data/raw/SPY_5min_SIP")

        expected_ranges = {
            "validation": (
                date(2025, 1, 1),
                date(2025, 12, 31),
            ),
            "test": (
                date(2026, 1, 1),
                date(2026, 9, 30),
            ),
        }

        for split_name, (start_date, end_date) in expected_ranges.items():
            with self.subTest(split=split_name):
                mock_load.reset_mock()

                load_dataset_split(
                    checkpoint_directory,
                    split_name,
                )

                mock_load.assert_called_once_with(
                    checkpoint_directory,
                    start_date,
                    end_date,
                )

    @patch("execai.data.dataset.load_spy_sessions")
    def test_rejects_unknown_dataset_partition(self, mock_load):
        checkpoint_directory = Path("data/raw/SPY_5min_SIP")

        with self.assertRaisesRegex(ValueError, "Unknown dataset split"):
            load_dataset_split(
                checkpoint_directory,
                "unknown",
            )

        mock_load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
