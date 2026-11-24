import csv
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import date, datetime, timedelta, timezone

from execai.data.storage import (
    find_completed_sessions,
    load_bars_from_csv,
    save_bars_to_csv,
    save_session_checkpoint,
)


class DataStorageTests(unittest.TestCase):

    def setUp(self):
        self.bars = [
            {
                "t": "2025-10-15T13:30:00Z",
                "o": 100.0,
                "h": 101.0,
                "l": 99.0,
                "c": 100.5,
                "v": 150_000,
            },
            {
                "t": "2025-10-15T13:35:00Z",
                "o": 100.5,
                "h": 102.0,
                "l": 100.0,
                "c": 101.5,
                "v": 175_000,
            },
        ]

    def test_saves_market_bars_to_csv(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "market_data.csv"

            save_bars_to_csv(self.bars, output_path)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                saved_bars = list(reader)

            self.assertEqual(len(saved_bars), 2)
            self.assertEqual(saved_bars[0]["t"], self.bars[0]["t"])
            self.assertEqual(float(saved_bars[0]["c"]), 100.5)
            self.assertEqual(int(saved_bars[1]["v"]), 175_000)

    def test_does_not_overwrite_existing_file(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "market_data.csv"

            save_bars_to_csv(self.bars, output_path)

            with self.assertRaises(FileExistsError):
                save_bars_to_csv(self.bars, output_path)

    def test_rejects_invalid_data_before_saving(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "market_data.csv"

            self.bars[0]["c"] = -100.0

            with self.assertRaises(ValueError):
                save_bars_to_csv(self.bars, output_path)

            self.assertFalse(output_path.exists())

    def make_complete_session(self) -> list[dict]:
        """creates a complete regular trading session for testing"""

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

    def test_loads_saved_market_bars(self):
        with TemporaryDirectory() as directory:
            csv_path = Path(directory) / "market_data.csv"

            save_bars_to_csv(self.bars, csv_path)

            loaded_bars = load_bars_from_csv(csv_path)

            self.assertEqual(loaded_bars, self.bars)

    def test_finds_valid_session_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            trading_date = date(2025, 10, 15)

            csv_path = checkpoint_directory / "2025-10-15.csv"

            save_bars_to_csv(
                self.make_complete_session(),
                csv_path,
            )

            completed_dates = find_completed_sessions(
                checkpoint_directory,
                trading_date,
                trading_date,
            )

            self.assertEqual(completed_dates, {trading_date})

    def test_rejects_incomplete_session_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            trading_date = date(2025, 10, 15)

            incomplete_bars = self.make_complete_session()
            incomplete_bars.pop(10)

            csv_path = checkpoint_directory / "2025-10-15.csv"

            save_bars_to_csv(incomplete_bars, csv_path)

            with self.assertRaisesRegex(
                ValueError,
                "Invalid checkpoint",
            ):
                find_completed_sessions(
                    checkpoint_directory,
                    trading_date,
                    trading_date,
                )

    def test_missing_checkpoint_directory(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory) / "missing"

            completed_dates = find_completed_sessions(
                checkpoint_directory,
                date(2025, 10, 13),
                date(2025, 10, 17),
            )

            self.assertEqual(completed_dates, set())

    def test_saves_complete_session_checkpoint(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            trading_date = date(2025, 10, 15)

            checkpoint_path = save_session_checkpoint(
                self.make_complete_session(),
                trading_date,
                checkpoint_directory,
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(checkpoint_path.name, "2025-10-15.csv")

            loaded_bars = load_bars_from_csv(checkpoint_path)

            self.assertEqual(len(loaded_bars), 78)

    def test_checkpoint_does_not_overwrite_existing_file(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            trading_date = date(2025, 10, 15)
            bars = self.make_complete_session()

            checkpoint_path = save_session_checkpoint(
                bars,
                trading_date,
                checkpoint_directory,
            )

            original_contents = checkpoint_path.read_bytes()

            with self.assertRaises(FileExistsError):
                save_session_checkpoint(
                    bars,
                    trading_date,
                    checkpoint_directory,
                )

            self.assertEqual(
                checkpoint_path.read_bytes(),
                original_contents,
            )

    def test_incomplete_session_is_not_saved(self):
        with TemporaryDirectory() as directory:
            checkpoint_directory = Path(directory)
            trading_date = date(2025, 10, 15)

            bars = self.make_complete_session()
            bars.pop(10)

            with self.assertRaisesRegex(ValueError, "Missing"):
                save_session_checkpoint(
                    bars,
                    trading_date,
                    checkpoint_directory,
                )

            checkpoint_path = checkpoint_directory / "2025-10-15.csv"

            self.assertFalse(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()
