import csv
import unittest

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.evaluate_historical import (
    evaluate_validation_baselines,
    save_daily_results,
)


class HistoricalEvaluationScriptTests(unittest.TestCase):

    def setUp(self):
        self.first_date = date(2025, 10, 15)
        self.second_date = date(2025, 10, 16)

        self.results = {
            "Immediate": {
                self.first_date: 26.0,
                self.second_date: 26.0,
            },
            "TWAP": {
                self.first_date: -7.0,
                self.second_date: 12.0,
            },
        }

    @patch("scripts.evaluate_historical.evaluate_historical_sessions")
    @patch("scripts.evaluate_historical.load_dataset_split")
    def test_evaluation_uses_validation_data(
        self,
        mock_load,
        mock_evaluate,
    ):
        checkpoint_directory = Path("data/raw/SPY_5min_SIP")

        mock_load.return_value = {self.first_date: [{"c": 100.0}]}

        mock_evaluate.return_value = self.results

        results = evaluate_validation_baselines(checkpoint_directory)

        mock_load.assert_called_once_with(
            checkpoint_directory,
            "validation",
        )

        self.assertEqual(results, self.results)

    def test_saves_daily_execution_costs(self):
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"

            save_daily_results(self.results, output_path)

            with output_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["trading_date"], "2025-10-15")

            self.assertAlmostEqual(
                float(rows[0]["immediate_bps"]),
                26.0,
            )

            self.assertAlmostEqual(
                float(rows[0]["twap_minus_immediate_bps"]),
                -33.0,
            )

    def test_rejects_mismatched_trading_dates(self):
        self.results["TWAP"].pop(self.second_date)

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"

            with self.assertRaisesRegex(
                ValueError,
                "matching trading dates",
            ):
                save_daily_results(self.results, output_path)

            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
