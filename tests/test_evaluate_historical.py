import csv
import json
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
            "VWAP": {
                self.first_date: 10.0,
                self.second_date: 9.0,
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

        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"

            calibration = {
                "training_start": "2023-01-01",
                "training_end": "2024-12-31",
                "volume_fractions": [1.0] * 78,
            }

            calibration_path.write_text(
                json.dumps(calibration),
                encoding="utf-8",
            )

            results = evaluate_validation_baselines(
                checkpoint_directory,
                calibration_path,
            )

        mock_load.assert_called_once_with(
            checkpoint_directory,
            "validation",
        )

        self.assertEqual(results, self.results)

        policies = mock_evaluate.call_args.args[3]
        self.assertIn("VWAP", policies)

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

            self.assertAlmostEqual(
                float(rows[0]["vwap_bps"]),
                10.0,
            )

            self.assertAlmostEqual(
                float(rows[0]["vwap_minus_twap_bps"]),
                17.0,
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

    def test_rejects_vwap_calibration_from_wrong_period(self):
        from execai.environment.config import ExecutionConfig
        from scripts.evaluate_historical import load_vwap_schedule

        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        calibration = {
            "training_start": "2025-01-01",
            "training_end": "2025-12-31",
            "volume_fractions": [1.0] * 78,
        }

        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"

            calibration_path.write_text(
                json.dumps(calibration),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "training period"):
                load_vwap_schedule(
                    config,
                    calibration_path,
                )


if __name__ == "__main__":
    unittest.main()
