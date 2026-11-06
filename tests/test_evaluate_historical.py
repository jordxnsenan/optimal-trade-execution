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
    summarize_almgren_chriss_risk,
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
            "AC Risk-Neutral": {
                self.first_date: 8.0,
                self.second_date: 7.0,
            },
            "AC Risk-Sensitive": {
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
                "volatility_per_step": 0.000721,
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
        self.assertIn("AC Risk-Neutral", policies)
        self.assertIn("AC Risk-Sensitive", policies)

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

            self.assertAlmostEqual(
                float(rows[0]["ac_risk_neutral_bps"]),
                8.0,
            )

            self.assertAlmostEqual(
                float(rows[0]["ac_risk_sensitive_bps"]),
                10.0,
            )

            self.assertAlmostEqual(
                float(rows[0]["ac_risk_neutral_minus_twap_bps"]),
                15.0,
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

    @patch("scripts.evaluate_historical.load_dataset_split")
    def test_almgren_chriss_baselines_on_flat_market(self, mock_load):
        mock_load.return_value = {self.first_date: [{"c": 100.0} for _ in range(78)]}

        calibration = {
            "training_start": "2023-01-01",
            "training_end": "2024-12-31",
            "volatility_per_step": 0.000721,
            "volume_fractions": [1.0] * 78,
        }

        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"

            calibration_path.write_text(
                json.dumps(calibration),
                encoding="utf-8",
            )

            results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
            )

        trading_date = self.first_date

        risk_neutral_cost = results["AC Risk-Neutral"][trading_date]
        risk_sensitive_cost = results["AC Risk-Sensitive"][trading_date]
        twap_cost = results["TWAP"][trading_date]

        self.assertAlmostEqual(risk_neutral_cost, 7.8)
        self.assertAlmostEqual(risk_neutral_cost, twap_cost)
        self.assertGreater(risk_sensitive_cost, risk_neutral_cost)

    def test_risk_summary_compares_schedules_under_same_objective(self):
        calibration = {
            "training_start": "2023-01-01",
            "training_end": "2024-12-31",
            "volatility_per_step": 0.000721,
        }

        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"

            calibration_path.write_text(
                json.dumps(calibration),
                encoding="utf-8",
            )

            summaries = summarize_almgren_chriss_risk(
                self.results,
                calibration_path,
            )

        neutral = summaries["AC Risk-Neutral"]
        sensitive = summaries["AC Risk-Sensitive"]

        self.assertAlmostEqual(
            neutral["expected_cost_bps"],
            7.8,
        )

        self.assertGreater(
            sensitive["expected_cost_bps"],
            neutral["expected_cost_bps"],
        )

        self.assertLess(
            sensitive["inventory_risk_bps_squared"],
            neutral["inventory_risk_bps_squared"],
        )

        self.assertLessEqual(
            sensitive["comparison_objective_bps"],
            neutral["comparison_objective_bps"],
        )


if __name__ == "__main__":
    unittest.main()
