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
    get_experiment_output_path,
)

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.liquidity import build_relative_liquidity_profile
from execai.policies.almgren_chriss import (
    calculate_expected_cost_bps,
    optimize_almgren_chriss_schedule,
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

    @patch("scripts.evaluate_historical.load_dataset_split")
    def test_liquidity_aware_evaluation_reoptimizes_ac(self, mock_load):
        mock_load.return_value = {self.first_date: [{"c": 100.0} for _ in range(78)]}

        volume_fractions = [1.0] * 78
        volume_fractions[6] = 8.0

        calibration = {
            "training_start": "2023-01-01",
            "training_end": "2024-12-31",
            "volatility_per_step": 0.000721,
            "volume_fractions": volume_fractions,
        }

        with TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "calibration.json"

            calibration_path.write_text(
                json.dumps(calibration),
                encoding="utf-8",
            )

            reference_results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
                liquidity_aware=False,
            )

            liquidity_results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
                liquidity_aware=True,
            )

        trading_date = self.first_date

        # original model should retain its known result
        self.assertAlmostEqual(
            reference_results["Immediate"][trading_date],
            26.0,
        )

        # higher first-interval liquidity reduces immediate impact
        self.assertLess(
            liquidity_results["Immediate"][trading_date],
            reference_results["Immediate"][trading_date],
        )

        # risk-neutral AC should minimize cost under the same liquidity assumptions used to evaluate TWAP
        self.assertLessEqual(
            liquidity_results["AC Risk-Neutral"][trading_date],
            liquidity_results["TWAP"][trading_date] + 1e-9,
        )

    def test_liquidity_aware_risk_summary_uses_forecast_liquidity(self):
        volume_fractions = [1.0] * 78
        volume_fractions[6] = 8.0

        calibration = {
            "training_start": "2023-01-01",
            "training_end": "2024-12-31",
            "volatility_per_step": 0.000721,
            "volume_fractions": volume_fractions,
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
                liquidity_aware=True,
            )

        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        liquidity_profile = build_relative_liquidity_profile(
            volume_fractions,
            config,
        )

        schedule = optimize_almgren_chriss_schedule(
            config,
            cost_config,
            reference_price=100.0,
            volatility_per_step=0.000721,
            risk_aversion=0.0,
            liquidity_profile=liquidity_profile,
        )

        expected_cost = calculate_expected_cost_bps(
            schedule,
            config,
            cost_config,
            reference_price=100.0,
            liquidity_profile=liquidity_profile,
        )

        self.assertAlmostEqual(
            summaries["AC Risk-Neutral"]["expected_cost_bps"],
            expected_cost,
        )

        self.assertLessEqual(
            summaries["AC Risk-Sensitive"]["comparison_objective_bps"],
            summaries["AC Risk-Neutral"]["comparison_objective_bps"],
        )

    def test_impact_models_use_separate_output_files(self):
        output_paths = {
            get_experiment_output_path("linear", False),
            get_experiment_output_path("linear", True),
            get_experiment_output_path("square_root", False),
            get_experiment_output_path("square_root", True),
        }

        self.assertEqual(len(output_paths), 4)

    @patch("scripts.evaluate_historical.load_dataset_split")
    def test_square_root_model_is_used_in_historical_evaluation(
        self,
        mock_load,
    ):
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

            linear_results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
                liquidity_aware=True,
                impact_model="linear",
            )

            square_root_results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
                liquidity_aware=True,
                impact_model="square_root",
            )

        trading_date = self.first_date

        # both models agree when the entire order is sold at once
        self.assertAlmostEqual(
            linear_results["Immediate"][trading_date],
            square_root_results["Immediate"][trading_date],
        )

        # square root impact is greater for smaller trades
        self.assertGreater(
            square_root_results["TWAP"][trading_date],
            linear_results["TWAP"][trading_date],
        )

        # almgren chriss should optimize using the selected model
        self.assertLessEqual(
            square_root_results["AC Risk-Neutral"][trading_date],
            square_root_results["TWAP"][trading_date] + 1e-9,
        )

    def test_risk_summary_uses_selected_impact_model(self):
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

            linear_summary = summarize_almgren_chriss_risk(
                self.results,
                calibration_path,
                impact_model="linear",
            )

            square_root_summary = summarize_almgren_chriss_risk(
                self.results,
                calibration_path,
                impact_model="square_root",
            )

        self.assertGreater(
            square_root_summary["AC Risk-Neutral"]["expected_cost_bps"],
            linear_summary["AC Risk-Neutral"]["expected_cost_bps"],
        )

    def test_different_impact_strengths_use_separate_files(self):
        default_path = get_experiment_output_path(
            "linear",
            True,
        )

        low_impact_path = get_experiment_output_path(
            "linear",
            True,
            10.0,
        )

        high_impact_path = get_experiment_output_path(
            "linear",
            True,
            40.0,
        )

        self.assertEqual(
            len({default_path, low_impact_path, high_impact_path}),
            3,
        )

    @patch("scripts.evaluate_historical.load_dataset_split")
    def test_impact_strength_changes_execution_costs(self, mock_load):
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

            low_impact_results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
                impact_strength_bps=10.0,
            )

            high_impact_results = evaluate_validation_baselines(
                Path(directory),
                calibration_path,
                impact_strength_bps=40.0,
            )

        trading_date = self.first_date

        self.assertAlmostEqual(
            low_impact_results["Immediate"][trading_date],
            16.0,
        )

        self.assertAlmostEqual(
            high_impact_results["Immediate"][trading_date],
            46.0,
        )

        self.assertLess(
            low_impact_results["TWAP"][trading_date],
            high_impact_results["TWAP"][trading_date],
        )


if __name__ == "__main__":
    unittest.main()
