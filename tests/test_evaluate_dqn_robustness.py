import csv
import unittest

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.evaluate_dqn_robustness import (
    ROBUSTNESS_SCENARIOS,
    run_robustness_scenarios,
    save_robustness_results,
)
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig


class DQNRobustnessScriptTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        self.trading_date = date(2025, 1, 2)

        self.sessions = {
            self.trading_date: [],
        }

        def immediate_policy(observation):
            return observation["remaining_lots"]

        self.policies = {
            "Immediate": immediate_policy,
        }

    @patch("scripts.evaluate_dqn_robustness.evaluate_frozen_scenario")
    def test_reuses_frozen_policies_across_scenarios(self, mock_evaluate):
        mock_evaluate.return_value = {"Immediate": {self.trading_date: 26.0}}

        results = run_robustness_scenarios(
            self.sessions,
            self.config,
            self.cost_config,
            self.policies,
            [1.0, 1.0, 1.0],
            ROBUSTNESS_SCENARIOS,
        )

        self.assertEqual(
            len(results),
            len(ROBUSTNESS_SCENARIOS),
        )

        self.assertEqual(
            mock_evaluate.call_count,
            len(ROBUSTNESS_SCENARIOS),
        )

        for recorded_call in mock_evaluate.call_args_list:
            self.assertIs(
                recorded_call.args[3],
                self.policies,
            )

        self.assertEqual(
            self.cost_config.temporary_impact_bps,
            20.0,
        )

    def test_robustness_scenarios_match_the_protocol(self):
        self.assertEqual(
            ROBUSTNESS_SCENARIOS["nominal"],
            ("linear", 20.0, 1.0),
        )

        self.assertEqual(
            ROBUSTNESS_SCENARIOS["high_impact"],
            ("linear", 40.0, 1.0),
        )

        self.assertEqual(
            ROBUSTNESS_SCENARIOS["square_root"],
            ("square_root", 20.0, 1.0),
        )

        self.assertEqual(
            ROBUSTNESS_SCENARIOS["reduced_liquidity"],
            ("linear", 20.0, 0.5),
        )

    def test_saves_daily_results_with_checkpoint_metadata(self):
        results = {
            "nominal": {
                "TWAP": {
                    self.trading_date: 5.0,
                },
                "Basic DQN (seed 42)": {
                    self.trading_date: 6.0,
                },
            },
        }

        metadata = {
            "Basic DQN (seed 42)": (42, 1800),
        }

        scenarios = {
            "nominal": ROBUSTNESS_SCENARIOS["nominal"],
        }

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"

            save_robustness_results(
                results,
                scenarios,
                metadata,
                output_path,
            )

            with output_path.open(
                newline="",
                encoding="utf-8",
            ) as csv_file:
                rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)

        rows_by_policy = {row["policy"]: row for row in rows}

        self.assertEqual(
            rows_by_policy["TWAP"]["training_seed"],
            "",
        )

        self.assertEqual(
            rows_by_policy["Basic DQN (seed 42)"]["training_seed"],
            "42",
        )

        self.assertEqual(
            rows_by_policy["Basic DQN (seed 42)"]["best_episode"],
            "1800",
        )

        self.assertEqual(
            rows_by_policy["Basic DQN (seed 42)"]["shortfall_bps"],
            "6.0",
        )


if __name__ == "__main__":
    unittest.main()
