import unittest

from datetime import date
from functools import partial

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.evaluation.historical import evaluate_historical_sessions
from execai.policies.baselines import immediate_action, twap_action


class HistoricalEvaluationTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        self.cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        self.first_date = date(2025, 10, 15)
        self.second_date = date(2025, 10, 16)

        self.policies = {
            "Immediate": immediate_action,
            "TWAP": partial(twap_action, config=self.config),
        }

    def make_bars(self) -> list[dict]:
        return [{"c": 100.0} for _ in range(78)]

    def test_evaluates_multiple_sessions(self):
        sessions = {
            self.first_date: self.make_bars(),
            self.second_date: self.make_bars(),
        }

        results = evaluate_historical_sessions(
            sessions,
            self.config,
            self.cost_config,
            self.policies,
        )

        self.assertEqual(len(results["Immediate"]), 2)
        self.assertEqual(len(results["TWAP"]), 2)

        for trading_date in sessions:
            self.assertAlmostEqual(
                results["Immediate"][trading_date],
                26.0,
            )

            self.assertAlmostEqual(
                results["TWAP"][trading_date],
                7.8,
            )

    def test_results_are_ordered_by_trading_date(self):
        sessions = {
            self.second_date: self.make_bars(),
            self.first_date: self.make_bars(),
        }

        results = evaluate_historical_sessions(
            sessions,
            self.config,
            self.cost_config,
            self.policies,
        )

        self.assertEqual(
            list(results["TWAP"]),
            [self.first_date, self.second_date],
        )

    def test_policies_use_the_same_historical_scenarios(self):
        first_bars = self.make_bars()
        second_bars = self.make_bars()

        # this price change is only available after the first decision
        second_bars[6]["c"] = 105.0

        sessions = {
            self.first_date: first_bars,
            self.second_date: second_bars,
        }

        results = evaluate_historical_sessions(
            sessions,
            self.config,
            self.cost_config,
            self.policies,
        )

        self.assertAlmostEqual(
            results["Immediate"][self.first_date],
            results["Immediate"][self.second_date],
        )

        self.assertLess(
            results["TWAP"][self.second_date],
            results["TWAP"][self.first_date],
        )

    def test_rejects_empty_sessions(self):
        with self.assertRaises(ValueError):
            evaluate_historical_sessions(
                {},
                self.config,
                self.cost_config,
                self.policies,
            )

    def test_rejects_empty_policies(self):
        with self.assertRaises(ValueError):
            evaluate_historical_sessions(
                {self.first_date: self.make_bars()},
                self.config,
                self.cost_config,
                {},
            )

    def test_historical_evaluation_uses_relative_liquidity(self):
        sessions = {
            self.first_date: self.make_bars(),
        }

        results = evaluate_historical_sessions(
            sessions,
            self.config,
            self.cost_config,
            {"Immediate": immediate_action},
            liquidity_profile=[2.0] + [1.0] * 11,
        )

        self.assertAlmostEqual(
            results["Immediate"][self.first_date],
            16.0,
        )


if __name__ == "__main__":
    unittest.main()
