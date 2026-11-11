import unittest

from datetime import date
from unittest.mock import patch

from execai.environment.config import ExecutionConfig
from execai.environment.costs import (
    ExecutionCostConfig,
    calculate_sell_execution,
)
from execai.evaluation.robustness import evaluate_frozen_scenario


class RobustnessEvaluationTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.nominal_cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        self.forecast_liquidity = [1.0, 1.0, 1.0]

        self.trading_date = date(2025, 1, 2)

        self.sessions = {
            self.trading_date: [],
        }

        def immediate_policy(observation):
            return observation["remaining_lots"]

        self.policies = {
            "Immediate": immediate_policy,
        }

    @patch("execai.evaluation.robustness.evaluate_historical_sessions")
    def test_nominal_scenario_preserves_market_settings(
        self,
        mock_evaluate,
    ):
        expected_results = {
            "Immediate": {
                self.trading_date: 26.0,
            },
        }

        mock_evaluate.return_value = expected_results

        results = evaluate_frozen_scenario(
            self.sessions,
            self.config,
            self.nominal_cost_config,
            self.policies,
            self.forecast_liquidity,
            impact_model="linear",
            impact_strength_bps=20.0,
        )

        self.assertEqual(results, expected_results)

        evaluated_cost_config = mock_evaluate.call_args.args[2]
        evaluated_policies = mock_evaluate.call_args.args[3]

        self.assertEqual(
            evaluated_cost_config,
            self.nominal_cost_config,
        )

        self.assertIs(evaluated_policies, self.policies)

        self.assertEqual(
            mock_evaluate.call_args.kwargs["liquidity_profile"],
            self.forecast_liquidity,
        )

    @patch("execai.evaluation.robustness.evaluate_historical_sessions")
    def test_impact_strength_changes_execution_costs(
        self,
        mock_evaluate,
    ):
        evaluated_cost_configs = []

        for impact_strength in (10.0, 40.0):
            evaluate_frozen_scenario(
                self.sessions,
                self.config,
                self.nominal_cost_config,
                self.policies,
                self.forecast_liquidity,
                impact_model="linear",
                impact_strength_bps=impact_strength,
            )

            evaluated_cost_configs.append(mock_evaluate.call_args.args[2])

        low_impact_execution = calculate_sell_execution(
            market_price=100.0,
            shares_sold=100,
            total_order_shares=400,
            cost_config=evaluated_cost_configs[0],
        )

        high_impact_execution = calculate_sell_execution(
            market_price=100.0,
            shares_sold=100,
            total_order_shares=400,
            cost_config=evaluated_cost_configs[1],
        )

        self.assertGreater(
            low_impact_execution.execution_price,
            high_impact_execution.execution_price,
        )

        self.assertEqual(
            self.nominal_cost_config.temporary_impact_bps,
            20.0,
        )

    @patch("execai.evaluation.robustness.evaluate_historical_sessions")
    def test_reduced_liquidity_increases_execution_cost(
        self,
        mock_evaluate,
    ):
        evaluate_frozen_scenario(
            self.sessions,
            self.config,
            self.nominal_cost_config,
            self.policies,
            self.forecast_liquidity,
            impact_model="linear",
            impact_strength_bps=20.0,
            liquidity_scale=0.5,
        )

        scenario_cost_config = mock_evaluate.call_args.args[2]

        scenario_liquidity = mock_evaluate.call_args.kwargs["liquidity_profile"]

        self.assertEqual(
            scenario_liquidity,
            [0.5, 0.5, 0.5],
        )

        reference_execution = calculate_sell_execution(
            100.0,
            100,
            400,
            self.nominal_cost_config,
            relative_liquidity=1.0,
        )

        stressed_execution = calculate_sell_execution(
            100.0,
            100,
            400,
            scenario_cost_config,
            relative_liquidity=scenario_liquidity[0],
        )

        self.assertLess(
            stressed_execution.execution_price,
            reference_execution.execution_price,
        )

    @patch("execai.evaluation.robustness.evaluate_historical_sessions")
    def test_square_root_impact_changes_execution_cost(
        self,
        mock_evaluate,
    ):
        evaluate_frozen_scenario(
            self.sessions,
            self.config,
            self.nominal_cost_config,
            self.policies,
            self.forecast_liquidity,
            impact_model="square_root",
            impact_strength_bps=20.0,
        )

        square_root_cost_config = mock_evaluate.call_args.args[2]

        linear_execution = calculate_sell_execution(
            100.0,
            100,
            400,
            self.nominal_cost_config,
        )

        square_root_execution = calculate_sell_execution(
            100.0,
            100,
            400,
            square_root_cost_config,
        )

        self.assertLess(
            square_root_execution.execution_price,
            linear_execution.execution_price,
        )

    @patch("execai.evaluation.robustness.evaluate_historical_sessions")
    def test_reuses_the_same_policy_objects(
        self,
        mock_evaluate,
    ):
        for impact_strength in (10.0, 20.0, 40.0):
            evaluate_frozen_scenario(
                self.sessions,
                self.config,
                self.nominal_cost_config,
                self.policies,
                self.forecast_liquidity,
                impact_model="linear",
                impact_strength_bps=impact_strength,
            )

        self.assertEqual(mock_evaluate.call_count, 3)

        for call in mock_evaluate.call_args_list:
            self.assertIs(
                call.args[3]["Immediate"],
                self.policies["Immediate"],
            )

    def test_rejects_invalid_liquidity_scale(self):
        for scale in (0.0, -1.0, float("nan"), True):
            with self.subTest(scale=scale):
                with self.assertRaises(ValueError):
                    evaluate_frozen_scenario(
                        self.sessions,
                        self.config,
                        self.nominal_cost_config,
                        self.policies,
                        self.forecast_liquidity,
                        impact_model="linear",
                        impact_strength_bps=20.0,
                        liquidity_scale=scale,
                    )

    def test_rejects_invalid_market_model(self):
        with self.assertRaises(ValueError):
            evaluate_frozen_scenario(
                self.sessions,
                self.config,
                self.nominal_cost_config,
                self.policies,
                self.forecast_liquidity,
                impact_model="unknown",
                impact_strength_bps=20.0,
            )

    def test_rejects_missing_policies(self):
        with self.assertRaises(ValueError):
            evaluate_frozen_scenario(
                self.sessions,
                self.config,
                self.nominal_cost_config,
                {},
                self.forecast_liquidity,
                impact_model="linear",
                impact_strength_bps=20.0,
            )


if __name__ == "__main__":
    unittest.main()
