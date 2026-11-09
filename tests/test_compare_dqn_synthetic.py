import unittest
import torch

from functools import partial
from unittest.mock import patch

from scripts.compare_dqn_synthetic import evaluate_policies
from execai.agents.dqn import DQNAgent
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.policies.baselines import immediate_action, twap_action


class SyntheticDQNComparisonTests(unittest.TestCase):

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

        self.environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

    def test_evaluates_policies_on_common_price_paths(self):
        price_paths = [
            [100.0, 100.0, 100.0],
            [100.0, 100.5, 101.0],
        ]

        policies = {
            "Immediate": immediate_action,
            "TWAP": partial(twap_action, config=self.config),
        }

        results = evaluate_policies(
            self.environment,
            price_paths,
            policies,
        )

        self.assertEqual(len(results["Immediate"]), 2)
        self.assertEqual(len(results["TWAP"]), 2)

        self.assertAlmostEqual(results["Immediate"][0], 26.0)
        self.assertAlmostEqual(results["TWAP"][0], 13.5)

        # immediate execution has the same cost on both paths because its entire order executes at the first price
        self.assertAlmostEqual(
            results["Immediate"][0],
            results["Immediate"][1],
        )

    def test_dqn_evaluation_disables_exploration(self):
        torch.manual_seed(42)

        agent = DQNAgent(self.config, seed=42)

        price_paths = [
            [100.0, 100.0, 100.0],
        ]

        with patch.object(
            agent,
            "select_action",
            wraps=agent.select_action,
        ) as select_action:
            evaluate_policies(
                self.environment,
                price_paths,
                {
                    "DQN": partial(
                        agent.select_action,
                        exploration_rate=0.0,
                    ),
                },
            )

        self.assertGreater(select_action.call_count, 0)

        for recorded_call in select_action.call_args_list:
            self.assertEqual(
                recorded_call.kwargs["exploration_rate"],
                0.0,
            )

    def test_rejects_empty_evaluation_paths(self):
        with self.assertRaisesRegex(ValueError, "evaluation path"):
            evaluate_policies(
                self.environment,
                [],
                {"Immediate": immediate_action},
            )

    def test_evaluates_episode_specific_liquidity(self):
        price_paths = [
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ]

        liquidity_profiles = [
            [0.5, 1.0, 1.0],
            [2.0, 1.0, 1.0],
        ]

        results = evaluate_policies(
            self.environment,
            price_paths,
            {"Immediate": immediate_action},
            liquidity_profiles=liquidity_profiles,
        )

        # immediate execution sells everything at first step
        self.assertAlmostEqual(results["Immediate"][0], 46.0)
        self.assertAlmostEqual(results["Immediate"][1], 16.0)

    def test_policies_receive_the_same_liquidity(self):
        price_paths = [
            [100.0, 100.0, 100.0],
        ]

        liquidity_profiles = [
            [0.5, 1.5, 1.0],
        ]

        results = evaluate_policies(
            self.environment,
            price_paths,
            {
                "First": immediate_action,
                "Second": immediate_action,
            },
            liquidity_profiles=liquidity_profiles,
        )

        self.assertEqual(results["First"], results["Second"])
        self.assertAlmostEqual(results["First"][0], 46.0)

    def test_rejects_unmatched_evaluation_liquidity(self):
        price_paths = [
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ]

        with self.assertRaisesRegex(ValueError, "matching liquidity"):
            evaluate_policies(
                self.environment,
                price_paths,
                {"Immediate": immediate_action},
                liquidity_profiles=[
                    [1.0, 1.0, 1.0],
                ],
            )


if __name__ == "__main__":
    unittest.main()
