import unittest

from functools import partial
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.evaluation.comparison import evaluate_policies
from execai.policies.baselines import immediate_action, twap_action


class PolicyComparisonTests(unittest.TestCase):

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

        self.price_paths = [
            [100.0, 100.0, 100.0],
            [100.0, 102.0, 104.0],
        ]

        self.policies = {
            "Immediate": immediate_action,
            "TWAP": partial(twap_action, config=self.config),
        }

    def test_evaluates_policies_on_same_scenarios(self):
        results = evaluate_policies(
            self.config,
            self.cost_config,
            self.price_paths,
            self.policies,
        )

        self.assertEqual(len(results["Immediate"]), 2)
        self.assertEqual(len(results["TWAP"]), 2)

        self.assertAlmostEqual(results["Immediate"][0], 26.0)
        self.assertAlmostEqual(results["Immediate"][1], 26.0)

        self.assertAlmostEqual(results["TWAP"][0], 13.5)

        # TWAP benefits from selling later in a rising market
        self.assertLess(results["TWAP"][1], results["TWAP"][0])

    def test_evaluation_does_not_modify_price_paths(self):
        original_paths = [
            prices.copy()
            for prices in self.price_paths
        ]

        evaluate_policies(
            self.config,
            self.cost_config,
            self.price_paths,
            self.policies,
        )

        self.assertEqual(self.price_paths, original_paths)

    def test_requires_price_paths(self):
        with self.assertRaises(ValueError):
            evaluate_policies(
                self.config,
                self.cost_config,
                [],
                self.policies,
            )

    def test_requires_policies(self):
        with self.assertRaises(ValueError):
            evaluate_policies(
                self.config,
                self.cost_config,
                self.price_paths,
                {},
            )


if __name__ == "__main__":
    unittest.main()