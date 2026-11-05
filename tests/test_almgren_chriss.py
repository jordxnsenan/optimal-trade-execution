import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.policies.almgren_chriss import calculate_expected_cost_bps


class AlmgrenChrissTests(unittest.TestCase):

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

    def test_immediate_execution_cost(self):
        cost = calculate_expected_cost_bps(
            [4, 0, 0],
            self.config,
            self.cost_config,
            reference_price=100.0,
        )

        self.assertAlmostEqual(cost, 26.0)

    def test_evenly_distributed_execution_cost(self):
        cost = calculate_expected_cost_bps(
            [2, 1, 1],
            self.config,
            self.cost_config,
            reference_price=100.0,
        )

        self.assertAlmostEqual(cost, 13.5)

    def test_schedule_order_does_not_change_flat_market_cost(self):
        first_cost = calculate_expected_cost_bps(
            [2, 1, 1],
            self.config,
            self.cost_config,
            reference_price=100.0,
        )

        second_cost = calculate_expected_cost_bps(
            [1, 1, 2],
            self.config,
            self.cost_config,
            reference_price=100.0,
        )

        self.assertAlmostEqual(first_cost, second_cost)

    def test_rejects_incomplete_schedule(self):
        with self.assertRaisesRegex(ValueError, "entire order"):
            calculate_expected_cost_bps(
                [1, 1, 1],
                self.config,
                self.cost_config,
                reference_price=100.0,
            )

    def test_rejects_invalid_trade_quantity(self):
        with self.assertRaises(ValueError):
            calculate_expected_cost_bps(
                [2, -1, 3],
                self.config,
                self.cost_config,
                reference_price=100.0,
            )

    def test_rejects_invalid_schedule_length(self):
        with self.assertRaises(ValueError):
            calculate_expected_cost_bps(
                [2, 2],
                self.config,
                self.cost_config,
                reference_price=100.0,
            )


if __name__ == "__main__":
    unittest.main()
