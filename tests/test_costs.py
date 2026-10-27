import unittest

from execai.environment.costs import (
    ExecutionCostConfig,
    calculate_sell_execution,
)


class ExecutionCostTests(unittest.TestCase):

    def setUp(self):
        self.cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

    def test_zero_cost_configuration(self):
        cost_config = ExecutionCostConfig()

        execution = calculate_sell_execution(
            100.0, 100, 400, cost_config
        )

        self.assertEqual(execution.execution_price, 100.0)
        self.assertEqual(execution.fees, 0.0)
        self.assertEqual(execution.net_proceeds, 10_000.0)

    def test_execution_price_includes_spread_and_impact(self):
        execution = calculate_sell_execution(
            100.0, 100, 400, self.cost_config
        )

        self.assertAlmostEqual(execution.execution_price, 99.90)

    def test_fees_reduce_net_proceeds(self):
        execution = calculate_sell_execution(
            100.0, 100, 400, self.cost_config
        )

        self.assertAlmostEqual(execution.fees, 1.0)
        self.assertAlmostEqual(execution.net_proceeds, 9989.0)

    def test_larger_trades_have_greater_impact(self):
        small_trade = calculate_sell_execution(
            100.0, 100, 400, self.cost_config
        )

        large_trade = calculate_sell_execution(
            100.0, 400, 400, self.cost_config
        )

        self.assertLess(
            large_trade.execution_price,
            small_trade.execution_price,
        )

    def test_selling_zero_shares_has_no_cost(self):
        execution = calculate_sell_execution(
            100.0, 0, 400, self.cost_config
        )

        self.assertEqual(execution.shares_sold, 0)
        self.assertEqual(execution.fees, 0.0)
        self.assertEqual(execution.net_proceeds, 0.0)

    def test_invalid_cost_configuration(self):
        with self.assertRaises(ValueError):
            ExecutionCostConfig(half_spread_bps=-1.0)

        with self.assertRaises(ValueError):
            ExecutionCostConfig(temporary_impact_bps=-1.0)

        with self.assertRaises(ValueError):
            ExecutionCostConfig(fee_per_share=-0.01)

        with self.assertRaises(ValueError):
            ExecutionCostConfig(half_spread_bps=10_000.0)

    def test_invalid_trade_quantity(self):
        with self.assertRaises(ValueError):
            calculate_sell_execution(
                100.0, -1, 400, self.cost_config
            )

        with self.assertRaises(ValueError):
            calculate_sell_execution(
                100.0, 500, 400, self.cost_config
            )

        with self.assertRaises(ValueError):
            calculate_sell_execution(
                100.0, 1.5, 400, self.cost_config
            )

    def test_invalid_market_price(self):
        with self.assertRaises(ValueError):
            calculate_sell_execution(
                0.0, 100, 400, self.cost_config
            )


if __name__ == "__main__":
    unittest.main()