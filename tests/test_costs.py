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

        execution = calculate_sell_execution(100.0, 100, 400, cost_config)

        self.assertEqual(execution.execution_price, 100.0)
        self.assertEqual(execution.fees, 0.0)
        self.assertEqual(execution.net_proceeds, 10_000.0)

    def test_execution_price_includes_spread_and_impact(self):
        execution = calculate_sell_execution(100.0, 100, 400, self.cost_config)

        self.assertAlmostEqual(execution.execution_price, 99.90)

    def test_fees_reduce_net_proceeds(self):
        execution = calculate_sell_execution(100.0, 100, 400, self.cost_config)

        self.assertAlmostEqual(execution.fees, 1.0)
        self.assertAlmostEqual(execution.net_proceeds, 9989.0)

    def test_larger_trades_have_greater_impact(self):
        small_trade = calculate_sell_execution(100.0, 100, 400, self.cost_config)

        large_trade = calculate_sell_execution(100.0, 400, 400, self.cost_config)

        self.assertLess(
            large_trade.execution_price,
            small_trade.execution_price,
        )

    def test_selling_zero_shares_has_no_cost(self):
        execution = calculate_sell_execution(100.0, 0, 400, self.cost_config)

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
            calculate_sell_execution(100.0, -1, 400, self.cost_config)

        with self.assertRaises(ValueError):
            calculate_sell_execution(100.0, 500, 400, self.cost_config)

        with self.assertRaises(ValueError):
            calculate_sell_execution(100.0, 1.5, 400, self.cost_config)

    def test_invalid_market_price(self):
        with self.assertRaises(ValueError):
            calculate_sell_execution(0.0, 100, 400, self.cost_config)

    def test_default_liquidity_preserves_existing_costs(self):
        execution = calculate_sell_execution(
            100.0,
            100,
            400,
            self.cost_config,
        )

        self.assertAlmostEqual(execution.execution_price, 99.90)
        self.assertAlmostEqual(execution.net_proceeds, 9989.0)

    def test_lower_liquidity_increases_market_impact(self):
        execution = calculate_sell_execution(
            100.0,
            100,
            400,
            self.cost_config,
            relative_liquidity=0.5,
        )

        self.assertAlmostEqual(execution.execution_price, 99.85)
        self.assertAlmostEqual(execution.fees, 1.0)
        self.assertAlmostEqual(execution.net_proceeds, 9984.0)

    def test_higher_liquidity_reduces_market_impact(self):
        execution = calculate_sell_execution(
            100.0,
            100,
            400,
            self.cost_config,
            relative_liquidity=2.0,
        )

        self.assertAlmostEqual(execution.execution_price, 99.925)
        self.assertAlmostEqual(execution.fees, 1.0)
        self.assertAlmostEqual(execution.net_proceeds, 9991.5)

    def test_invalid_relative_liquidity(self):
        invalid_liquidity_values = [
            0.0,
            -1.0,
            float("nan"),
            float("inf"),
            True,
        ]

        for liquidity in invalid_liquidity_values:
            with self.subTest(liquidity=liquidity):
                with self.assertRaisesRegex(
                    ValueError,
                    "Relative liquidity",
                ):
                    calculate_sell_execution(
                        100.0,
                        100,
                        400,
                        self.cost_config,
                        relative_liquidity=liquidity,
                    )

    def test_extremely_low_liquidity_rejects_invalid_price(self):
        with self.assertRaisesRegex(ValueError, "discount"):
            calculate_sell_execution(
                100.0,
                400,
                400,
                self.cost_config,
                relative_liquidity=0.001,
            )

    def test_default_impact_model_is_linear(self):
        cost_config = ExecutionCostConfig(
            temporary_impact_bps=20.0,
        )

        self.assertEqual(cost_config.impact_model, "linear")

        execution = calculate_sell_execution(
            100.0,
            100,
            400,
            cost_config,
        )

        self.assertAlmostEqual(execution.execution_price, 99.95)

    def test_square_root_impact_calculation(self):
        cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
            impact_model="square_root",
        )

        execution = calculate_sell_execution(
            100.0,
            100,
            400,
            cost_config,
        )

        # sqrt(100 / 400) = 0.5
        # temp. impact is 10 basis points
        self.assertAlmostEqual(execution.execution_price, 99.85)
        self.assertAlmostEqual(execution.fees, 1.0)
        self.assertAlmostEqual(execution.net_proceeds, 9984.0)

    def test_square_root_impact_respects_liquidity(self):
        cost_config = ExecutionCostConfig(
            temporary_impact_bps=20.0,
            impact_model="square_root",
        )

        low_liquidity = calculate_sell_execution(
            100.0,
            100,
            400,
            cost_config,
            relative_liquidity=0.5,
        )

        high_liquidity = calculate_sell_execution(
            100.0,
            100,
            400,
            cost_config,
            relative_liquidity=2.0,
        )

        self.assertLess(
            low_liquidity.execution_price,
            high_liquidity.execution_price,
        )

    def test_rejects_unknown_impact_model(self):
        with self.assertRaisesRegex(
            ValueError,
            "Unsupported market impact model",
        ):
            ExecutionCostConfig(
                impact_model="unknown",
            )


if __name__ == "__main__":
    unittest.main()
