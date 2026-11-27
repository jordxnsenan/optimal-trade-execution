import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment


class ExecutionInvariantTests(unittest.TestCase):

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

        self.prices = [100.0, 99.0, 101.0]

    def test_inventory_conservation_across_schedules(self):
        execution_schedules = [
            [4],
            [1, 1, 2],
            [0, 0, 0],
            [0, 1, 0],
            [2, 0, 2],
        ]

        for schedule in execution_schedules:
            with self.subTest(schedule=schedule):
                environment = ExecutionEnvironment(
                    self.config,
                    self.cost_config,
                )

                environment.reset(self.prices)

                remaining_shares = self.config.total_shares
                total_reward = 0.0

                for action in schedule:
                    _, reward, is_done = environment.step(action)
                    total_reward += reward

                    record = environment.execution_history[-1]

                    remaining_shares -= record.shares_sold

                    self.assertEqual(
                        record.remaining_shares,
                        remaining_shares,
                    )

                    self.assertGreaterEqual(remaining_shares, 0)

                    if is_done:
                        break

                self.assertTrue(environment.is_done)
                self.assertEqual(remaining_shares, 0)

                records = environment.execution_history

                total_shares_sold = sum(
                    record.shares_sold for record in records
                )

                self.assertEqual(
                    total_shares_sold,
                    self.config.total_shares,
                )

                self.assertAlmostEqual(
                    total_reward,
                    -environment.implementation_shortfall_bps,
                )

    def test_execution_records_match_financial_totals(self):
        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        environment.reset(self.prices)

        for action in [1, 0, 0]:
            environment.step(action)

        records = environment.execution_history

        recorded_proceeds = sum(
            record.net_proceeds for record in records
        )

        recorded_fees = sum(
            record.fees for record in records
        )

        arrival_notional = self.config.total_shares * self.prices[0]

        expected_shortfall_bps = (
            (arrival_notional - recorded_proceeds)
            / arrival_notional
            * 10_000
        )

        self.assertAlmostEqual(
            recorded_proceeds,
            environment.total_proceeds,
        )

        self.assertAlmostEqual(
            recorded_fees,
            environment.total_fees,
        )

        self.assertAlmostEqual(
            expected_shortfall_bps,
            environment.implementation_shortfall_bps,
        )


if __name__ == "__main__":
    unittest.main()