import math
import unittest

from itertools import product

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment


BASIS_POINTS_MULTIPLIER = 10_000


class ExactSolutionTests(unittest.TestCase):

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

        self.market_price = 100.0
        self.prices = [self.market_price] * self.config.num_time_steps

    def get_complete_schedules(self):
        """generates every schedule that sells all inventory"""

        possible_actions = range(self.config.num_lots + 1)

        for schedule in product(
            possible_actions,
            repeat=self.config.num_time_steps,
        ):
            if sum(schedule) == self.config.num_lots:
                yield schedule

    def calculate_reference_cost(self, schedule):
        """calculates execution cost independently of the simulator"""

        total_cost_dollars = 0.0

        for num_lots in schedule:
            shares_sold = num_lots * self.config.shares_per_lot
            order_fraction = shares_sold / self.config.total_shares

            spread_cost = (
                shares_sold
                * self.market_price
                * self.cost_config.half_spread_bps
                / BASIS_POINTS_MULTIPLIER
            )

            impact_cost = (
                shares_sold
                * self.market_price
                * self.cost_config.temporary_impact_bps
                * order_fraction
                / BASIS_POINTS_MULTIPLIER
            )

            transaction_fees = (
                shares_sold * self.cost_config.fee_per_share
            )

            total_cost_dollars += (
                spread_cost + impact_cost + transaction_fees
            )

        arrival_notional = (
            self.config.total_shares * self.market_price
        )

        return (
            total_cost_dollars
            / arrival_notional
            * BASIS_POINTS_MULTIPLIER
        )

    def test_exhaustive_search_finds_optimal_schedule(self):
        schedule_costs = {}

        for schedule in self.get_complete_schedules():
            schedule_costs[schedule] = self.calculate_reference_cost(
                schedule
            )

        self.assertEqual(len(schedule_costs), 15)

        minimum_cost = min(schedule_costs.values())

        optimal_schedules = {
            schedule
            for schedule, cost in schedule_costs.items()
            if math.isclose(cost, minimum_cost, abs_tol=1e-9)
        }

        self.assertAlmostEqual(minimum_cost, 13.5)

        self.assertEqual(
            optimal_schedules,
            {
                (1, 1, 2),
                (1, 2, 1),
                (2, 1, 1),
            },
        )

    def test_simulator_matches_reference_costs(self):
        for schedule in self.get_complete_schedules():
            with self.subTest(schedule=schedule):
                environment = ExecutionEnvironment(
                    self.config,
                    self.cost_config,
                )

                environment.reset(self.prices)

                for action in schedule:
                    if environment.is_done:
                        break

                    environment.step(action)

                expected_cost = self.calculate_reference_cost(
                    schedule
                )

                self.assertTrue(environment.is_done)
                self.assertEqual(
                    environment.inventory.remaining_shares,
                    0,
                )

                self.assertAlmostEqual(
                    environment.implementation_shortfall_bps,
                    expected_cost,
                )

    def test_forced_liquidation_matches_reference(self):
        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        environment.reset(self.prices)

        environment.step(1)
        environment.step(1)
        environment.step(0)

        expected_cost = self.calculate_reference_cost((1, 1, 2))

        self.assertEqual(
            environment.execution_history[-1].forced_shares,
            200,
        )

        self.assertAlmostEqual(
            environment.implementation_shortfall_bps,
            expected_cost,
        )


if __name__ == "__main__":
    unittest.main()