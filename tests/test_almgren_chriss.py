import unittest

from itertools import product
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.policies.almgren_chriss import (
    calculate_expected_cost_bps,
    optimize_almgren_chriss_schedule,
    calculate_inventory_risk_bps_squared,
)


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

    def test_risk_neutral_optimizer_matches_known_optimum(self):
        schedule = optimize_almgren_chriss_schedule(
            self.config,
            self.cost_config,
            reference_price=100.0,
            volatility_per_step=0.000721,
            risk_aversion=0.0,
        )

        expected_cost = calculate_expected_cost_bps(
            schedule,
            self.config,
            self.cost_config,
            reference_price=100.0,
        )

        self.assertEqual(len(schedule), 3)
        self.assertEqual(sum(schedule), 4)
        self.assertAlmostEqual(expected_cost, 13.5)

    def test_full_size_optimizer_matches_known_optimum(self):
        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        schedule = optimize_almgren_chriss_schedule(
            config,
            self.cost_config,
            reference_price=100.0,
            volatility_per_step=0.000721,
            risk_aversion=0.0,
        )

        execution_cost = calculate_expected_cost_bps(
            schedule,
            config,
            self.cost_config,
            reference_price=100.0,
        )

        self.assertEqual(len(schedule), 12)
        self.assertEqual(sum(schedule), 20)
        self.assertEqual(schedule.count(2), 8)
        self.assertEqual(schedule.count(1), 4)
        self.assertAlmostEqual(execution_cost, 7.8)

    def test_risk_aversion_encourages_earlier_execution(self):
        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        risk_neutral_schedule = optimize_almgren_chriss_schedule(
            config,
            self.cost_config,
            reference_price=100.0,
            volatility_per_step=0.000721,
            risk_aversion=0.0,
        )

        risk_sensitive_schedule = optimize_almgren_chriss_schedule(
            config,
            self.cost_config,
            reference_price=100.0,
            volatility_per_step=0.000721,
            risk_aversion=0.05,
        )

        neutral_early_lots = sum(risk_neutral_schedule[:3])
        risk_sensitive_early_lots = sum(risk_sensitive_schedule[:3])

        self.assertGreater(
            risk_sensitive_early_lots,
            neutral_early_lots,
        )

        self.assertEqual(sum(risk_sensitive_schedule), 20)

    def test_rejects_invalid_optimizer_parameters(self):
        invalid_parameters = [
            {"reference_price": 0.0},
            {"volatility_per_step": -0.001},
            {"risk_aversion": -1.0},
            {"risk_aversion": float("nan")},
        ]

        valid_parameters = {
            "reference_price": 100.0,
            "volatility_per_step": 0.000721,
            "risk_aversion": 0.0,
        }

        for invalid_parameter in invalid_parameters:
            with self.subTest(parameters=invalid_parameter):
                parameters = {
                    **valid_parameters,
                    **invalid_parameter,
                }

                with self.assertRaises(ValueError):
                    optimize_almgren_chriss_schedule(
                        self.config,
                        self.cost_config,
                        **parameters,
                    )

    def test_risk_sensitive_optimizer_matches_exhaustive_search(self):
        reference_price = 100.0
        volatility_per_step = 0.000721
        risk_aversion = 0.5

        optimized_schedule = optimize_almgren_chriss_schedule(
            self.config,
            self.cost_config,
            reference_price,
            volatility_per_step,
            risk_aversion,
        )

        arrival_notional = self.config.total_shares * reference_price
        volatility_bps = volatility_per_step * 10_000

        def calculate_reference_objective(schedule):
            """calculates the objective independently of the optimizer"""

            remaining_shares = self.config.total_shares
            total_objective = 0.0

            for lots_to_sell in schedule:
                shares_sold = lots_to_sell * self.config.shares_per_lot
                trade_fraction = shares_sold / self.config.total_shares

                # calculates the modeled execution costs
                spread_cost_bps = self.cost_config.half_spread_bps * trade_fraction

                impact_cost_bps = (
                    self.cost_config.temporary_impact_bps * trade_fraction**2
                )

                fee_cost_bps = (
                    shares_sold
                    * self.cost_config.fee_per_share
                    / arrival_notional
                    * 10_000
                )

                remaining_shares -= shares_sold
                remaining_fraction = remaining_shares / self.config.total_shares

                risk_cost_bps = (
                    risk_aversion * volatility_bps**2 * remaining_fraction**2
                )

                total_objective += (
                    spread_cost_bps + impact_cost_bps + fee_cost_bps + risk_cost_bps
                )

            return total_objective

        possible_actions = range(self.config.num_lots + 1)
        lowest_reference_objective = float("inf")
        complete_schedules = 0

        for schedule in product(
            possible_actions,
            repeat=self.config.num_time_steps,
        ):
            if sum(schedule) != self.config.num_lots:
                continue

            complete_schedules += 1

            objective = calculate_reference_objective(schedule)

            lowest_reference_objective = min(
                lowest_reference_objective,
                objective,
            )

        optimized_objective = calculate_reference_objective(optimized_schedule)

        self.assertEqual(complete_schedules, 15)

        self.assertEqual(
            sum(optimized_schedule),
            self.config.num_lots,
        )

        self.assertAlmostEqual(
            optimized_objective,
            lowest_reference_objective,
        )

    def test_inventory_risk_matches_manual_calculation(self):
        schedule = [1, 1, 2]
        volatility = 0.001

        risk = calculate_inventory_risk_bps_squared(
            schedule,
            self.config,
            volatility,
        )

        # uses ten basis points of volatility per step. the remaining inventory fractions are 3/4, 2/4, 0
        expected_risk = 100 * ((3 / 4) ** 2 + (2 / 4) ** 2)

        self.assertAlmostEqual(risk, expected_risk)

    def test_immediate_execution_has_no_inventory_risk(self):
        risk = calculate_inventory_risk_bps_squared(
            [4, 0, 0],
            self.config,
            volatility_per_step=0.001,
        )

        self.assertEqual(risk, 0.0)

    def test_delaying_execution_increases_inventory_risk(self):
        early_schedule = [2, 1, 1]
        delayed_schedule = [0, 0, 4]

        early_risk = calculate_inventory_risk_bps_squared(
            early_schedule,
            self.config,
            volatility_per_step=0.001,
        )

        delayed_risk = calculate_inventory_risk_bps_squared(
            delayed_schedule,
            self.config,
            volatility_per_step=0.001,
        )

        self.assertGreater(delayed_risk, early_risk)

    def test_risk_sensitive_optimizer_reduces_inventory_risk(self):
        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        volatility = 0.000721

        neutral_schedule = optimize_almgren_chriss_schedule(
            config,
            self.cost_config,
            reference_price=100.0,
            volatility_per_step=volatility,
            risk_aversion=0.0,
        )

        sensitive_schedule = optimize_almgren_chriss_schedule(
            config,
            self.cost_config,
            reference_price=100.0,
            volatility_per_step=volatility,
            risk_aversion=0.05,
        )

        neutral_risk = calculate_inventory_risk_bps_squared(
            neutral_schedule,
            config,
            volatility,
        )

        sensitive_risk = calculate_inventory_risk_bps_squared(
            sensitive_schedule,
            config,
            volatility,
        )

        self.assertLess(sensitive_risk, neutral_risk)

    def test_inventory_risk_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            calculate_inventory_risk_bps_squared(
                [1, 1, 1],
                self.config,
                volatility_per_step=0.001,
            )

        with self.assertRaises(ValueError):
            calculate_inventory_risk_bps_squared(
                [2, 1, 1],
                self.config,
                volatility_per_step=-0.001,
            )


if __name__ == "__main__":
    unittest.main()
