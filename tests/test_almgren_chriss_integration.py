import unittest

from functools import partial

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy
from execai.policies.almgren_chriss import (
    optimize_almgren_chriss_schedule,
)
from execai.policies.schedules import scheduled_action


class AlmgrenChrissIntegrationTests(unittest.TestCase):

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

        self.reference_price = 100.0
        self.volatility_per_step = 0.000721
        self.prices = [self.reference_price] * self.config.num_time_steps

    def test_risk_neutral_schedule_executes_correctly(self):
        schedule = optimize_almgren_chriss_schedule(
            self.config,
            self.cost_config,
            self.reference_price,
            self.volatility_per_step,
            risk_aversion=0.0,
        )

        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        policy = partial(scheduled_action, schedule=schedule)

        execution_cost = run_policy(
            environment,
            self.prices,
            policy,
        )

        executed_lots = [
            record.shares_sold // self.config.shares_per_lot
            for record in environment.execution_history
        ]

        self.assertEqual(executed_lots, schedule)
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertTrue(environment.is_done)
        self.assertAlmostEqual(execution_cost, 7.8)

    def test_risk_sensitive_schedule_executes_correctly(self):
        schedule = optimize_almgren_chriss_schedule(
            self.config,
            self.cost_config,
            self.reference_price,
            self.volatility_per_step,
            risk_aversion=0.05,
        )

        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        execution_cost = run_policy(
            environment,
            self.prices,
            partial(scheduled_action, schedule=schedule),
        )

        total_shares_sold = sum(
            record.shares_sold for record in environment.execution_history
        )

        total_reward = sum(
            record.reward_bps for record in environment.execution_history
        )

        self.assertEqual(total_shares_sold, self.config.total_shares)
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertAlmostEqual(total_reward, -execution_cost)

    def test_scheduled_action_rejects_invalid_inventory(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 1,
        }

        schedule = [2] + [0] * 11

        with self.assertRaisesRegex(ValueError, "available inventory"):
            scheduled_action(observation, schedule)

    def test_scheduled_action_rejects_invalid_time_step(self):
        observation = {
            "time_step": 12,
            "remaining_lots": 20,
        }

        schedule = [2] * 8 + [1] * 4

        with self.assertRaisesRegex(ValueError, "outside"):
            scheduled_action(observation, schedule)


if __name__ == "__main__":
    unittest.main()
