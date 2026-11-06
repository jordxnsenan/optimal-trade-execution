import unittest

from functools import partial

from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy
from execai.environment.config import ExecutionConfig
from execai.policies.vwap import build_vwap_schedule, vwap_action


class VWAPTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        self.volume_fractions = [1.0] * 78

    def test_schedule_executes_complete_order(self):
        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        self.assertEqual(len(schedule), 12)
        self.assertEqual(sum(schedule), 20)
        self.assertTrue(all(lots >= 0 for lots in schedule))

    def test_uniform_volume_distributes_trades_evenly(self):
        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        self.assertLessEqual(max(schedule) - min(schedule), 1)

    def test_higher_forecast_volume_receives_more_inventory(self):
        self.volume_fractions[6] = 10.0

        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        self.assertGreater(schedule[0], schedule[1])
        self.assertEqual(sum(schedule), 20)

    def test_volume_outside_execution_window_does_not_affect_schedule(self):
        original_schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        self.volume_fractions[77] = 1000.0

        updated_schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        self.assertEqual(original_schedule, updated_schedule)

    def test_rejects_zero_volume_in_execution_window(self):
        self.volume_fractions[6:18] = [0.0] * 12

        with self.assertRaisesRegex(ValueError, "positive forecast volume"):
            build_vwap_schedule(
                self.config,
                self.volume_fractions,
            )

    def test_vwap_action_follows_schedule(self):
        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        observation = {
            "time_step": 0,
            "remaining_lots": 20,
        }

        action = vwap_action(observation, schedule)

        self.assertEqual(action, schedule[0])

    def test_vwap_action_respects_remaining_inventory(self):
        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        observation = {
            "time_step": 0,
            "remaining_lots": 1,
        }

        with self.assertRaisesRegex(ValueError, "available inventory"):
            vwap_action(observation, schedule)

    def test_vwap_action_rejects_invalid_time_step(self):
        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        observation = {
            "time_step": 12,
            "remaining_lots": 20,
        }

        with self.assertRaisesRegex(ValueError, "outside"):
            vwap_action(observation, schedule)

    def test_vwap_completes_execution_episode(self):
        cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        environment = ExecutionEnvironment(
            self.config,
            cost_config,
        )

        schedule = build_vwap_schedule(
            self.config,
            self.volume_fractions,
        )

        vwap_policy = partial(
            vwap_action,
            schedule=schedule,
        )

        prices = [100.0] * self.config.num_time_steps

        execution_cost = run_policy(
            environment,
            prices,
            vwap_policy,
        )

        executed_lots = [
            record.shares_sold // self.config.shares_per_lot
            for record in environment.execution_history
        ]

        self.assertEqual(executed_lots, schedule)
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertTrue(environment.is_done)
        self.assertAlmostEqual(execution_cost, 7.8)


if __name__ == "__main__":
    unittest.main()
