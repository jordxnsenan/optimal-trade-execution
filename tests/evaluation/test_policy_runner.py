import unittest

from functools import partial

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy
from execai.policies.baselines import immediate_action, twap_action


class PolicyRunnerTests(unittest.TestCase):

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

        self.prices = [100.0, 100.0, 100.0]

    def test_immediate_execution_completes_order(self):
        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        execution_cost = run_policy(
            environment,
            self.prices,
            immediate_action,
        )

        shares_sold = [
            record.shares_sold
            for record in environment.execution_history
        ]

        self.assertEqual(shares_sold, [400])
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertAlmostEqual(execution_cost, 26.0)

    def test_twap_distributes_execution(self):
        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        twap_policy = partial(twap_action, config=self.config)

        execution_cost = run_policy(
            environment,
            self.prices,
            twap_policy,
        )

        shares_sold = [
            record.shares_sold
            for record in environment.execution_history
        ]

        self.assertEqual(shares_sold, [200, 100, 100])
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertAlmostEqual(execution_cost, 13.5)

    def test_twap_costs_less_than_immediate_execution(self):
        immediate_environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        twap_environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        immediate_cost = run_policy(
            immediate_environment,
            self.prices,
            immediate_action,
        )

        twap_policy = partial(twap_action, config=self.config)

        twap_cost = run_policy(
            twap_environment,
            self.prices,
            twap_policy,
        )

        self.assertLess(twap_cost, immediate_cost)

    def test_runner_resets_environment_between_episodes(self):
        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        run_policy(environment, self.prices, immediate_action)

        self.assertEqual(len(environment.execution_history), 1)

        twap_policy = partial(twap_action, config=self.config)

        run_policy(environment, self.prices, twap_policy)

        self.assertEqual(len(environment.execution_history), 3)
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertAlmostEqual(environment.total_fees, 4.0)


    def test_full_size_twap_matches_flat_market_reference(self):
        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        environment = ExecutionEnvironment(
            config,
            self.cost_config,
        )

        prices = [100.0] * config.num_time_steps
        twap_policy = partial(twap_action, config=config)

        execution_cost = run_policy(
            environment,
            prices,
            twap_policy,
        )

        lots_sold = [
            record.shares_sold // config.shares_per_lot
            for record in environment.execution_history
        ]

        expected_schedule = [2] * 8 + [1] * 4

        self.assertEqual(lots_sold, expected_schedule)
        self.assertEqual(environment.inventory.remaining_shares, 0)
        self.assertAlmostEqual(execution_cost, 7.8)


if __name__ == "__main__":
    unittest.main()