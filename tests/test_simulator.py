import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.environment.costs import ExecutionCostConfig


class ExecutionEnvironmentTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.environment = ExecutionEnvironment(self.config)
        self.prices = [100.0, 99.0, 101.0]
        
        self.cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        self.cost_environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

    def test_reset_returns_initial_observation(self):
        observation = self.environment.reset(self.prices)

        self.assertEqual(observation["time_step"], 0)
        self.assertEqual(observation["remaining_lots"], 4)
        self.assertEqual(observation["market_price"], 100.0)

    def test_step_updates_inventory_and_time(self):
        self.environment.reset(self.prices)

        observation, reward, done = self.environment.step(1)

        self.assertEqual(self.environment.inventory.remaining_shares, 300)
        self.assertEqual(self.environment.total_proceeds, 10_000)
        self.assertEqual(observation["time_step"], 1)
        self.assertEqual(observation["market_price"], 99.0)
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

    def test_early_completion(self):
        self.environment.reset(self.prices)

        observation, reward, done = self.environment.step(4)

        self.assertTrue(done)
        self.assertIsNone(observation)
        self.assertEqual(self.environment.inventory.remaining_shares, 0)
        self.assertEqual(reward, 0.0)

    def test_remaining_inventory_is_sold_at_deadline(self):
        self.environment.reset(self.prices)

        self.environment.step(1)
        self.environment.step(0)
        observation, reward, done = self.environment.step(0)

        self.assertTrue(done)
        self.assertIsNone(observation)
        self.assertEqual(self.environment.inventory.remaining_shares, 0)
        self.assertEqual(self.environment.total_proceeds, 40_300)
        self.assertAlmostEqual(reward, 75.0)

    def test_reward_matches_implementation_shortfall(self):
        self.environment.reset(self.prices)

        total_reward = 0.0

        for action in [1, 0, 0]:
            _, reward, _ = self.environment.step(action)
            total_reward += reward

        self.assertAlmostEqual(
            total_reward,
            -self.environment.implementation_shortfall_bps,
        )

    def test_invalid_action_preserves_state(self):
        self.environment.reset(self.prices)

        with self.assertRaises(ValueError):
            self.environment.step(5)

        self.assertEqual(self.environment.current_step, 0)
        self.assertEqual(self.environment.inventory.remaining_lots, 4)
        self.assertEqual(self.environment.total_proceeds, 0.0)

    def test_cannot_step_before_reset(self):
        with self.assertRaises(RuntimeError):
            self.environment.step(1)

    def test_cannot_step_after_completion(self):
        self.environment.reset(self.prices)
        self.environment.step(4)

        with self.assertRaises(RuntimeError):
            self.environment.step(1)

    def test_reset_restores_episode_state(self):
        self.environment.reset(self.prices)
        self.environment.step(2)

        observation = self.environment.reset(self.prices)

        self.assertEqual(self.environment.inventory.remaining_lots, 4)
        self.assertEqual(self.environment.total_proceeds, 0.0)
        self.assertEqual(self.environment.current_step, 0)
        self.assertFalse(self.environment.is_done)
        self.assertEqual(observation["market_price"], 100.0)

    def test_invalid_price_path_length(self):
        with self.assertRaises(ValueError):
            self.environment.reset([100.0, 101.0])

    def test_invalid_market_price(self):
        with self.assertRaises(ValueError):
            self.environment.reset([100.0, 0.0, 101.0])

        with self.assertRaises(ValueError):
            self.environment.reset([100.0, float("nan"), 101.0])
            
    
    def test_trade_costs_reduce_proceeds_and_reward(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])

        observation, reward, done = self.cost_environment.step(1)

        self.assertAlmostEqual(self.cost_environment.total_proceeds, 9989.0)
        self.assertAlmostEqual(self.cost_environment.total_fees, 1.0)
        self.assertAlmostEqual(reward, -2.75)
        self.assertEqual(self.cost_environment.inventory.remaining_shares, 300)
        self.assertFalse(done)

    def test_zero_trade_has_no_execution_cost(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])

        observation, reward, done = self.cost_environment.step(0)

        self.assertEqual(self.cost_environment.total_proceeds, 0.0)
        self.assertEqual(self.cost_environment.total_fees, 0.0)
        self.assertEqual(reward, 0.0)
        self.assertEqual(self.cost_environment.inventory.remaining_shares, 400)
        self.assertFalse(done)

    def test_deadline_trade_charges_costs_on_remaining_inventory(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])

        self.cost_environment.step(1)
        self.cost_environment.step(0)
        observation, reward, done = self.cost_environment.step(0)

        self.assertTrue(done)
        self.assertIsNone(observation)
        self.assertEqual(self.cost_environment.inventory.remaining_shares, 0)
        self.assertAlmostEqual(self.cost_environment.total_proceeds, 39926.0)
        self.assertAlmostEqual(self.cost_environment.total_fees, 4.0)

    def test_early_completion_includes_execution_costs(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])

        observation, reward, done = self.cost_environment.step(4)

        self.assertTrue(done)
        self.assertIsNone(observation)
        self.assertAlmostEqual(self.cost_environment.total_proceeds, 39896.0)
        self.assertAlmostEqual(self.cost_environment.total_fees, 4.0)
        self.assertAlmostEqual(reward, -26.0)

    def test_rewards_reconcile_with_shortfall_with_costs(self):
        self.cost_environment.reset(self.prices)

        total_reward = 0.0

        for action in [1, 0, 0]:
            _, reward, _ = self.cost_environment.step(action)
            total_reward += reward

        self.assertAlmostEqual(
            total_reward,
            -self.cost_environment.implementation_shortfall_bps,
        )

    def test_reset_clears_execution_cost_totals(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])
        self.cost_environment.step(4)

        self.cost_environment.reset(self.prices)

        self.assertEqual(self.cost_environment.total_proceeds, 0.0)
        self.assertEqual(self.cost_environment.total_fees, 0.0)
        self.assertEqual(self.cost_environment.inventory.remaining_shares, 400)


    def test_execution_history_starts_empty(self):
        self.environment.reset(self.prices)

        self.assertEqual(len(self.environment.execution_history), 0)

    def test_execution_history_records_trade(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])
        self.cost_environment.step(1)

        record = self.cost_environment.execution_history[0]

        self.assertEqual(record.time_step, 0)
        self.assertEqual(record.requested_lots, 1)
        self.assertEqual(record.shares_sold, 100)
        self.assertEqual(record.forced_shares, 0)
        self.assertEqual(record.remaining_shares, 300)

        self.assertAlmostEqual(record.market_price, 100.0)
        self.assertAlmostEqual(record.execution_price, 99.90)
        self.assertAlmostEqual(record.fees, 1.0)
        self.assertAlmostEqual(record.net_proceeds, 9989.0)
        self.assertAlmostEqual(record.reward_bps, -2.75)

    def test_execution_history_records_zero_trade(self):
        self.environment.reset(self.prices)
        self.environment.step(0)

        record = self.environment.execution_history[0]

        self.assertEqual(record.requested_lots, 0)
        self.assertEqual(record.shares_sold, 0)
        self.assertEqual(record.net_proceeds, 0.0)
        self.assertEqual(record.remaining_shares, 400)

    def test_execution_history_records_forced_liquidation(self):
        self.environment.reset(self.prices)

        self.environment.step(1)
        self.environment.step(0)
        self.environment.step(0)

        final_record = self.environment.execution_history[-1]

        self.assertEqual(final_record.time_step, 2)
        self.assertEqual(final_record.requested_lots, 0)
        self.assertEqual(final_record.shares_sold, 300)
        self.assertEqual(final_record.forced_shares, 300)
        self.assertEqual(final_record.remaining_shares, 0)

    def test_reset_clears_execution_history(self):
        self.environment.reset(self.prices)
        self.environment.step(1)

        self.environment.reset(self.prices)

        self.assertEqual(len(self.environment.execution_history), 0)

    def test_execution_history_reconciles_with_totals(self):
        self.cost_environment.reset([100.0, 100.0, 100.0])

        for action in [1, 0, 0]:
            self.cost_environment.step(action)

        records = self.cost_environment.execution_history

        total_shares = sum(record.shares_sold for record in records)
        total_proceeds = sum(record.net_proceeds for record in records)
        total_fees = sum(record.fees for record in records)
        total_reward = sum(record.reward_bps for record in records)

        self.assertEqual(total_shares, self.config.total_shares)
        self.assertAlmostEqual(total_proceeds, self.cost_environment.total_proceeds)
        self.assertAlmostEqual(total_fees, self.cost_environment.total_fees)
        self.assertAlmostEqual(
            total_reward,
            -self.cost_environment.implementation_shortfall_bps,
        )

    def test_invalid_action_does_not_create_record(self):
        self.environment.reset(self.prices)

        with self.assertRaises(ValueError):
            self.environment.step(5)

        self.assertEqual(len(self.environment.execution_history), 0)


if __name__ == "__main__":
    unittest.main()