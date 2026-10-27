import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.simulator import ExecutionEnvironment


class ExecutionEnvironmentTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.environment = ExecutionEnvironment(self.config)
        self.prices = [100.0, 99.0, 101.0]

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


if __name__ == "__main__":
    unittest.main()