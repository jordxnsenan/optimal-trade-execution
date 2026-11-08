import unittest

from execai.agents.dqn_scenarios import generate_training_paths
from execai.environment.config import ExecutionConfig


class DQNScenarioTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )

        self.volatility = 0.0007210663654409027

    def test_generates_requested_number_of_paths(self):
        paths = generate_training_paths(
            self.config,
            self.volatility,
            num_episodes=5,
            starting_seed=42,
        )

        self.assertEqual(len(paths), 5)

        for prices in paths:
            self.assertEqual(len(prices), 12)
            self.assertEqual(prices[0], 100.0)
            self.assertTrue(all(price > 0 for price in prices))

    def test_generation_is_reproducible(self):
        first_paths = generate_training_paths(self.config, self.volatility, 5, 42)

        second_paths = generate_training_paths(self.config, self.volatility, 5, 42)

        self.assertEqual(first_paths, second_paths)

    def test_separate_seed_ranges_produce_different_paths(self):
        training_paths = generate_training_paths(self.config, self.volatility, 5, 42)

        evaluation_paths = generate_training_paths(self.config, self.volatility, 5, 47)

        self.assertNotEqual(training_paths, evaluation_paths)

    def test_rejects_invalid_episode_count(self):
        with self.assertRaises(ValueError):
            generate_training_paths(self.config, self.volatility, 0, 42)


if __name__ == "__main__":
    unittest.main()
