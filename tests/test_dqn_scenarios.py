import unittest
import math

from execai.agents.dqn_scenarios import (
    generate_training_paths,
    generate_liquidity_profiles,
)
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

    def test_generates_variable_liquidity_profiles(self):
        expected_profile = [1.0] * self.config.num_time_steps

        profiles = generate_liquidity_profiles(
            self.config,
            expected_profile,
            num_episodes=5,
            starting_seed=42,
        )

        self.assertEqual(len(profiles), 5)

        for profile in profiles:
            self.assertEqual(len(profile), self.config.num_time_steps)
            self.assertTrue(
                all(math.isfinite(value) and value > 0 for value in profile)
            )

        self.assertNotEqual(profiles[0], profiles[1])

    def test_liquidity_generation_is_reproducible(self):
        expected_profile = [1.0] * self.config.num_time_steps

        first_profiles = generate_liquidity_profiles(
            self.config,
            expected_profile,
            num_episodes=5,
            starting_seed=42,
        )

        second_profiles = generate_liquidity_profiles(
            self.config,
            expected_profile,
            num_episodes=5,
            starting_seed=42,
        )

        self.assertEqual(first_profiles, second_profiles)

    def test_zero_noise_preserves_expected_liquidity(self):
        expected_profile = [1.2, 0.8] * 6

        profiles = generate_liquidity_profiles(
            self.config,
            expected_profile,
            num_episodes=3,
            starting_seed=42,
            noise_std=0.0,
        )

        for profile in profiles:
            self.assertEqual(profile, expected_profile)

    def test_future_forecast_does_not_change_current_liquidity(self):
        first_profile = [1.0] * self.config.num_time_steps
        second_profile = first_profile.copy()

        second_profile[-1] = 2.0

        first_result = generate_liquidity_profiles(
            self.config,
            first_profile,
            num_episodes=1,
            starting_seed=42,
        )

        second_result = generate_liquidity_profiles(
            self.config,
            second_profile,
            num_episodes=1,
            starting_seed=42,
        )

        self.assertEqual(first_result[0][0], second_result[0][0])

    def test_rejects_invalid_liquidity_settings(self):
        expected_profile = [1.0] * self.config.num_time_steps

        with self.assertRaises(ValueError):
            generate_liquidity_profiles(
                self.config,
                expected_profile,
                num_episodes=0,
                starting_seed=42,
            )

        with self.assertRaises(ValueError):
            generate_liquidity_profiles(
                self.config,
                expected_profile[:-1],
                num_episodes=5,
                starting_seed=42,
            )

        with self.assertRaises(ValueError):
            generate_liquidity_profiles(
                self.config,
                expected_profile,
                num_episodes=5,
                starting_seed=42,
                noise_std=-0.1,
            )

        with self.assertRaises(ValueError):
            generate_liquidity_profiles(
                self.config,
                expected_profile,
                num_episodes=5,
                starting_seed=42,
                noise_std=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
