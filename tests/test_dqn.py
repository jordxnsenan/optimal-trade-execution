import unittest
import torch

from execai.agents.dqn import (
    DQNAgent,
    QNetwork,
    encode_observation,
)
from execai.environment.config import ExecutionConfig
from unittest.mock import patch


class DQNTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

    def test_encodes_normalized_state(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 2,
        }

        state = encode_observation(observation, self.config)

        self.assertEqual(state.dtype, torch.float32)

        torch.testing.assert_close(
            state,
            torch.tensor([0.5, 0.5]),
        )

    def test_encoding_does_not_use_market_price(self):
        first_observation = {
            "time_step": 1,
            "remaining_lots": 2,
            "market_price": 100.0,
        }

        second_observation = {
            "time_step": 1,
            "remaining_lots": 2,
            "market_price": 105.0,
        }

        first_state = encode_observation(first_observation, self.config)
        second_state = encode_observation(second_observation, self.config)

        torch.testing.assert_close(first_state, second_state)

    def test_rejects_invalid_observation(self):
        observation = {
            "time_step": 3,
            "remaining_lots": 2,
        }

        with self.assertRaises(ValueError):
            encode_observation(observation, self.config)

    def test_network_returns_one_value_per_action(self):
        network = QNetwork(num_lots=self.config.num_lots)

        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        state = encode_observation(observation, self.config)

        with torch.no_grad():
            q_values = network(state)

        self.assertEqual(q_values.shape, (5,))
        self.assertTrue(torch.isfinite(q_values).all().item())

    def test_network_supports_batches(self):
        network = QNetwork(num_lots=self.config.num_lots)

        observations = [
            {"time_step": 0, "remaining_lots": 4},
            {"time_step": 1, "remaining_lots": 2},
        ]

        states = torch.stack(
            [
                encode_observation(observation, self.config)
                for observation in observations
            ]
        )

        with torch.no_grad():
            q_values = network(states)

        self.assertEqual(q_values.shape, (2, 5))

    def test_full_size_network_has_twenty_one_actions(self):
        network = QNetwork(num_lots=20)

        states = torch.zeros((2, 2), dtype=torch.float32)

        with torch.no_grad():
            q_values = network(states)

        self.assertEqual(q_values.shape, (2, 21))

    def test_rejects_invalid_network_configuration(self):
        with self.assertRaises(ValueError):
            QNetwork(num_lots=0)

        with self.assertRaises(ValueError):
            QNetwork(num_lots=4, hidden_units=0)


class DQNActionSelectionTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.agent = DQNAgent(self.config, seed=42)

    def test_greedy_selection_chooses_highest_q_value(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        predicted_values = torch.tensor([1.0, 2.0, 5.0, 3.0, 0.0])

        with patch.object(
            self.agent.network,
            "forward",
            return_value=predicted_values,
        ):
            action = self.agent.select_action(observation)

        self.assertEqual(action, 2)

    def test_greedy_selection_ignores_invalid_actions(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 2,
        }

        # actions 3 and 4 have high values but are invalid
        predicted_values = torch.tensor([0.0, 5.0, 4.0, 100.0, 50.0])

        with patch.object(
            self.agent.network,
            "forward",
            return_value=predicted_values,
        ):
            action = self.agent.select_action(observation)

        self.assertEqual(action, 1)

    def test_final_step_requires_remaining_inventory(self):
        observation = {
            "time_step": 2,
            "remaining_lots": 3,
        }

        for exploration_rate in (0.0, 1.0):
            action = self.agent.select_action(
                observation,
                exploration_rate=exploration_rate,
            )

            self.assertEqual(action, 3)

    def test_exploration_is_reproducible(self):
        first_agent = DQNAgent(self.config, seed=42)
        second_agent = DQNAgent(self.config, seed=42)

        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        first_actions = [
            first_agent.select_action(observation, exploration_rate=1.0)
            for _ in range(20)
        ]

        second_actions = [
            second_agent.select_action(observation, exploration_rate=1.0)
            for _ in range(20)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertGreater(len(set(first_actions)), 1)

    def test_exploration_respects_remaining_inventory(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 2,
        }

        for _ in range(100):
            action = self.agent.select_action(
                observation,
                exploration_rate=1.0,
            )

            self.assertIn(action, [0, 1, 2])

    def test_invalid_exploration_rate(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        invalid_rates = [
            -0.1,
            1.1,
            float("nan"),
            float("inf"),
            True,
        ]

        for exploration_rate in invalid_rates:
            with self.subTest(exploration_rate=exploration_rate):
                with self.assertRaises(ValueError):
                    self.agent.select_action(
                        observation,
                        exploration_rate=exploration_rate,
                    )


if __name__ == "__main__":
    unittest.main()
