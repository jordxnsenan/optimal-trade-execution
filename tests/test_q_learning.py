import unittest

from execai.environment.config import ExecutionConfig
from execai.agents.q_learning import (
    QLearningAgent,
    get_state,
    get_valid_actions,
)


class QLearningStateTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

    def test_state_representation(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 3,
            "market_price": 100.0,
        }

        state = get_state(observation)

        self.assertEqual(state, (1, 3))

    def test_state_does_not_depend_on_market_price(self):
        first_observation = {
            "time_step": 1,
            "remaining_lots": 3,
            "market_price": 100.0,
        }

        second_observation = {
            "time_step": 1,
            "remaining_lots": 3,
            "market_price": 105.0,
        }

        self.assertEqual(
            get_state(first_observation),
            get_state(second_observation),
        )

    def test_valid_actions_include_all_available_lots(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        actions = get_valid_actions(observation, self.config)

        self.assertEqual(actions, [0, 1, 2, 3, 4])

    def test_actions_respect_remaining_inventory(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 2,
        }

        actions = get_valid_actions(observation, self.config)

        self.assertEqual(actions, [0, 1, 2])

    def test_final_step_requires_remaining_inventory(self):
        observation = {
            "time_step": 2,
            "remaining_lots": 3,
        }

        actions = get_valid_actions(observation, self.config)

        self.assertEqual(actions, [3])


class QLearningAgentTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.agent = QLearningAgent(
            self.config,
            learning_rate=0.5,
            discount_factor=1.0,
        )

    def test_q_values_initialize_to_zero(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        self.assertEqual(self.agent.q_table, {})

        q_value = self.agent.get_q_value(observation, 2)

        self.assertEqual(q_value, 0.0)
        self.assertEqual(len(self.agent.q_table[(0, 4)]), 5)

    def test_terminal_update_uses_immediate_reward(self):
        observation = {
            "time_step": 2,
            "remaining_lots": 3,
        }

        self.agent.update(
            observation,
            action=3,
            reward=-10.0,
            next_observation=None,
        )

        self.assertAlmostEqual(
            self.agent.get_q_value(observation, 3),
            -5.0,
        )

    def test_repeated_updates_adjust_q_value(self):
        observation = {
            "time_step": 2,
            "remaining_lots": 3,
        }

        self.agent.update(observation, 3, -10.0, None)
        self.agent.update(observation, 3, -10.0, None)

        self.assertAlmostEqual(
            self.agent.get_q_value(observation, 3),
            -7.5,
        )

    def test_update_includes_best_future_value(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 3,
        }

        next_observation = {
            "time_step": 2,
            "remaining_lots": 3,
        }

        # learns future Q-value of 4.0
        self.agent.update(next_observation, 3, 8.0, None)

        # target is -2 + 4 = 2
        self.agent.update(
            observation,
            action=0,
            reward=-2.0,
            next_observation=next_observation,
        )

        self.assertAlmostEqual(
            self.agent.get_q_value(observation, 0),
            1.0,
        )

    def test_invalid_action_is_rejected(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 2,
        }

        with self.assertRaises(ValueError):
            self.agent.update(observation, 3, -1.0, None)

        self.assertEqual(self.agent.q_table, {})

    def test_invalid_learning_parameters(self):
        invalid_parameters = [
            {"learning_rate": 0.0},
            {"learning_rate": 1.5},
            {"discount_factor": -0.1},
            {"discount_factor": 1.5},
        ]

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    QLearningAgent(self.config, **parameters)


if __name__ == "__main__":
    unittest.main()