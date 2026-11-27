import unittest

from execai.environment.config import ExecutionConfig
from execai.policies.baselines import immediate_action, twap_action


class BaselinePolicyTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

    def test_immediate_execution_sells_everything(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
            "market_price": 100.0,
        }

        action = immediate_action(observation)

        self.assertEqual(action, 4)

    def test_immediate_execution_uses_remaining_inventory(self):
        observation = {
            "time_step": 1,
            "remaining_lots": 2,
            "market_price": 99.0,
        }

        action = immediate_action(observation)

        self.assertEqual(action, 2)

    def test_twap_first_action(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
            "market_price": 100.0,
        }

        action = twap_action(observation, self.config)

        self.assertEqual(action, 2)

    def test_twap_distributes_remaining_inventory(self):
        observations = [
            {"time_step": 0, "remaining_lots": 4},
            {"time_step": 1, "remaining_lots": 2},
            {"time_step": 2, "remaining_lots": 1},
        ]

        actions = [
            twap_action(observation, self.config)
            for observation in observations
        ]

        self.assertEqual(actions, [2, 1, 1])

    def test_twap_with_evenly_divisible_inventory(self):
        config = ExecutionConfig(
            total_shares=600,
            num_time_steps=3,
            num_lots=6,
        )

        observation = {
            "time_step": 0,
            "remaining_lots": 6,
        }

        action = twap_action(observation, config)

        self.assertEqual(action, 2)

    def test_twap_sells_remaining_inventory_at_deadline(self):
        observation = {
            "time_step": 2,
            "remaining_lots": 3,
        }

        action = twap_action(observation, self.config)

        self.assertEqual(action, 3)


if __name__ == "__main__":
    unittest.main()