import unittest

from scripts.diagnose_dqn_liquidity import track_policy_actions


class DQNLiquidityDiagnosticTests(unittest.TestCase):

    def test_records_actions_without_changing_policy(self):
        def policy(observation):
            return observation["remaining_lots"] // 2

        tracked_policy, action_totals = track_policy_actions(
            policy,
            num_time_steps=3,
        )

        first_action = tracked_policy(
            {
                "time_step": 0,
                "remaining_lots": 4,
            }
        )

        second_action = tracked_policy(
            {
                "time_step": 1,
                "remaining_lots": 2,
            }
        )

        self.assertEqual(first_action, 2)
        self.assertEqual(second_action, 1)
        self.assertEqual(action_totals, [2, 1, 0])

    def test_records_multiple_actions_at_the_same_step(self):
        def policy(observation):
            return 2

        tracked_policy, action_totals = track_policy_actions(
            policy,
            num_time_steps=3,
        )

        tracked_policy({"time_step": 1})
        tracked_policy({"time_step": 1})

        self.assertEqual(action_totals, [0, 4, 0])


if __name__ == "__main__":
    unittest.main()
