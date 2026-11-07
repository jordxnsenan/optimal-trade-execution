import unittest

from execai.agents.replay_buffer import ReplayBuffer


class ReplayBufferTests(unittest.TestCase):

    def setUp(self):
        self.buffer = ReplayBuffer(capacity=3, seed=42)

    def add_transition(self, buffer: ReplayBuffer, sample_id: int) -> None:
        """adds a sample transition for testing"""

        buffer.add(
            observation={
                "time_step": 0,
                "remaining_lots": 4,
                "sample_id": sample_id,
            },
            action=1,
            reward=-1.0,
            next_observation={
                "time_step": 1,
                "remaining_lots": 3,
            },
            done=False,
        )

    def test_stores_transitions(self):
        self.add_transition(self.buffer, 1)

        self.assertEqual(len(self.buffer), 1)

        transition = self.buffer.sample(1)[0]

        self.assertEqual(transition.action, 1)
        self.assertEqual(transition.reward, -1.0)
        self.assertFalse(transition.done)
        self.assertEqual(transition.observation["remaining_lots"], 4)

    def test_removes_oldest_transition_when_full(self):
        for sample_id in range(5):
            self.add_transition(self.buffer, sample_id)

        self.assertEqual(len(self.buffer), 3)

        transitions = self.buffer.sample(3)

        stored_ids = {transition.observation["sample_id"] for transition in transitions}

        self.assertEqual(stored_ids, {2, 3, 4})

    def test_sampling_does_not_repeat_transitions(self):
        for sample_id in range(3):
            self.add_transition(self.buffer, sample_id)

        transitions = self.buffer.sample(3)

        sampled_ids = [
            transition.observation["sample_id"] for transition in transitions
        ]

        self.assertEqual(len(set(sampled_ids)), 3)
        self.assertEqual(len(self.buffer), 3)

    def test_sampling_is_reproducible(self):
        first_buffer = ReplayBuffer(capacity=10, seed=42)
        second_buffer = ReplayBuffer(capacity=10, seed=42)

        for sample_id in range(10):
            self.add_transition(first_buffer, sample_id)
            self.add_transition(second_buffer, sample_id)

        first_sample = first_buffer.sample(5)
        second_sample = second_buffer.sample(5)

        self.assertEqual(first_sample, second_sample)

    def test_stores_terminal_transition(self):
        self.buffer.add(
            observation={
                "time_step": 2,
                "remaining_lots": 2,
            },
            action=2,
            reward=-10.0,
            next_observation=None,
            done=True,
        )

        transition = self.buffer.sample(1)[0]

        self.assertTrue(transition.done)
        self.assertIsNone(transition.next_observation)

    def test_copies_observations_when_storing(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        next_observation = {
            "time_step": 1,
            "remaining_lots": 3,
        }

        self.buffer.add(
            observation,
            action=1,
            reward=-2.0,
            next_observation=next_observation,
            done=False,
        )

        observation["remaining_lots"] = 0
        next_observation["remaining_lots"] = 0

        stored_transition = self.buffer.sample(1)[0]

        self.assertEqual(
            stored_transition.observation["remaining_lots"],
            4,
        )

        self.assertEqual(
            stored_transition.next_observation["remaining_lots"],
            3,
        )

    def test_rejects_invalid_capacity(self):
        for capacity in (0, -1, True):
            with self.subTest(capacity=capacity):
                with self.assertRaises(ValueError):
                    ReplayBuffer(capacity)

    def test_rejects_invalid_batch_size(self):
        self.add_transition(self.buffer, 1)

        for batch_size in (0, -1, 2, True):
            with self.subTest(batch_size=batch_size):
                with self.assertRaises(ValueError):
                    self.buffer.sample(batch_size)

    def test_rejects_inconsistent_terminal_state(self):
        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        with self.assertRaises(ValueError):
            self.buffer.add(
                observation,
                action=1,
                reward=-1.0,
                next_observation=None,
                done=False,
            )

        with self.assertRaises(ValueError):
            self.buffer.add(
                observation,
                action=1,
                reward=-1.0,
                next_observation=observation,
                done=True,
            )

        self.assertEqual(len(self.buffer), 0)


if __name__ == "__main__":
    unittest.main()
