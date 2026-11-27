import unittest

from execai.environment.config import ExecutionConfig


class ExecutionConfigTests(unittest.TestCase):

    def test_default_configuration(self):
        config = ExecutionConfig()

        self.assertEqual(config.total_shares, 10_000)
        self.assertEqual(config.num_time_steps, 12)
        self.assertEqual(config.num_lots, 20)
        self.assertEqual(config.shares_per_lot, 500)

    def test_invalid_total_shares(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(total_shares=0)

    def test_invalid_time_steps(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(num_time_steps=0)

    def test_invalid_num_lots(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(num_lots=0)

    def test_shares_must_be_divisible_into_lots(self):
        with self.assertRaises(ValueError):
            ExecutionConfig(total_shares=10_001)


if __name__ == "__main__":
    unittest.main()