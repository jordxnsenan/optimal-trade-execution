
import unittest

from execai.environment.config import ExecutionConfig
from execai.environment.inventory import Inventory


class InventoryTests(unittest.TestCase):

    def setUp(self):
        self.inventory = Inventory(ExecutionConfig())

    def test_initial_inventory(self):
        self.assertEqual(self.inventory.remaining_lots, 20)
        self.assertEqual(self.inventory.remaining_shares, 10_000)

    def test_selling_lots_updates_inventory(self):
        shares_sold = self.inventory.execute_sell(4)

        self.assertEqual(shares_sold, 2_000)
        self.assertEqual(self.inventory.remaining_lots, 16)
        self.assertEqual(self.inventory.remaining_shares, 8_000)

    def test_selling_zero_lots(self):
        shares_sold = self.inventory.execute_sell(0)

        self.assertEqual(shares_sold, 0)
        self.assertEqual(self.inventory.remaining_shares, 10_000)

    def test_cannot_sell_more_than_remaining_inventory(self):
        with self.assertRaises(ValueError):
            self.inventory.execute_sell(21)

        self.assertEqual(self.inventory.remaining_lots, 20)

    def test_cannot_sell_negative_lots(self):
        with self.assertRaises(ValueError):
            self.inventory.execute_sell(-1)

        self.assertEqual(self.inventory.remaining_lots, 20)

    def test_action_must_be_an_integer(self):
        with self.assertRaises(ValueError):
            self.inventory.execute_sell(2.5)

        with self.assertRaises(ValueError):
            self.inventory.execute_sell(True)

    def test_reset_restores_inventory(self):
        self.inventory.execute_sell(8)
        self.inventory.reset()

        self.assertEqual(self.inventory.remaining_lots, 20)
        self.assertEqual(self.inventory.remaining_shares, 10_000)


if __name__ == "__main__":
    unittest.main()