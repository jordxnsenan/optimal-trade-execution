import unittest

from datetime import date

from execai.data.splits import (
    DEFAULT_SPLITS,
    DatasetSplits,
    DateRange,
)


class DatasetSplitTests(unittest.TestCase):

    def setUp(self):
        self.training = DateRange(
            date(2023, 1, 1),
            date(2024, 12, 31),
        )

        self.validation = DateRange(
            date(2025, 1, 1),
            date(2025, 12, 31),
        )

        self.test = DateRange(
            date(2026, 1, 1),
            date(2026, 9, 30),
        )

    def test_default_split_dates(self):
        self.assertEqual(
            DEFAULT_SPLITS.training,
            self.training,
        )

        self.assertEqual(
            DEFAULT_SPLITS.validation,
            self.validation,
        )

        self.assertEqual(
            DEFAULT_SPLITS.test,
            self.test,
        )

    def test_get_range_by_name(self):
        self.assertEqual(
            DEFAULT_SPLITS.get_range("training"),
            self.training,
        )

        self.assertEqual(
            DEFAULT_SPLITS.get_range("validation"),
            self.validation,
        )

        self.assertEqual(
            DEFAULT_SPLITS.get_range("test"),
            self.test,
        )

    def test_rejects_unknown_split(self):
        with self.assertRaisesRegex(ValueError, "Unknown dataset split"):
            DEFAULT_SPLITS.get_range("unknown")

    def test_rejects_invalid_date_range(self):
        with self.assertRaises(ValueError):
            DateRange(
                date(2025, 12, 31),
                date(2025, 1, 1),
            )

    def test_rejects_overlapping_training_and_validation(self):
        overlapping_training = DateRange(
            date(2023, 1, 1),
            date(2025, 1, 1),
        )

        with self.assertRaisesRegex(ValueError, "Training must end"):
            DatasetSplits(
                training=overlapping_training,
                validation=self.validation,
                test=self.test,
            )

    def test_rejects_overlapping_validation_and_test(self):
        overlapping_validation = DateRange(
            date(2025, 1, 1),
            date(2026, 1, 1),
        )

        with self.assertRaisesRegex(ValueError, "Validation must end"):
            DatasetSplits(
                training=self.training,
                validation=overlapping_validation,
                test=self.test,
            )


if __name__ == "__main__":
    unittest.main()
