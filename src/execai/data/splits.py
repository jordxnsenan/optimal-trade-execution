from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateRange:
    """represents an inclusive range of calendar dates"""

    start_date: date
    end_date: date

    def __post_init__(self):
        if self.start_date > self.end_date:
            raise ValueError("Start date must not be after end date.")


@dataclass(frozen=True)
class DatasetSplits:
    """defines the chronological partitions of historical market data"""

    training: DateRange
    validation: DateRange
    test: DateRange

    def __post_init__(self):
        if self.training.end_date >= self.validation.start_date:
            raise ValueError("Training must end before validation begins.")

        if self.validation.end_date >= self.test.start_date:
            raise ValueError("Validation must end before testing begins.")

    def get_range(self, split_name: str) -> DateRange:
        """returns the date range for the requested dataset"""

        if split_name == "training":
            return self.training

        if split_name == "validation":
            return self.validation

        if split_name == "test":
            return self.test

        raise ValueError(f"Unknown dataset split: {split_name}")


DEFAULT_SPLITS = DatasetSplits(
    training=DateRange(
        start_date=date(2023, 1, 1),
        end_date=date(2024, 12, 31),
    ),
    validation=DateRange(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 12, 31),
    ),
    test=DateRange(
        start_date=date(2026, 1, 1),
        end_date=date(2026, 9, 30),
    ),
)
