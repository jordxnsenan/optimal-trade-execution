import math
import random

from statistics import mean

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SAMPLES = 2000
DEFAULT_BOOTSTRAP_SEED = 42


def paired_bootstrap_confidence_interval(
    first_costs: list[float],
    second_costs: list[float],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    num_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """estimates a confidence interval for the mean paired cost difference"""

    if len(first_costs) != len(second_costs) or len(first_costs) < 2:
        raise ValueError("At least two matching cost observations are required.")

    if (
        isinstance(confidence_level, bool)
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("Confidence level must be between 0 and 1.")

    if type(num_samples) is not int or num_samples < 2:
        raise ValueError("Number of bootstrap samples must be at least 2.")

    differences = [
        first_cost - second_cost
        for first_cost, second_cost in zip(first_costs, second_costs)
    ]

    if any(not math.isfinite(difference) for difference in differences):
        raise ValueError("Cost differences must be finite.")

    random_generator = random.Random(seed)
    bootstrap_means = []

    for _ in range(num_samples):
        sampled_differences = random_generator.choices(
            differences,
            k=len(differences),
        )

        bootstrap_means.append(mean(sampled_differences))

    bootstrap_means.sort()

    lower_tail = (1.0 - confidence_level) / 2.0

    lower_index = int(lower_tail * num_samples)
    upper_index = min(
        int((1.0 - lower_tail) * num_samples),
        num_samples - 1,
    )

    return bootstrap_means[lower_index], bootstrap_means[upper_index]


def two_way_bootstrap_confidence_interval(
    differences_by_seed: list[list[float]],
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    num_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """estimates uncertainty across training seeds and evaluation days"""

    num_seeds = len(differences_by_seed)

    if num_seeds < 2:
        raise ValueError("At least two training seeds are required.")

    num_days = len(differences_by_seed[0])

    if num_days < 2:
        raise ValueError("At least two evaluation days are required.")

    if any(len(differences) != num_days for differences in differences_by_seed):
        raise ValueError("Training seeds must have matching evaluation days.")

    if any(
        not math.isfinite(value)
        for differences in differences_by_seed
        for value in differences
    ):
        raise ValueError("Paired differences must be finite.")

    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(confidence_level)
        or not 0 < confidence_level < 1
    ):
        raise ValueError("Confidence level must be between 0 and 1.")

    if type(num_samples) is not int or num_samples < 2:
        raise ValueError("Number of bootstrap samples must be at least 2.")

    if type(seed) is not int:
        raise ValueError("Bootstrap seed must be an integer.")

    random_generator = random.Random(seed)
    bootstrap_means = []

    for _ in range(num_samples):
        sampled_seeds = random_generator.choices(
            range(num_seeds),
            k=num_seeds,
        )

        sampled_days = random_generator.choices(
            range(num_days),
            k=num_days,
        )

        total_difference = sum(
            differences_by_seed[seed_index][day_index]
            for seed_index in sampled_seeds
            for day_index in sampled_days
        )

        bootstrap_means.append(total_difference / (num_seeds * num_days))

    bootstrap_means.sort()

    lower_tail = (1.0 - confidence_level) / 2.0

    lower_index = int(lower_tail * num_samples)
    upper_index = min(
        int((1.0 - lower_tail) * num_samples),
        num_samples - 1,
    )

    return bootstrap_means[lower_index], bootstrap_means[upper_index]
