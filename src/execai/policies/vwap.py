import math

from execai.environment.config import ExecutionConfig
from execai.environment.historical import FIRST_DECISION_BAR_INDEX
from execai.policies.schedules import scheduled_action

REGULAR_SESSION_BARS = 78


def build_vwap_schedule(
    config: ExecutionConfig,
    volume_fractions: list[float],
) -> list[int]:
    """allocates inventory lots using forecast trading volume"""

    if len(volume_fractions) != REGULAR_SESSION_BARS:
        raise ValueError("Volume profile must contain 78 intervals.")

    if any(
        not isinstance(volume, (int, float))
        or isinstance(volume, bool)
        or not math.isfinite(volume)
        or volume < 0
        for volume in volume_fractions
    ):
        raise ValueError("Volume fractions must be finite and non-negative.")

    first_index = FIRST_DECISION_BAR_INDEX
    last_index = first_index + config.num_time_steps

    window_volumes = volume_fractions[first_index:last_index]

    if len(window_volumes) != config.num_time_steps:
        raise ValueError("Volume profile does not cover the execution window.")

    total_window_volume = sum(window_volumes)

    if total_window_volume <= 0:
        raise ValueError("Execution window must have positive forecast volume.")

    schedule = []
    cumulative_volume = 0.0
    previously_allocated_lots = 0

    for index, volume in enumerate(window_volumes):
        cumulative_volume += volume

        if index == config.num_time_steps - 1:
            target_lots = config.num_lots
        else:
            target_lots = int(
                config.num_lots * cumulative_volume / total_window_volume + 0.5
            )

        lots_to_sell = target_lots - previously_allocated_lots

        schedule.append(lots_to_sell)
        previously_allocated_lots = target_lots

    return schedule


def vwap_action(observation: dict, schedule: list[int]) -> int:
    """returns the planned VWAP trade for the current time step"""

    return scheduled_action(observation, schedule)
