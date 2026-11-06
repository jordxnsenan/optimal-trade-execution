import math

from execai.environment.config import ExecutionConfig
from execai.environment.historical import FIRST_DECISION_BAR_INDEX


def build_relative_liquidity_profile(
    volume_fractions: list[float],
    config: ExecutionConfig,
    first_decision_bar_index: int = FIRST_DECISION_BAR_INDEX,
) -> list[float]:
    """converts forecast volume into relative execution-window liquidity"""

    if type(first_decision_bar_index) is not int or first_decision_bar_index < 0:
        raise ValueError("First decision bar index must be non-negative.")

    if any(
        isinstance(volume, bool)
        or not isinstance(volume, (int, float))
        or not math.isfinite(volume)
        or volume < 0
        for volume in volume_fractions
    ):
        raise ValueError("Forecast volumes must be finite and non-negative.")

    last_index = first_decision_bar_index + config.num_time_steps

    if last_index > len(volume_fractions):
        raise ValueError("Volume profile does not cover the execution window.")

    window_volumes = volume_fractions[first_decision_bar_index:last_index]

    if any(volume <= 0 for volume in window_volumes):
        raise ValueError("Execution-window forecast volumes must be positive.")

    average_volume = sum(window_volumes) / len(window_volumes)

    relative_liquidity = [volume / average_volume for volume in window_volumes]

    return relative_liquidity
