from execai.environment.config import ExecutionConfig

FIRST_DECISION_BAR_INDEX = 6


def build_historical_price_path(
    bars: list[dict],
    config: ExecutionConfig,
    first_decision_bar_index: int = FIRST_DECISION_BAR_INDEX,
) -> list[float]:
    """builds a price path using only previously completed bars"""

    if (
        isinstance(first_decision_bar_index, bool)
        or not isinstance(first_decision_bar_index, int)
        or first_decision_bar_index < 1
    ):
        raise ValueError("First decision bar index must be a positive integer.")

    last_required_bar_index = first_decision_bar_index + config.num_time_steps - 2

    if last_required_bar_index >= len(bars):
        raise ValueError("Not enough historical bars for the execution window.")

    prices = []

    for time_step in range(config.num_time_steps):
        decision_bar_index = first_decision_bar_index + time_step

        # uses the close of the bar completed before this decision
        previous_bar = bars[decision_bar_index - 1]
        prices.append(previous_bar["c"])

    return prices
