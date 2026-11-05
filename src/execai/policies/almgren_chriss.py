import math

from execai.environment.config import ExecutionConfig
from execai.environment.costs import (
    BASIS_POINTS_MULTIPLIER,
    ExecutionCostConfig,
    calculate_sell_execution,
)


def calculate_expected_cost_bps(
    schedule: list[int],
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    reference_price: float,
) -> float:
    """calculates expected execution cost for a fixed trade schedule"""

    if not math.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("Reference price must be positive and finite.")

    if len(schedule) != config.num_time_steps:
        raise ValueError("Schedule length must match the number of time steps.")

    if any(type(lots) is not int or lots < 0 for lots in schedule):
        raise ValueError("Schedule must contain non-negative integer lots.")

    if sum(schedule) != config.num_lots:
        raise ValueError("Schedule must execute the entire order.")

    total_net_proceeds = 0.0

    for lots_to_sell in schedule:
        shares_sold = lots_to_sell * config.shares_per_lot

        execution = calculate_sell_execution(
            reference_price,
            shares_sold,
            config.total_shares,
            cost_config,
        )

        total_net_proceeds += execution.net_proceeds

    arrival_notional = config.total_shares * reference_price
    shortfall_dollars = arrival_notional - total_net_proceeds

    return shortfall_dollars / arrival_notional * BASIS_POINTS_MULTIPLIER
