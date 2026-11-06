import math

from execai.environment.config import ExecutionConfig
from execai.environment.costs import (
    BASIS_POINTS_MULTIPLIER,
    ExecutionCostConfig,
    calculate_sell_execution,
)

COST_COMPARISON_TOLERANCE = 1e-9


def _validate_schedule(
    schedule: list[int],
    config: ExecutionConfig,
) -> None:
    """checks that a schedule executes the complete order"""

    if len(schedule) != config.num_time_steps:
        raise ValueError("Schedule length must match the number of time steps.")

    if any(type(lots) is not int or lots < 0 for lots in schedule):
        raise ValueError("Schedule must contain non-negative integer lots.")

    if sum(schedule) != config.num_lots:
        raise ValueError("Schedule must execute the entire order.")


def calculate_expected_cost_bps(
    schedule: list[int],
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    reference_price: float,
) -> float:
    """calculates expected execution cost for a fixed trade schedule"""

    if not math.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("Reference price must be positive and finite.")

    _validate_schedule(schedule, config)

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


def optimize_almgren_chriss_schedule(
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    reference_price: float,
    volatility_per_step: float,
    risk_aversion: float = 0.0,
) -> list[int]:
    """finds the lowest-cost execution schedule with optional inventory risk"""

    if not math.isfinite(reference_price) or reference_price <= 0:
        raise ValueError("Reference price must be positive and finite.")

    if not math.isfinite(volatility_per_step) or volatility_per_step < 0:
        raise ValueError("Volatility must be non-negative and finite.")

    if not math.isfinite(risk_aversion) or risk_aversion < 0:
        raise ValueError("Risk aversion must be non-negative and finite.")

    num_steps = config.num_time_steps
    num_lots = config.num_lots

    arrival_notional = config.total_shares * reference_price

    volatility_variance_bps = (volatility_per_step * BASIS_POINTS_MULTIPLIER) ** 2

    # finds min. future cost for each time step and remaining inventory
    minimum_costs = [[math.inf] * (num_lots + 1) for _ in range(num_steps + 1)]

    best_actions = [[0] * (num_lots + 1) for _ in range(num_steps)]

    # only zero remaining inventory is valid at the end
    minimum_costs[num_steps][0] = 0.0

    # works backward from the final decision
    for time_step in range(num_steps - 1, -1, -1):
        for remaining_lots in range(num_lots + 1):

            is_final_step = time_step == num_steps - 1

            if is_final_step:
                valid_actions = [remaining_lots]
            else:
                valid_actions = range(remaining_lots + 1)

            for lots_to_sell in valid_actions:
                shares_sold = lots_to_sell * config.shares_per_lot
                next_remaining_lots = remaining_lots - lots_to_sell

                execution = calculate_sell_execution(
                    reference_price,
                    shares_sold,
                    config.total_shares,
                    cost_config,
                )

                execution_cost_bps = (
                    (shares_sold * reference_price - execution.net_proceeds)
                    / arrival_notional
                    * BASIS_POINTS_MULTIPLIER
                )

                remaining_fraction = next_remaining_lots / num_lots

                risk_cost_bps = (
                    risk_aversion * volatility_variance_bps * remaining_fraction**2
                )

                future_cost_bps = minimum_costs[time_step + 1][next_remaining_lots]

                total_cost_bps = execution_cost_bps + risk_cost_bps + future_cost_bps

                current_best_cost = minimum_costs[time_step][remaining_lots]

                is_lower_cost = (
                    total_cost_bps < current_best_cost - COST_COMPARISON_TOLERANCE
                )

                is_equal_cost = math.isclose(
                    total_cost_bps,
                    current_best_cost,
                    rel_tol=0.0,
                    abs_tol=COST_COMPARISON_TOLERANCE,
                )

                is_earlier_execution = (
                    lots_to_sell > best_actions[time_step][remaining_lots]
                )

                if is_lower_cost or (is_equal_cost and is_earlier_execution):
                    minimum_costs[time_step][remaining_lots] = total_cost_bps
                    best_actions[time_step][remaining_lots] = lots_to_sell

    # reconstructs the optimal schedule from the initial inventory
    schedule = []
    remaining_lots = num_lots

    for time_step in range(num_steps):
        lots_to_sell = best_actions[time_step][remaining_lots]

        schedule.append(lots_to_sell)
        remaining_lots -= lots_to_sell

    return schedule


def calculate_inventory_risk_bps_squared(
    schedule: list[int],
    config: ExecutionConfig,
    volatility_per_step: float,
) -> float:
    """estimates price-risk variance from inventory held between decisions"""

    _validate_schedule(schedule, config)

    if not math.isfinite(volatility_per_step) or volatility_per_step < 0:
        raise ValueError("Volatility must be non-negative and finite.")

    volatility_bps_squared = (volatility_per_step * BASIS_POINTS_MULTIPLIER) ** 2

    remaining_lots = config.num_lots
    total_risk = 0.0

    for lots_to_sell in schedule:
        remaining_lots -= lots_to_sell

        remaining_fraction = remaining_lots / config.num_lots

        total_risk += volatility_bps_squared * remaining_fraction**2

    return total_risk
