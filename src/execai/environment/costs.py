import math

from dataclasses import dataclass


BASIS_POINTS_MULTIPLIER = 10_000


@dataclass(frozen=True)
class ExecutionCostConfig:
    half_spread_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    fee_per_share: float = 0.0

    def __post_init__(self):
        costs = [
            self.half_spread_bps,
            self.temporary_impact_bps,
            self.fee_per_share,
        ]

        if any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("Execution costs must be non-negative and finite.")

        total_discount_bps = (
            self.half_spread_bps + self.temporary_impact_bps
        )

        if total_discount_bps >= BASIS_POINTS_MULTIPLIER:
            raise ValueError("Execution price discount must be below 100%.")


@dataclass(frozen=True)
class SellExecution:
    shares_sold: int
    execution_price: float
    fees: float
    net_proceeds: float


def calculate_sell_execution(
    market_price: float,
    shares_sold: int,
    total_order_shares: int,
    cost_config: ExecutionCostConfig,
) -> SellExecution:
    """calculates the execution price and net proceeds of a sell trade"""

    if not math.isfinite(market_price) or market_price <= 0:
        raise ValueError("Market price must be positive and finite.")

    if not isinstance(total_order_shares, int) or isinstance(total_order_shares, bool):
        raise ValueError("Total order shares must be an integer.")

    if total_order_shares <= 0:
        raise ValueError("Total order shares must be positive.")

    if not isinstance(shares_sold, int) or isinstance(shares_sold, bool):
        raise ValueError("Shares sold must be an integer.")

    if shares_sold < 0 or shares_sold > total_order_shares:
        raise ValueError("Shares sold must be within the order size.")

    if shares_sold == 0:
        return SellExecution(0, market_price, 0.0, 0.0)

    order_fraction = shares_sold / total_order_shares

    impact_bps = cost_config.temporary_impact_bps * order_fraction
    total_discount_bps = cost_config.half_spread_bps + impact_bps

    execution_price = market_price * (
        1 - total_discount_bps / BASIS_POINTS_MULTIPLIER
    )

    fees = shares_sold * cost_config.fee_per_share
    net_proceeds = shares_sold * execution_price - fees

    return SellExecution(
        shares_sold=shares_sold,
        execution_price=execution_price,
        fees=fees,
        net_proceeds=net_proceeds,
    )