import math

from dataclasses import dataclass

BASIS_POINTS_MULTIPLIER = 10_000


@dataclass(frozen=True)
class ExecutionCostConfig:
    half_spread_bps: float = 0.0
    temporary_impact_bps: float = 0.0
    fee_per_share: float = 0.0
    impact_model: str = "linear"

    def __post_init__(self):
        costs = [
            self.half_spread_bps,
            self.temporary_impact_bps,
            self.fee_per_share,
        ]

        if any(not math.isfinite(cost) or cost < 0 for cost in costs):
            raise ValueError("Execution costs must be non-negative and finite.")

        total_discount_bps = self.half_spread_bps + self.temporary_impact_bps

        if total_discount_bps >= BASIS_POINTS_MULTIPLIER:
            raise ValueError("Execution price discount must be below 100%.")

        if self.impact_model not in ("linear", "square_root"):
            raise ValueError("Unsupported market impact model.")


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
    relative_liquidity: float = 1.0,
) -> SellExecution:
    """calculates execution costs for a sell trade"""

    if not math.isfinite(market_price) or market_price <= 0:
        raise ValueError("Market price must be positive and finite.")

    if type(total_order_shares) is not int or total_order_shares <= 0:
        raise ValueError("Total order shares must be a positive integer.")

    if type(shares_sold) is not int:
        raise ValueError("Shares sold must be an integer.")

    if shares_sold < 0 or shares_sold > total_order_shares:
        raise ValueError("Shares sold must be within the order size.")

    if (
        isinstance(relative_liquidity, bool)
        or not isinstance(relative_liquidity, (int, float))
        or not math.isfinite(relative_liquidity)
        or relative_liquidity <= 0
    ):
        raise ValueError("Relative liquidity must be positive and finite.")

    if shares_sold == 0:
        return SellExecution(0, market_price, 0.0, 0.0)

    order_fraction = shares_sold / total_order_shares

    trade_intensity = order_fraction / relative_liquidity

    if cost_config.impact_model == "square_root":
        impact_factor = math.sqrt(trade_intensity)
    else:
        impact_factor = trade_intensity

    impact_bps = cost_config.temporary_impact_bps * impact_factor

    total_discount_bps = cost_config.half_spread_bps + impact_bps

    if total_discount_bps >= BASIS_POINTS_MULTIPLIER:
        raise ValueError("Execution price discount must be below 100%.")

    execution_price = market_price * (1 - total_discount_bps / BASIS_POINTS_MULTIPLIER)

    fees = shares_sold * cost_config.fee_per_share
    net_proceeds = shares_sold * execution_price - fees

    return SellExecution(
        shares_sold=shares_sold,
        execution_price=execution_price,
        fees=fees,
        net_proceeds=net_proceeds,
    )
