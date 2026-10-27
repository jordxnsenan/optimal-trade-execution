from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionRecord:
    """stores the results of one execution decision"""

    time_step: int
    requested_lots: int
    shares_sold: int
    forced_shares: int
    market_price: float
    execution_price: float
    fees: float
    net_proceeds: float
    remaining_shares: int
    reward_bps: float