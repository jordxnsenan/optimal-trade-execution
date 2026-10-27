import math

from execai.environment.config import ExecutionConfig
from execai.environment.costs import (
    BASIS_POINTS_MULTIPLIER,
    ExecutionCostConfig,
    calculate_sell_execution,
)
from execai.environment.inventory import Inventory
from execai.environment.records import ExecutionRecord


class ExecutionEnvironment:
    """simulates the execution of a fixed sell order"""

    def __init__(
        self,
        config: ExecutionConfig,
        cost_config: ExecutionCostConfig | None = None,
    ):
        self.config = config
        self.cost_config = (
            cost_config if cost_config is not None else ExecutionCostConfig()
        )
        self.inventory = Inventory(config)

        self._prices = []
        self.current_step = 0
        self.total_proceeds = 0.0
        self.total_fees = 0.0
        self.is_done = True
        
        self._execution_history: list[ExecutionRecord] = []

    def reset(self, prices: list[float]) -> dict:
        """starts a new execution episode"""

        if len(prices) != self.config.num_time_steps:
            raise ValueError("Price path length must match the number of time steps.")

        if any(not math.isfinite(price) or price <= 0 for price in prices):
            raise ValueError("All market prices must be positive and finite.")

        self._prices = list(prices)
        self.inventory.reset()

        self.current_step = 0
        self.total_proceeds = 0.0
        self.total_fees = 0.0
        self.is_done = False
        self._execution_history.clear()

        return self._get_observation()

    def _get_observation(self) -> dict:
        """returns information available at the current decision step"""

        return {
            "time_step": self.current_step,
            "remaining_lots": self.inventory.remaining_lots,
            "market_price": self._prices[self.current_step],
        }

    def step(self, num_lots: int) -> tuple[dict | None, float, bool]:
        """executes a sell trade and advances the episode"""

        if self.is_done:
            raise RuntimeError("Reset the environment before taking another step.")

        shares_sold = self.inventory.execute_sell(num_lots)
        forced_shares = 0
        
        is_final_step = self.current_step == (self.config.num_time_steps - 1)
        
        if is_final_step and self.inventory.remaining_lots > 0:
            forced_shares = self.inventory.execute_sell(
                self.inventory.remaining_lots
            )
            
            shares_sold += forced_shares

        market_price = self._prices[self.current_step]

        execution = calculate_sell_execution(
            market_price,
            shares_sold,
            self.config.total_shares,
            self.cost_config,
        )

        self.total_proceeds += execution.net_proceeds
        self.total_fees += execution.fees

        arrival_price = self._prices[0]
        arrival_notional = self.config.total_shares * arrival_price

        shortfall_dollars = (
            shares_sold * arrival_price - execution.net_proceeds
        )

        reward = (
            -shortfall_dollars
            / arrival_notional
            * BASIS_POINTS_MULTIPLIER
        )
        
        
        execution_record = ExecutionRecord(
            time_step=self.current_step,
            requested_lots=num_lots,
            shares_sold=shares_sold,
            forced_shares=forced_shares,
            market_price=market_price,
            execution_price=execution.execution_price,
            fees=execution.fees,
            net_proceeds=execution.net_proceeds,
            remaining_shares=self.inventory.remaining_shares,
            reward_bps=reward,
        )

        self._execution_history.append(execution_record)

        self.current_step += 1

        self.is_done = (
            self.inventory.remaining_lots == 0
            or self.current_step == self.config.num_time_steps
        )

        next_observation = None if self.is_done else self._get_observation()

        return next_observation, reward, self.is_done

    @property
    def implementation_shortfall_bps(self) -> float:
        """calculates implementation shortfall for a completed episode"""

        if not self.is_done or not self._prices:
            raise RuntimeError("The episode must be completed first.")

        arrival_notional = self.config.total_shares * self._prices[0]
        shortfall_dollars = arrival_notional - self.total_proceeds

        return (
            shortfall_dollars
            / arrival_notional
            * BASIS_POINTS_MULTIPLIER
        )
        
    @property
    def execution_history(self) -> tuple[ExecutionRecord, ...]:
        """returns the execution records for the current episode"""
        
        return tuple(self._execution_history)   