import math

from execai.environment.config import ExecutionConfig
from execai.environment.inventory import Inventory


BASIS_POINTS_MULTIPLIER = 10_000


class ExecutionEnvironment:
    """simulates the execution of a fixed sell order"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.inventory = Inventory(config)

        self._prices = []
        self.current_step = 0
        self.total_proceeds = 0.0
        self.is_done = True

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
        self.is_done = False

        return self._get_observation()

    def _get_observation(self) -> dict:
        """returns information available at the current decision step"""

        return {
            "time_step": self.current_step,
            "remaining_lots": self.inventory.remaining_lots,
            "market_price": self._prices[self.current_step],
        }

    def step(self, num_lots: int) -> tuple[dict | None, float, bool]:
        """executes a trade and advances the episode"""

        if self.is_done:
            raise RuntimeError("Reset the environment before taking another step.")

        shares_sold = self.inventory.execute_sell(num_lots)

        # ensures any remaining inventory is sold at the final decision
        if self.current_step == self.config.num_time_steps - 1:
            shares_sold += self.inventory.execute_sell(self.inventory.remaining_lots)

        market_price = self._prices[self.current_step]
        self.total_proceeds += shares_sold * market_price

        arrival_price = self._prices[0]
        arrival_notional = self.config.total_shares * arrival_price

        shortfall_dollars = shares_sold * (arrival_price - market_price)
        reward = -shortfall_dollars / arrival_notional * BASIS_POINTS_MULTIPLIER

        self.current_step += 1

        self.is_done = (
            self.inventory.remaining_lots == 0
            or self.current_step == self.config.num_time_steps
        )

        next_observation = None if self.is_done else self._get_observation()

        return next_observation, reward, self.is_done

    @property
    def implementation_shortfall_bps(self) -> float:
        """Calculate the completed order's implementation shortfall."""

        if not self.is_done or not self._prices:
            raise RuntimeError("The episode must be completed first.")

        arrival_notional = self.config.total_shares * self._prices[0]
        shortfall_dollars = arrival_notional - self.total_proceeds

        return shortfall_dollars / arrival_notional * BASIS_POINTS_MULTIPLIER