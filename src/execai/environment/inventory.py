from execai.environment.config import ExecutionConfig


class Inventory:
    """tracks remaining shares in an execution order"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.remaining_lots = config.num_lots

    @property
    def remaining_shares(self) -> int:
        return self.remaining_lots * self.config.shares_per_lot

    def execute_sell(self, num_lots: int) -> int:
        """sells inventory lots + returns number of shares sold"""

        if not isinstance(num_lots, int) or isinstance(num_lots, bool):
            raise ValueError("Number of lots must be an integer.")

        if num_lots < 0:
            raise ValueError("Number of lots cannot be negative.")

        if num_lots > self.remaining_lots:
            raise ValueError("Cannot sell more lots than remain.")

        shares_sold = num_lots * self.config.shares_per_lot
        self.remaining_lots -= num_lots

        return shares_sold

    def reset(self) -> None:
        """restores original inventory"""
        self.remaining_lots = self.config.num_lots