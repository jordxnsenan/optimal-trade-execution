from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionConfig:
    total_shares: int = 10_000
    num_time_steps: int = 12
    num_lots: int = 20

    def __post_init__(self):
        if self.total_shares <= 0:
            raise ValueError("Total shares must be positive.")

        if self.num_time_steps <= 0:
            raise ValueError("Number of time steps must be positive.")

        if self.num_lots <= 0:
            raise ValueError("Number of lots must be positive.")

        if self.total_shares % self.num_lots != 0:
            raise ValueError("Total shares must be divisible by the number of lots.")

    @property
    def shares_per_lot(self):
        return self.total_shares // self.num_lots