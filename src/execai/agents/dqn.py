import torch

from torch import nn
from execai.agents.q_learning import get_state
from execai.environment.config import ExecutionConfig

STATE_FEATURE_COUNT = 2
DEFAULT_HIDDEN_UNITS = 64


def encode_observation(
    observation: dict,
    config: ExecutionConfig,
) -> torch.Tensor:
    """converts an execution observation into normalized state features"""

    current_step, remaining_lots = get_state(observation)

    if type(current_step) is not int or not 0 <= current_step < config.num_time_steps:
        raise ValueError("Invalid execution time step.")

    if type(remaining_lots) is not int or not 0 <= remaining_lots <= config.num_lots:
        raise ValueError("Invalid remaining inventory.")

    max_time_step = max(config.num_time_steps - 1, 1)

    normalized_time = current_step / max_time_step
    normalized_inventory = remaining_lots / config.num_lots

    return torch.tensor(
        [normalized_time, normalized_inventory],
        dtype=torch.float32,
    )


class QNetwork(nn.Module):
    """estimates the Q-value of each discrete execution action"""

    def __init__(
        self,
        num_lots: int,
        hidden_units: int = DEFAULT_HIDDEN_UNITS,
    ):
        super().__init__()

        if type(num_lots) is not int or num_lots <= 0:
            raise ValueError("Number of lots must be a positive integer.")

        if type(hidden_units) is not int or hidden_units <= 0:
            raise ValueError("Hidden units must be a positive integer.")

        num_actions = num_lots + 1

        self.network = nn.Sequential(
            nn.Linear(STATE_FEATURE_COUNT, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, num_actions),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """predicts Q-values for the supplied states"""

        return self.network(states)
