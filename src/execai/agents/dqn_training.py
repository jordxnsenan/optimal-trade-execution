import copy
import math
import torch

from torch import nn

from execai.agents.dqn import DQNAgent, QNetwork, encode_observation
from execai.agents.q_learning import get_valid_actions
from execai.agents.replay_buffer import ReplayBuffer, Transition
from execai.environment.config import ExecutionConfig

DEFAULT_LEARNING_RATE = 0.001
DEFAULT_BATCH_SIZE = 32
DEFAULT_TARGET_SYNC_INTERVAL = 100


def calculate_dqn_targets(
    transitions: list[Transition],
    target_network: QNetwork,
    config: ExecutionConfig,
    discount_factor: float,
) -> torch.Tensor:
    """calculates Bellman targets using only valid future actions"""

    targets = []

    with torch.no_grad():
        for transition in transitions:
            target = transition.reward

            if not transition.done:
                next_observation = transition.next_observation

                if next_observation is None:
                    raise ValueError(
                        "Nonterminal transitions require a next observation."
                    )

                next_state = encode_observation(
                    next_observation,
                    config,
                    use_liquidity=target_network.use_liquidity,
                )

                next_q_values = target_network(next_state)
                valid_actions = get_valid_actions(next_observation, config)

                best_future_value = max(
                    next_q_values[action].item() for action in valid_actions
                )

                target += discount_factor * best_future_value

            targets.append(target)

    return torch.tensor(targets, dtype=torch.float32)


class DQNTrainer:
    """trains a DQN agent using replayed execution experiences"""

    def __init__(
        self,
        agent: DQNAgent,
        replay_buffer: ReplayBuffer,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        discount_factor: float = 1.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        target_sync_interval: int = DEFAULT_TARGET_SYNC_INTERVAL,
    ):
        if (
            isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("Learning rate must be positive and finite.")

        if (
            isinstance(discount_factor, bool)
            or not math.isfinite(discount_factor)
            or not 0 <= discount_factor <= 1
        ):
            raise ValueError("Discount factor must be between 0 and 1.")

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        if type(target_sync_interval) is not int or target_sync_interval <= 0:
            raise ValueError("Target sync interval must be a positive integer.")

        self.agent = agent
        self.replay_buffer = replay_buffer
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval

        self.target_network = copy.deepcopy(agent.network)
        self.target_network.eval()
        self.target_network.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            agent.network.parameters(),
            lr=learning_rate,
        )

        self.update_count = 0

    def train_step(self) -> float | None:
        """updates the online network using one sampled replay batch"""

        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)

        states = torch.stack(
            [
                encode_observation(
                    transition.observation,
                    self.agent.config,
                    use_liquidity=self.agent.use_liquidity,
                )
                for transition in transitions
            ]
        )

        actions = torch.tensor(
            [transition.action for transition in transitions],
            dtype=torch.long,
        )

        # predicts values for the actions actually taken
        all_q_values = self.agent.network(states)

        # retrieves prediction associated with selected action
        chosen_q_values = all_q_values.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        targets = calculate_dqn_targets(
            transitions,
            self.target_network,
            self.agent.config,
            self.discount_factor,
        )

        loss = nn.functional.smooth_l1_loss(
            chosen_q_values,
            targets,
        )

        # calc. gradients and updates the online networks weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1

        if self.update_count % self.target_sync_interval == 0:
            self.target_network.load_state_dict(self.agent.network.state_dict())

        return loss.item()
