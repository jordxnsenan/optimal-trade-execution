import random

from collections import deque
from dataclasses import dataclass


@dataclass
class Transition:
    """stores one experience collected from the execution environment"""

    observation: dict
    action: int
    reward: float
    next_observation: dict | None
    done: bool


class ReplayBuffer:
    """stores past experiences for DQN training"""

    def __init__(
        self,
        capacity: int,
        seed: int | None = None,
    ):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive.")

        self.capacity = capacity
        self.random_generator = random.Random(seed)

        self.transitions: deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        """returns the number of stored transitions"""

        return len(self.transitions)

    def add(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict | None,
        done: bool,
    ) -> None:
        """stores a transition collected from the environment"""

        if type(done) is not bool:
            raise ValueError("Done must be a boolean.")

        if done != (next_observation is None):
            raise ValueError("Terminal transitions must have no next observation.")

        transition = Transition(
            observation=observation.copy(),
            action=action,
            reward=reward,
            next_observation=(
                next_observation.copy() if next_observation is not None else None
            ),
            done=done,
        )

        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        """returns a random batch of stored experiences"""

        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("Batch size must be a positive integer.")

        if batch_size > len(self.transitions):
            raise ValueError(
                "Not enough transitions available for the requested batch."
            )

        return self.random_generator.sample(
            list(self.transitions),
            batch_size,
        )
