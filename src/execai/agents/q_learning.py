from execai.environment.config import ExecutionConfig


def get_state(observation: dict) -> tuple[int, int]:
    """represents the current time and remaining inventory"""

    return (
        observation["time_step"],
        observation["remaining_lots"],
    )


def get_valid_actions(
    observation: dict,
    config: ExecutionConfig,
) -> list[int]:
    """returns the trade quantities available to the agent"""

    remaining_lots = observation["remaining_lots"]
    current_step = observation["time_step"]

    is_final_step = current_step == config.num_time_steps - 1

    if is_final_step:
        return [remaining_lots]

    return list(range(remaining_lots + 1))


class QLearningAgent:
    """learns trade execution decisions using tabular Q-learning"""

    def __init__(
        self,
        config: ExecutionConfig,
        learning_rate: float = 0.1,
        discount_factor: float = 1.0,
    ):
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1.")

        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be between 0 and 1.")

        self.config = config
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.q_table: dict[tuple[int, int], dict[int, float]] = {}

    def _get_action_values(self, observation: dict) -> dict[int, float]:
        """gets Q-values + initializes unseen states to zero"""

        state = get_state(observation)

        if state not in self.q_table:
            valid_actions = get_valid_actions(observation, self.config)

            self.q_table[state] = {
                action: 0.0 for action in valid_actions
            }

        return self.q_table[state]

    def get_q_value(self, observation: dict, action: int) -> float:
        """returns the estimated value of a state-action pair"""

        valid_actions = get_valid_actions(observation, self.config)

        if action not in valid_actions:
            raise ValueError("Action is not valid for the current state.")

        action_values = self._get_action_values(observation)

        return action_values[action]

    def update(
        self,
        observation: dict,
        action: int,
        reward: float,
        next_observation: dict | None,
    ) -> None:
        """updates a Q-value using observed reward and next state"""

        valid_actions = get_valid_actions(observation, self.config)

        if action not in valid_actions:
            raise ValueError("Action is not valid for the current state.")

        action_values = self._get_action_values(observation)
        current_q_value = action_values[action]

        best_future_value = 0.0

        if next_observation is not None:
            next_action_values = self._get_action_values(next_observation)
            best_future_value = max(next_action_values.values())

        target = reward + self.discount_factor * best_future_value

        updated_q_value = current_q_value + self.learning_rate * (
            target - current_q_value
        )

        action_values[action] = updated_q_value