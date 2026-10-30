import random

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
        seed: int | None = None,
    ):
        if not 0 < learning_rate <= 1:
            raise ValueError("Learning rate must be between 0 and 1.")

        if not 0 <= discount_factor <= 1:
            raise ValueError("Discount factor must be between 0 and 1.")

        self.config = config
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.random_generator = random.Random(seed)

        self.q_table: dict[tuple[int, int], dict[int, float]] = {}
        self.action_update_counts: dict[tuple[int, int], dict[int, int]] = {}


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
    
    
    def select_action(
        self,
        observation: dict,
        exploration_rate: float = 0.0,
        random_tie_breaking: bool = False,
    ) -> int:
        """selects a valid action using epsilon-greedy exploration"""

        if not 0 <= exploration_rate <= 1:
            raise ValueError("Exploration rate must be between 0 and 1.")

        action_values = self._get_action_values(observation)
        valid_actions = list(action_values)

        # explores by selecting a random valid action
        if (
            exploration_rate > 0
            and self.random_generator.random() < exploration_rate
        ):
            return self.random_generator.choice(valid_actions)

        # finds all actions with the highest Q-value
        highest_q_value = max(action_values.values())

        best_actions = [
            action
            for action in valid_actions
            if action_values[action] == highest_q_value
        ]

        # optionally chooses randomly when multiple actions are equally valued
        if random_tie_breaking and len(best_actions) > 1:
            return self.random_generator.choice(best_actions)

        return best_actions[0]


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
        
        state = get_state(observation)

        if state not in self.action_update_counts:
            self.action_update_counts[state] = {}

        previous_count = self.action_update_counts[state].get(action, 0)
        self.action_update_counts[state][action] = previous_count + 1