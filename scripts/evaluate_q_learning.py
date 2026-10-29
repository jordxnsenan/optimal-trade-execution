from functools import partial
from statistics import mean

from execai.agents.q_learning import QLearningAgent
from execai.agents.training import train_q_learning
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.market import generate_price_path
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.comparison import evaluate_policies
from execai.policies.baselines import immediate_action, twap_action
from execai.evaluation.runner import run_policy


TRAINING_EPISODES = 5_000
TEST_EPISODES = 500

AGENT_SEED = 42
TRAINING_SEED_START = 0
TEST_SEED_START = 100_000

STARTING_PRICE = 100.0
VOLATILITY_PER_STEP = 0.002


def generate_episodes(
    num_episodes: int,
    config: ExecutionConfig,
    starting_seed: int,
) -> list[list[float]]:
    """generates reproducible synthetic market episodes."""

    return [
        generate_price_path(
            starting_price=STARTING_PRICE,
            volatility_per_step=VOLATILITY_PER_STEP,
            num_time_steps=config.num_time_steps,
            seed=starting_seed + episode_index,
        )
        for episode_index in range(num_episodes)
    ]


def main():
    config = ExecutionConfig(
        total_shares=400,
        num_time_steps=3,
        num_lots=4,
    )

    cost_config = ExecutionCostConfig(
        half_spread_bps=5.0,
        temporary_impact_bps=20.0,
        fee_per_share=0.01,
    )

    training_paths = generate_episodes(
        TRAINING_EPISODES,
        config,
        TRAINING_SEED_START,
    )

    test_paths = generate_episodes(
        TEST_EPISODES,
        config,
        TEST_SEED_START,
    )

    agent = QLearningAgent(
        config,
        learning_rate=0.1,
        discount_factor=1.0,
        seed=AGENT_SEED,
    )

    training_environment = ExecutionEnvironment(config, cost_config)

    train_q_learning(
        agent,
        training_environment,
        training_paths,
        initial_exploration_rate=1.0,
        final_exploration_rate=0.1,
    )

    policies = {
        "Immediate": immediate_action,
        "TWAP": partial(twap_action, config=config),
        "Q-Learning": agent.select_action,
    }

    results = evaluate_policies(
        config,
        cost_config,
        test_paths,
        policies,
    )

    print("Held-Out Q-Learning Evaluation")
    print("------------------------------")
    print(f"Training episodes: {TRAINING_EPISODES}")
    print(f"Test episodes: {TEST_EPISODES}")
    print(f"Q-table states: {len(agent.q_table)}")

    print("\nMean implementation shortfall:")

    for policy_name, execution_costs in results.items():
        average_cost = mean(execution_costs)
        print(f"{policy_name}: {average_cost:.2f} bps")

    q_learning_costs = results["Q-Learning"]
    twap_costs = results["TWAP"]

    paired_differences = [
        q_learning_cost - twap_cost
        for q_learning_cost, twap_cost in zip(
            q_learning_costs,
            twap_costs,
        )
    ]

    average_difference = mean(paired_differences)

    # inspects the learned policy using a constant reference price path
    reference_prices = [STARTING_PRICE] * config.num_time_steps

    reference_environment = ExecutionEnvironment(config, cost_config)

    run_policy(
        reference_environment,
        reference_prices,
        agent.select_action,
    )

    learned_schedule = [
        record.shares_sold
        for record in reference_environment.execution_history
    ]

    num_better_episodes = sum(
        difference < -1e-9
        for difference in paired_differences
    )

    print(f"\nLearned execution schedule: {learned_schedule}")
    print(f"Q-Learning minus TWAP: {average_difference:+.6f} bps")

    print(
        f"Episodes where Q-Learning beats TWAP: "
        f"{num_better_episodes}/{TEST_EPISODES}"
    )

    print("Negative cost differences favor Q-Learning.")


if __name__ == "__main__":
    main()