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

AGENT_SEEDS = (42, 43, 44, 45, 46)
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

    # evaluates the traditional baselines once on the shared test paths
    baseline_policies = {
        "Immediate": immediate_action,
        "TWAP": partial(twap_action, config=config),
    }

    baseline_results = evaluate_policies(
        config,
        cost_config,
        test_paths,
        baseline_policies,
    )

    twap_costs = baseline_results["TWAP"]

    print("Multi-Seed Q-Learning Evaluation")
    print("--------------------------------")
    print(f"Training episodes per agent: {TRAINING_EPISODES}")
    print(f"Test episodes: {TEST_EPISODES}")
    print(f"Training seeds: {len(AGENT_SEEDS)}")

    print("\nBaseline mean execution costs:")

    for policy_name, costs in baseline_results.items():
        print(f"{policy_name}: {mean(costs):.4f} bps")

    seed_differences = []

    for agent_seed in AGENT_SEEDS:
        agent = QLearningAgent(
            config,
            learning_rate=0.1,
            discount_factor=1.0,
            seed=agent_seed,
        )

        training_environment = ExecutionEnvironment(
            config,
            cost_config,
        )

        train_q_learning(
            agent,
            training_environment,
            training_paths,
            initial_exploration_rate=1.0,
            final_exploration_rate=0.1,
        )

        # evaluates the trained agent without exploration
        q_learning_results = evaluate_policies(
            config,
            cost_config,
            test_paths,
            {"Q-Learning": agent.select_action},
        )

        q_learning_costs = q_learning_results["Q-Learning"]

        paired_differences = [
            q_learning_cost - twap_cost
            for q_learning_cost, twap_cost in zip(
                q_learning_costs,
                twap_costs,
            )
        ]

        average_difference = mean(paired_differences)
        seed_differences.append(average_difference)

        # inspects the learned schedule on a reference price path
        reference_environment = ExecutionEnvironment(
            config,
            cost_config,
        )

        reference_prices = [STARTING_PRICE] * config.num_time_steps

        run_policy(
            reference_environment,
            reference_prices,
            agent.select_action,
        )

        learned_schedule = [
            record.shares_sold
            for record in reference_environment.execution_history
        ]

        print(f"\nAgent seed: {agent_seed}")
        print(f"Learned schedule: {learned_schedule}")
        print(f"Mean execution cost: {mean(q_learning_costs):.4f} bps")
        print(f"Difference from TWAP: {average_difference:+.4f} bps")

    print("\nResults across training seeds:")
    print(f"Mean difference from TWAP: {mean(seed_differences):+.4f} bps")
    print(f"Best seed difference: {min(seed_differences):+.4f} bps")
    print(f"Worst seed difference: {max(seed_differences):+.4f} bps")


if __name__ == "__main__":
    main()