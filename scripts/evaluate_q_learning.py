import argparse
import math

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
TRAINING_SEED_STRIDE = 100_000

STARTING_PRICE = 100.0
VOLATILITY_PER_STEP = 0.002


def generate_episodes(
    num_episodes: int,
    config: ExecutionConfig,
    starting_seed: int,
    volatility_per_step: float = VOLATILITY_PER_STEP,
) -> list[list[float]]:
    """generates reproducible synthetic market episodes"""

    return [
        generate_price_path(
            starting_price=STARTING_PRICE,
            volatility_per_step=volatility_per_step,
            num_time_steps=config.num_time_steps,
            seed=starting_seed + episode_index,
        )
        for episode_index in range(num_episodes)
    ]


def main():
    
    parser = argparse.ArgumentParser(
        description="Evaluate the Q-learning agent"
    )

    parser.add_argument(
        "--full-size",
        action="store_true",
        help="Use the 10,000-share execution problem.",
    )
    
    parser.add_argument(
        "--training-episodes",
        type=int,
        default=TRAINING_EPISODES,
        help="Number of training episodes per agent.",
    )
    
    parser.add_argument(
        "--training-volatility",
        type=float,
        default=VOLATILITY_PER_STEP,
        help="Volatility used to generate training price paths.",
    )
    
    arguments = parser.parse_args()
    
    if not 0 < arguments.training_episodes <= TRAINING_SEED_STRIDE:
        parser.error(
            "Training episodes must be between 1 and 100,000."
        )

    if (
        not math.isfinite(arguments.training_volatility)
        or arguments.training_volatility < 0
    ):
        parser.error(
            "Training volatility must be non-negative and finite."
        )
        
    if arguments.full_size:
        config = ExecutionConfig(
            total_shares=10_000,
            num_time_steps=12,
            num_lots=20,
        )
        
    else:
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
    
    print(f"Order size: {config.total_shares:,} shares")
    print(f"Decision intervals: {config.num_time_steps}")
    print(f"Inventory lots: {config.num_lots}")
    
    print(f"Training episodes per agent: {arguments.training_episodes}")
    print(f"Training volatility: {arguments.training_volatility:.4f}")
    print(f"Validation volatility: {VOLATILITY_PER_STEP:.4f}")
    
    print(f"Test episodes: {TEST_EPISODES}")
    print(f"Training seeds: {len(AGENT_SEEDS)}")

    print("\nBaseline mean execution costs:")

    for policy_name, costs in baseline_results.items():
        print(f"{policy_name}: {mean(costs):.4f} bps")

    seed_differences = []

    for agent_seed in AGENT_SEEDS:
        
        training_paths = generate_episodes(
            arguments.training_episodes,
            config,
            TRAINING_SEED_START + agent_seed * TRAINING_SEED_STRIDE,
            volatility_per_step=arguments.training_volatility,
        )
        
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
        
        # measures how much inventory is left until the final interval
        final_step_shares = sum(
            record.shares_sold
            for record in reference_environment.execution_history
            if record.time_step == config.num_time_steps - 1
        )

        final_step_percentage = (
            final_step_shares / config.total_shares * 100
        )

        flat_market_cost = (
            reference_environment.implementation_shortfall_bps
        )

        # measures state-action learning coverage
        total_state_actions = sum(
            len(action_values)
            for action_values in agent.q_table.values()
        )

        updated_state_actions = sum(
            len(action_counts)
            for action_counts in agent.action_update_counts.values()
        )

        # inspects the agent's initial decision
        initial_state = (0, config.num_lots)

        initial_action_values = agent.q_table[initial_state]
        initial_update_counts = agent.action_update_counts.get(
            initial_state, {}
        )

        print(f"\nAgent seed: {agent_seed}")
        
        print(
            f"Updated state-action pairs: "
            f"{updated_state_actions}/{total_state_actions}"
        )

        print("Initial-state Q-values and update counts:")

        for action in (0, 1, 2, config.num_lots):
            q_value = initial_action_values[action]
            update_count = initial_update_counts.get(action, 0)

            print(
                f"  Sell {action:2} lots: "
                f"Q = {q_value:+.4f}, "
                f"updates = {update_count}"
            )

        print(f"Q-table states: {len(agent.q_table)}")
        print(f"Learned schedule: {learned_schedule}")
        print(f"Flat-market cost: {flat_market_cost:.4f} bps")

        print(
            f"Final-interval execution: {final_step_shares:,} shares "
            f"({final_step_percentage:.1f}%)"
        )

        print(f"Mean execution cost: {mean(q_learning_costs):.4f} bps")
        print(f"Difference from TWAP: {average_difference:+.4f} bps")

    print("\nResults across training seeds:")
    print(f"Mean difference from TWAP: {mean(seed_differences):+.4f} bps")
    print(f"Best seed difference: {min(seed_differences):+.4f} bps")
    print(f"Worst seed difference: {max(seed_differences):+.4f} bps")


if __name__ == "__main__":
    main()