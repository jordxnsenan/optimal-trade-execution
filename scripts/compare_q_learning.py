from functools import partial

from execai.agents.q_learning import QLearningAgent
from execai.agents.training import train_q_learning
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy
from execai.policies.baselines import immediate_action, twap_action


TRAINING_EPISODES = 5_000
RANDOM_SEED = 42
KNOWN_OPTIMAL_COST_BPS = 13.5


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

    prices = [100.0] * config.num_time_steps

    # trains agent using the known-answer market
    agent = QLearningAgent(
        config,
        learning_rate=0.1,
        discount_factor=1.0,
        seed=RANDOM_SEED,
    )

    training_environment = ExecutionEnvironment(config, cost_config)

    training_price_paths = [
        prices.copy()
        for _ in range(TRAINING_EPISODES)
    ]

    training_costs = train_q_learning(
        agent,
        training_environment,
        training_price_paths,
        initial_exploration_rate=1.0,
        final_exploration_rate=0.1,
    )

    print("Q-Learning Comparison")
    print("---------------------")
    print(f"Training episodes: {len(training_costs)}")
    print(f"Q-table states learned: {len(agent.q_table)}")

    policies = [
        ("Immediate Execution", immediate_action),
        ("TWAP", partial(twap_action, config=config)),
        ("Q-Learning", agent.select_action),
    ]

    for policy_name, choose_action in policies:
        environment = ExecutionEnvironment(config, cost_config)

        execution_cost = run_policy(
            environment,
            prices,
            choose_action,
        )

        execution_schedule = [
            record.shares_sold
            for record in environment.execution_history
        ]

        print(f"\n{policy_name}")
        print(f"Execution schedule: {execution_schedule}")
        print(f"Implementation shortfall: {execution_cost:.2f} bps")

    print(f"\nKnown optimal cost: {KNOWN_OPTIMAL_COST_BPS:.2f} bps")


if __name__ == "__main__":
    main()