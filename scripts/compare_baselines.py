from functools import partial

from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy
from execai.policies.baselines import immediate_action, twap_action


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

    policies = [
        ("Immediate Execution", immediate_action),
        ("TWAP", partial(twap_action, config=config)),
    ]

    print("Baseline Strategy Comparison")
    print("----------------------------")

    for policy_name, choose_action in policies:
        environment = ExecutionEnvironment(config, cost_config)

        execution_cost = run_policy(
            environment,
            prices,
            choose_action,
        )

        shares_per_step = [
            record.shares_sold
            for record in environment.execution_history
        ]

        print(f"\n{policy_name}")
        print(f"Execution schedule: {shares_per_step}")
        print(f"Implementation shortfall: {execution_cost:.2f} bps")


if __name__ == "__main__":
    main()