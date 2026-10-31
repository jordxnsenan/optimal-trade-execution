from collections.abc import Callable
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.evaluation.runner import run_policy


def evaluate_policies(
    config: ExecutionConfig,
    cost_config: ExecutionCostConfig,
    price_paths: list[list[float]],
    policies: dict[str, Callable[[dict], int]],
) -> dict[str, list[float]]:
    """evaluates each policy on the same market scenarios"""

    if not price_paths:
        raise ValueError("At least one price path is required.")

    if not policies:
        raise ValueError("At least one policy is required.")

    execution_costs = {
        policy_name: []
        for policy_name in policies
    }

    for prices in price_paths:
        for policy_name, choose_action in policies.items():
            environment = ExecutionEnvironment(config, cost_config)

            cost = run_policy(
                environment,
                prices,
                choose_action,
            )

            execution_costs[policy_name].append(cost)

    return execution_costs