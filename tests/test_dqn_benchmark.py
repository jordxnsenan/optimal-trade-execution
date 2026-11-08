import unittest
import torch

from functools import partial
from unittest.mock import patch

from scripts.benchmark_dqn import run_execution_episode
from execai.agents.dqn import DQNAgent
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.policies.baselines import immediate_action, twap_action


class DQNBenchmarkTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.cost_config = ExecutionCostConfig(
            half_spread_bps=5.0,
            temporary_impact_bps=20.0,
            fee_per_share=0.01,
        )

        self.environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        self.prices = [100.0] * self.config.num_time_steps

    def test_immediate_execution_cost(self):
        actions, cost = run_execution_episode(
            self.environment,
            self.prices,
            immediate_action,
        )

        self.assertEqual(sum(actions), self.config.num_lots)
        self.assertAlmostEqual(cost, 26.0)

    def test_twap_reaches_known_optimum(self):
        actions, cost = run_execution_episode(
            self.environment,
            self.prices,
            partial(twap_action, config=self.config),
        )

        self.assertEqual(sum(actions), self.config.num_lots)
        self.assertAlmostEqual(cost, 13.5)

    def test_dqn_evaluation_disables_exploration(self):
        torch.manual_seed(42)
        agent = DQNAgent(self.config, seed=42)

        with patch.object(
            agent,
            "select_action",
            wraps=agent.select_action,
        ) as select_action:
            actions, cost = run_execution_episode(
                self.environment,
                self.prices,
                partial(agent.select_action, exploration_rate=0.0),
            )

        self.assertEqual(sum(actions), self.config.num_lots)
        self.assertGreater(cost, 0)

        for recorded_call in select_action.call_args_list:
            self.assertEqual(
                recorded_call.kwargs["exploration_rate"],
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
