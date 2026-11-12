import math
import torch
import unittest

from functools import partial
from unittest.mock import patch
from scripts.compare_dqn_liquidity import (
    average_actions_by_step,
    generate_market_scenarios,
    train_agent,
)
from scripts.compare_dqn_synthetic import evaluate_policies
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment
from execai.policies.baselines import immediate_action, twap_action


class LiquidityDQNComparisonTests(unittest.TestCase):

    def setUp(self):
        torch.set_num_threads(1)

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

    def test_market_scenarios_are_reproducible(self):
        expected_liquidity = [1.2, 1.0, 0.8]

        first_scenarios = generate_market_scenarios(
            self.config,
            0.001,
            expected_liquidity,
            num_episodes=5,
            price_seed=42,
            liquidity_seed=100,
        )

        second_scenarios = generate_market_scenarios(
            self.config,
            0.001,
            expected_liquidity,
            num_episodes=5,
            price_seed=42,
            liquidity_seed=100,
        )

        self.assertEqual(first_scenarios, second_scenarios)

    def test_liquidity_randomness_is_separate_from_prices(self):
        expected_liquidity = [1.0, 1.0, 1.0]

        first_prices, first_liquidity = generate_market_scenarios(
            self.config,
            0.001,
            expected_liquidity,
            num_episodes=5,
            price_seed=42,
            liquidity_seed=100,
        )

        second_prices, second_liquidity = generate_market_scenarios(
            self.config,
            0.001,
            expected_liquidity,
            num_episodes=5,
            price_seed=42,
            liquidity_seed=200,
        )

        self.assertEqual(first_prices, second_prices)
        self.assertNotEqual(first_liquidity, second_liquidity)

    def test_both_dqn_variants_complete_variable_liquidity_episodes(self):
        prices = [
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ]

        liquidity_profiles = [
            [0.5, 1.0, 1.5],
            [1.5, 1.0, 0.5],
            [1.0, 0.8, 1.2],
        ]

        agents = {}

        for use_liquidity in (False, True):

            agent, costs, _, validation_history = train_agent(
                self.config,
                self.cost_config,
                prices,
                liquidity_profiles,
                validation_paths=[[100.0, 100.0, 100.0]],
                validation_liquidity=[[1.0, 1.0, 1.0]],
                seed=42,
                use_liquidity=use_liquidity,
            )

            self.assertEqual(len(validation_history), 1)
            self.assertEqual(validation_history[0][0], len(prices))
            self.assertTrue(math.isfinite(validation_history[0][1]))
            self.assertEqual(len(costs), len(prices))
            self.assertTrue(all(math.isfinite(cost) for cost in costs))

            agents[use_liquidity] = agent

        self.assertFalse(agents[False].use_liquidity)
        self.assertTrue(agents[True].use_liquidity)

        policies = {
            "Basic DQN": partial(
                agents[False].select_action,
                exploration_rate=0.0,
            ),
            "Liquidity-Aware DQN": partial(
                agents[True].select_action,
                exploration_rate=0.0,
            ),
        }

        results = evaluate_policies(
            ExecutionEnvironment(self.config, self.cost_config),
            prices,
            policies,
            liquidity_profiles=liquidity_profiles,
        )

        self.assertEqual(len(results["Basic DQN"]), 3)
        self.assertEqual(len(results["Liquidity-Aware DQN"]), 3)

    def test_average_actions_for_known_policies(self):
        price_paths = [
            [100.0, 100.0, 100.0],
            [100.0, 100.0, 100.0],
        ]

        liquidity_profiles = [
            [1.0, 1.0, 1.0],
            [0.5, 1.5, 1.0],
        ]

        environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

        immediate_actions = average_actions_by_step(
            environment,
            price_paths,
            liquidity_profiles,
            immediate_action,
        )

        twap_actions = average_actions_by_step(
            environment,
            price_paths,
            liquidity_profiles,
            partial(twap_action, config=self.config),
        )

        self.assertEqual(immediate_actions, [4.0, 0.0, 0.0])
        self.assertEqual(twap_actions, [2.0, 1.0, 1.0])

    def test_average_actions_reflect_current_liquidity(self):
        config = ExecutionConfig(
            total_shares=400,
            num_time_steps=2,
            num_lots=4,
        )

        environment = ExecutionEnvironment(
            config,
            self.cost_config,
        )

        def liquidity_policy(observation):
            if (
                observation["time_step"] == 0
                and observation["relative_liquidity"] < 1.0
            ):
                return 0

            return observation["remaining_lots"]

        average_actions = average_actions_by_step(
            environment,
            price_paths=[
                [100.0, 100.0],
                [100.0, 100.0],
            ],
            liquidity_profiles=[
                [0.5, 1.0],
                [2.0, 1.0],
            ],
            policy=liquidity_policy,
        )

        self.assertEqual(average_actions, [2.0, 2.0])

    @patch("scripts.compare_dqn_liquidity.evaluate_policies")
    @patch("scripts.compare_dqn_liquidity.train_dqn")
    def test_training_restores_best_validation_checkpoint(
        self,
        mock_train_dqn,
        mock_evaluate_policies,
    ):
        mock_evaluate_policies.side_effect = [
            {"DQN": [12.0]},
            {"DQN": [6.0]},
            {"DQN": [10.0]},
        ]

        def simulate_training(
            agent,
            trainer,
            environment,
            price_paths,
            **kwargs,
        ):
            validation_callback = kwargs["validation_callback"]

            checkpoints = [
                (200, 1.0),
                (400, 2.0),
                (600, 3.0),
            ]

            for completed_episodes, weight_value in checkpoints:
                with torch.no_grad():
                    for parameter in agent.network.parameters():
                        parameter.fill_(weight_value)

                validation_callback(agent, completed_episodes)

            return [5.0] * len(price_paths)

        mock_train_dqn.side_effect = simulate_training

        agent, episode_costs, _, validation_history = train_agent(
            self.config,
            self.cost_config,
            price_paths=[[100.0, 100.0, 100.0]],
            liquidity_profiles=[[1.0, 1.0, 1.0]],
            validation_paths=[[100.0, 100.0, 100.0]],
            validation_liquidity=[[1.0, 1.0, 1.0]],
            seed=42,
            use_liquidity=True,
        )

        self.assertEqual(
            validation_history,
            [(200, 12.0), (400, 6.0), (600, 10.0)],
        )

        self.assertEqual(episode_costs, [5.0])

        # episode 400 had the best validation cost so returned network must contain its weights
        for parameter in agent.network.parameters():
            self.assertTrue(torch.all(parameter == 2.0).item())


if __name__ == "__main__":
    unittest.main()
