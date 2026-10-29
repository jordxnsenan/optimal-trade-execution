import unittest

from unittest.mock import patch

from execai.agents.q_learning import QLearningAgent
from execai.agents.training import train_q_learning
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.market import generate_price_path
from execai.environment.simulator import ExecutionEnvironment


class QLearningTrainingTests(unittest.TestCase):

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

    def test_training_runs_multiple_episodes(self):
        agent = QLearningAgent(self.config, seed=42)
        environment = ExecutionEnvironment(self.config, self.cost_config)

        price_paths = [[100.0, 100.0, 100.0] for _ in range(10)]

        episode_costs = train_q_learning(
            agent,
            environment,
            price_paths,
        )

        self.assertEqual(len(episode_costs), 10)
        self.assertGreater(len(agent.q_table), 0)

        self.assertTrue(environment.is_done)
        self.assertEqual(environment.inventory.remaining_shares, 0)

    def test_training_updates_q_values(self):
        config = ExecutionConfig(
            total_shares=400,
            num_time_steps=1,
            num_lots=4,
        )

        agent = QLearningAgent(
            config,
            learning_rate=0.5,
            seed=42,
        )

        environment = ExecutionEnvironment(config, self.cost_config)

        episode_costs = train_q_learning(
            agent,
            environment,
            [[100.0], [100.0]],
        )

        self.assertEqual(episode_costs, [26.0, 26.0])

        observation = {
            "time_step": 0,
            "remaining_lots": 4,
        }

        self.assertAlmostEqual(
            agent.get_q_value(observation, 4),
            -19.5,
        )

    def test_exploration_rate_decreases(self):
        config = ExecutionConfig(
            total_shares=400,
            num_time_steps=1,
            num_lots=4,
        )

        agent = QLearningAgent(config, seed=42)
        environment = ExecutionEnvironment(config, self.cost_config)

        with patch.object(
            agent,
            "select_action",
            wraps=agent.select_action,
        ) as select_action:
            train_q_learning(
                agent,
                environment,
                [[100.0], [100.0], [100.0]],
                initial_exploration_rate=1.0,
                final_exploration_rate=0.0,
            )

        used_rates = [
            call.args[1]
            for call in select_action.call_args_list
        ]

        self.assertEqual(used_rates, [1.0, 0.5, 0.0])

    def test_training_is_reproducible(self):
        price_paths = [
            generate_price_path(
                100.0,
                0.002,
                self.config.num_time_steps,
                seed=episode_index,
            )
            for episode_index in range(10)
        ]

        first_agent = QLearningAgent(self.config, seed=42)
        second_agent = QLearningAgent(self.config, seed=42)

        first_environment = ExecutionEnvironment(
            self.config, self.cost_config
        )

        second_environment = ExecutionEnvironment(
            self.config, self.cost_config
        )

        first_costs = train_q_learning(
            first_agent, first_environment, price_paths
        )

        second_costs = train_q_learning(
            second_agent, second_environment, price_paths
        )

        self.assertEqual(first_costs, second_costs)
        self.assertEqual(first_agent.q_table, second_agent.q_table)

    def test_training_requires_price_paths(self):
        agent = QLearningAgent(self.config)
        environment = ExecutionEnvironment(self.config, self.cost_config)

        with self.assertRaises(ValueError):
            train_q_learning(agent, environment, [])

    def test_invalid_exploration_rates(self):
        agent = QLearningAgent(self.config)
        environment = ExecutionEnvironment(self.config, self.cost_config)

        invalid_rates = [
            (-0.1, 0.0),
            (1.1, 0.1),
            (0.2, 0.5),
            (float("nan"), 0.0),
        ]

        for initial_rate, final_rate in invalid_rates:
            with self.subTest(initial_rate=initial_rate, final_rate=final_rate):
                with self.assertRaises(ValueError):
                    train_q_learning(
                        agent,
                        environment,
                        [[100.0, 100.0, 100.0]],
                        initial_exploration_rate=initial_rate,
                        final_exploration_rate=final_rate,
                    )

    def test_agent_and_environment_must_match(self):
        agent = QLearningAgent(self.config)

        different_config = ExecutionConfig(
            total_shares=500,
            num_time_steps=3,
            num_lots=5,
        )

        environment = ExecutionEnvironment(different_config)

        with self.assertRaises(ValueError):
            train_q_learning(
                agent,
                environment,
                [[100.0, 100.0, 100.0]],
            )


if __name__ == "__main__":
    unittest.main()