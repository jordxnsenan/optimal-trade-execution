import unittest
import torch

from unittest.mock import patch
from execai.agents.dqn import DQNAgent
from execai.agents.dqn_episodes import train_dqn
from execai.agents.dqn_training import DQNTrainer
from execai.agents.replay_buffer import ReplayBuffer
from execai.environment.config import ExecutionConfig
from execai.environment.costs import ExecutionCostConfig
from execai.environment.simulator import ExecutionEnvironment


class DQNEpisodeTrainingTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

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

        self.agent = DQNAgent(self.config, seed=42)
        self.replay_buffer = ReplayBuffer(capacity=100, seed=42)

        self.trainer = DQNTrainer(
            self.agent,
            self.replay_buffer,
            batch_size=2,
            target_sync_interval=5,
        )

        self.environment = ExecutionEnvironment(
            self.config,
            self.cost_config,
        )

    def test_training_completes_multiple_episodes(self):
        price_paths = [[100.0, 100.0, 100.0] for _ in range(5)]

        episode_costs = train_dqn(
            self.agent,
            self.trainer,
            self.environment,
            price_paths,
        )

        self.assertEqual(len(episode_costs), 5)
        self.assertTrue(self.environment.is_done)
        self.assertEqual(self.environment.inventory.remaining_shares, 0)

        self.assertGreater(len(self.replay_buffer), 0)
        self.assertGreater(self.trainer.update_count, 0)

    def test_training_stores_terminal_transitions(self):
        config = ExecutionConfig(
            total_shares=400,
            num_time_steps=1,
            num_lots=4,
        )

        agent = DQNAgent(config, seed=42)
        replay_buffer = ReplayBuffer(capacity=10, seed=42)

        trainer = DQNTrainer(
            agent,
            replay_buffer,
            batch_size=1,
        )

        environment = ExecutionEnvironment(
            config,
            self.cost_config,
        )

        episode_costs = train_dqn(
            agent,
            trainer,
            environment,
            [[100.0], [100.0]],
        )

        self.assertEqual(episode_costs, [26.0, 26.0])
        self.assertEqual(len(replay_buffer), 2)
        self.assertEqual(trainer.update_count, 2)

        for transition in replay_buffer.transitions:
            self.assertTrue(transition.done)
            self.assertIsNone(transition.next_observation)
            self.assertEqual(transition.action, 4)

    def test_exploration_rate_decreases(self):
        config = ExecutionConfig(
            total_shares=400,
            num_time_steps=1,
            num_lots=4,
        )

        agent = DQNAgent(config, seed=42)
        replay_buffer = ReplayBuffer(capacity=10, seed=42)
        trainer = DQNTrainer(agent, replay_buffer, batch_size=1)

        environment = ExecutionEnvironment(
            config,
            self.cost_config,
        )

        with patch.object(
            agent,
            "select_action",
            wraps=agent.select_action,
        ) as select_action:
            train_dqn(
                agent,
                trainer,
                environment,
                [[100.0], [100.0], [100.0]],
                initial_exploration_rate=1.0,
                final_exploration_rate=0.0,
            )

        used_rates = [
            recorded_call.kwargs["exploration_rate"]
            for recorded_call in select_action.call_args_list
        ]

        self.assertEqual(used_rates, [1.0, 0.5, 0.0])

    def test_training_is_reproducible(self):
        price_paths = [[100.0, 100.5, 101.0] for _ in range(5)]

        def run_training():
            torch.manual_seed(42)

            agent = DQNAgent(self.config, seed=42)
            replay_buffer = ReplayBuffer(capacity=100, seed=42)

            trainer = DQNTrainer(
                agent,
                replay_buffer,
                batch_size=2,
                target_sync_interval=5,
            )

            environment = ExecutionEnvironment(
                self.config,
                self.cost_config,
            )

            costs = train_dqn(
                agent,
                trainer,
                environment,
                price_paths,
            )

            parameters = [
                parameter.detach().clone() for parameter in agent.network.parameters()
            ]

            return costs, parameters

        first_costs, first_parameters = run_training()
        second_costs, second_parameters = run_training()

        self.assertEqual(first_costs, second_costs)

        for first, second in zip(
            first_parameters,
            second_parameters,
        ):
            torch.testing.assert_close(first, second)

    def test_rejects_missing_training_paths(self):
        with self.assertRaisesRegex(ValueError, "price path"):
            train_dqn(
                self.agent,
                self.trainer,
                self.environment,
                [],
            )

    def test_rejects_invalid_exploration_rates(self):
        invalid_rates = [
            (-0.1, 0.0),
            (1.1, 0.1),
            (0.2, 0.5),
            (float("nan"), 0.0),
        ]

        for initial_rate, final_rate in invalid_rates:
            with self.subTest(
                initial_rate=initial_rate,
                final_rate=final_rate,
            ):
                with self.assertRaises(ValueError):
                    train_dqn(
                        self.agent,
                        self.trainer,
                        self.environment,
                        [[100.0, 100.0, 100.0]],
                        initial_exploration_rate=initial_rate,
                        final_exploration_rate=final_rate,
                    )

    def test_requires_matching_environment_configuration(self):
        different_config = ExecutionConfig(
            total_shares=500,
            num_time_steps=3,
            num_lots=5,
        )

        different_environment = ExecutionEnvironment(different_config)

        with self.assertRaisesRegex(ValueError, "configurations must match"):
            train_dqn(
                self.agent,
                self.trainer,
                different_environment,
                [[100.0, 100.0, 100.0]],
            )


if __name__ == "__main__":
    unittest.main()
