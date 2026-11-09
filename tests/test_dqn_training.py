import unittest
import math
import torch

from unittest.mock import patch
from execai.agents.dqn import DQNAgent, QNetwork
from execai.agents.dqn_training import (
    DQNTrainer,
    calculate_dqn_targets,
)
from execai.agents.replay_buffer import ReplayBuffer, Transition
from execai.environment.config import ExecutionConfig


class DQNTrainingTests(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.agent = DQNAgent(self.config, seed=42)
        self.replay_buffer = ReplayBuffer(capacity=10, seed=42)

    def add_terminal_experience(self):
        self.replay_buffer.add(
            observation={
                "time_step": 2,
                "remaining_lots": 2,
            },
            action=2,
            reward=-5.0,
            next_observation=None,
            done=True,
        )

    def test_terminal_target_uses_only_reward(self):
        transition = Transition(
            observation={
                "time_step": 2,
                "remaining_lots": 2,
            },
            action=2,
            reward=-5.0,
            next_observation=None,
            done=True,
        )

        target_network = QNetwork(self.config.num_lots)

        with patch.object(
            target_network,
            "forward",
            side_effect=AssertionError("Target network should not be called."),
        ):
            targets = calculate_dqn_targets(
                [transition],
                target_network,
                self.config,
                discount_factor=1.0,
            )

        torch.testing.assert_close(
            targets,
            torch.tensor([-5.0]),
        )

    def test_target_ignores_invalid_future_actions(self):
        transition = Transition(
            observation={
                "time_step": 0,
                "remaining_lots": 2,
            },
            action=0,
            reward=-2.0,
            next_observation={
                "time_step": 1,
                "remaining_lots": 2,
            },
            done=False,
        )

        target_network = QNetwork(self.config.num_lots)

        # actions 3 and 4 are invalid because only 2 lots remain
        predicted_values = torch.tensor([0.0, 5.0, 4.0, 100.0, 200.0])

        with patch.object(
            target_network,
            "forward",
            return_value=predicted_values,
        ):
            targets = calculate_dqn_targets(
                [transition],
                target_network,
                self.config,
                discount_factor=0.5,
            )

        # best valid future Q-value is 5
        # target = -2 + 0.5 * 5 = 0.5
        self.assertAlmostEqual(targets[0].item(), 0.5)

    def test_final_step_target_requires_remaining_inventory(self):
        transition = Transition(
            observation={
                "time_step": 1,
                "remaining_lots": 3,
            },
            action=0,
            reward=-1.0,
            next_observation={
                "time_step": 2,
                "remaining_lots": 3,
            },
            done=False,
        )

        target_network = QNetwork(self.config.num_lots)

        predicted_values = torch.tensor([2.0, 3.0, 4.0, 5.0, 100.0])

        with patch.object(
            target_network,
            "forward",
            return_value=predicted_values,
        ):
            targets = calculate_dqn_targets(
                [transition],
                target_network,
                self.config,
                discount_factor=1.0,
            )

        # only action 3 is valid at final decision
        self.assertAlmostEqual(targets[0].item(), 4.0)

    def test_training_waits_for_enough_experiences(self):
        trainer = DQNTrainer(
            self.agent,
            self.replay_buffer,
            batch_size=2,
        )

        self.assertIsNone(trainer.train_step())

        self.add_terminal_experience()

        self.assertIsNone(trainer.train_step())
        self.assertEqual(trainer.update_count, 0)

    def test_training_updates_network_weights(self):
        self.add_terminal_experience()

        # initializes weights to zero for a predictable gradient test
        with torch.no_grad():
            for parameter in self.agent.network.parameters():
                parameter.zero_()

        trainer = DQNTrainer(
            self.agent,
            self.replay_buffer,
            batch_size=1,
        )

        original_parameters = [
            parameter.detach().clone() for parameter in self.agent.network.parameters()
        ]

        loss = trainer.train_step()

        parameters_changed = any(
            not torch.equal(original, updated)
            for original, updated in zip(
                original_parameters,
                self.agent.network.parameters(),
            )
        )

        self.assertIsNotNone(loss)
        self.assertGreater(loss, 0)
        self.assertTrue(parameters_changed)
        self.assertEqual(trainer.update_count, 1)

    def test_target_network_syncs_at_configured_interval(self):
        self.add_terminal_experience()

        with torch.no_grad():
            for parameter in self.agent.network.parameters():
                parameter.zero_()

        trainer = DQNTrainer(
            self.agent,
            self.replay_buffer,
            batch_size=1,
            target_sync_interval=2,
        )

        trainer.train_step()

        # online network changed but target network has not
        networks_differ = any(
            not torch.equal(online, target)
            for online, target in zip(
                self.agent.network.parameters(),
                trainer.target_network.parameters(),
            )
        )

        self.assertTrue(networks_differ)

        trainer.train_step()

        # after the second update the target network is synchronized
        for online, target in zip(
            self.agent.network.parameters(),
            trainer.target_network.parameters(),
        ):
            torch.testing.assert_close(online, target)

        self.assertEqual(trainer.update_count, 2)

    def test_rejects_invalid_training_parameters(self):
        invalid_parameters = [
            {"learning_rate": -0.1},
            {"learning_rate": float("nan")},
            {"discount_factor": 1.5},
            {"batch_size": 0},
            {"target_sync_interval": 0},
        ]

        for parameters in invalid_parameters:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValueError):
                    DQNTrainer(
                        self.agent,
                        self.replay_buffer,
                        **parameters,
                    )

    def test_liquidity_aware_training_step(self):
        agent = DQNAgent(
            self.config,
            seed=42,
            use_liquidity=True,
        )

        replay_buffer = ReplayBuffer(capacity=10, seed=42)

        replay_buffer.add(
            observation={
                "time_step": 0,
                "remaining_lots": 4,
                "relative_liquidity": 0.5,
            },
            action=1,
            reward=-2.0,
            next_observation={
                "time_step": 1,
                "remaining_lots": 3,
                "relative_liquidity": 1.5,
            },
            done=False,
        )

        trainer = DQNTrainer(
            agent,
            replay_buffer,
            batch_size=1,
        )

        loss = trainer.train_step()

        self.assertIsNotNone(loss)
        self.assertTrue(math.isfinite(loss))
        self.assertEqual(trainer.update_count, 1)

    def test_liquidity_aware_targets_use_three_features(self):
        transition = Transition(
            observation={
                "time_step": 0,
                "remaining_lots": 2,
                "relative_liquidity": 0.5,
            },
            action=0,
            reward=-2.0,
            next_observation={
                "time_step": 1,
                "remaining_lots": 2,
                "relative_liquidity": 1.5,
            },
            done=False,
        )

        target_network = QNetwork(
            self.config.num_lots,
            use_liquidity=True,
        )

        targets = calculate_dqn_targets(
            [transition],
            target_network,
            self.config,
            discount_factor=1.0,
        )

        self.assertEqual(targets.shape, (1,))
        self.assertTrue(torch.isfinite(targets).all().item())


if __name__ == "__main__":
    unittest.main()
