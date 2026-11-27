import torch
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from execai.agents.checkpoints import (
    load_dqn_checkpoint,
    save_dqn_checkpoint,
)
from execai.agents.dqn import DQNAgent, encode_observation
from execai.environment.config import ExecutionConfig


class DQNCheckpointTests(unittest.TestCase):

    def setUp(self):
        self.config = ExecutionConfig(
            total_shares=400,
            num_time_steps=3,
            num_lots=4,
        )

        self.observation = {
            "time_step": 1,
            "remaining_lots": 2,
            "relative_liquidity": 1.5,
        }

    def test_save_and_load_both_dqn_configurations(self):
        for use_liquidity in (False, True):
            with self.subTest(use_liquidity=use_liquidity):
                torch.manual_seed(42)

                original_agent = DQNAgent(
                    self.config,
                    seed=42,
                    use_liquidity=use_liquidity,
                )

                state = encode_observation(
                    self.observation,
                    self.config,
                    use_liquidity=use_liquidity,
                )

                with torch.no_grad():
                    original_values = original_agent.network(state)

                with TemporaryDirectory() as directory:
                    checkpoint_path = Path(directory) / "models" / "dqn.pt"

                    save_dqn_checkpoint(
                        original_agent,
                        checkpoint_path,
                        training_seed=42,
                        best_episode=1800,
                        validation_cost_bps=6.5,
                    )

                    self.assertTrue(checkpoint_path.exists())

                    loaded_agent, metadata = load_dqn_checkpoint(checkpoint_path)

                with torch.no_grad():
                    loaded_values = loaded_agent.network(state)

                torch.testing.assert_close(
                    original_values,
                    loaded_values,
                )

                self.assertEqual(loaded_agent.config, self.config)
                self.assertEqual(
                    loaded_agent.use_liquidity,
                    use_liquidity,
                )

                self.assertEqual(
                    original_agent.select_action(self.observation),
                    loaded_agent.select_action(self.observation),
                )

                self.assertFalse(loaded_agent.network.training)

                self.assertEqual(metadata["training_seed"], 42)
                self.assertEqual(metadata["best_episode"], 1800)
                self.assertAlmostEqual(
                    metadata["validation_cost_bps"],
                    6.5,
                )


if __name__ == "__main__":
    unittest.main()
