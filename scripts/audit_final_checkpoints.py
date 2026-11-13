"""verifies saved DQN models before final historical evaluation"""

import math

from pathlib import Path

from execai.agents.checkpoints import load_dqn_checkpoint
from execai.environment.config import ExecutionConfig

from scripts.analyze_dqn_robustness import TRAINING_SEEDS
from scripts.evaluate_historical import (
    PROJECT_ROOT,
    build_evaluation_settings,
)

MODEL_DIRECTORY = PROJECT_ROOT / "results" / "models"

TRAINING_EPISODES = 2000
VALIDATION_INTERVAL = 200

MODEL_CONFIGURATIONS = (
    ("Basic DQN", "basic", False),
    ("Liquidity-Aware DQN", "liquidity_aware", True),
)


def audit_checkpoints(
    model_directory: Path,
    config: ExecutionConfig,
    training_seeds: tuple[int, ...] = TRAINING_SEEDS,
) -> list[dict]:
    """validates every saved model and its training metadata"""

    checkpoint_records = []

    for training_seed in training_seeds:
        for model_name, filename, use_liquidity in MODEL_CONFIGURATIONS:
            checkpoint_path = (
                model_directory / f"dqn_{filename}_seed_{training_seed}.pt"
            )

            if not checkpoint_path.is_file():
                raise FileNotFoundError(f"Missing DQN checkpoint: {checkpoint_path}")

            agent, metadata = load_dqn_checkpoint(checkpoint_path)

            if agent.config != config:
                raise ValueError(
                    f"Incorrect execution configuration: {checkpoint_path}"
                )

            if agent.use_liquidity != use_liquidity:
                raise ValueError(f"Incorrect state features: {checkpoint_path}")

            if metadata["training_seed"] != training_seed:
                raise ValueError(f"Incorrect training seed: {checkpoint_path}")

            best_episode = metadata["best_episode"]

            if (
                type(best_episode) is not int
                or not VALIDATION_INTERVAL <= best_episode <= TRAINING_EPISODES
                or best_episode % VALIDATION_INTERVAL != 0
            ):
                raise ValueError(f"Invalid checkpoint episode: {checkpoint_path}")

            validation_cost = metadata["validation_cost_bps"]

            if (
                isinstance(validation_cost, bool)
                or not isinstance(validation_cost, (int, float))
                or not math.isfinite(validation_cost)
            ):
                raise ValueError(f"Invalid validation cost: {checkpoint_path}")

            checkpoint_records.append(
                {
                    "model": model_name,
                    "training_seed": training_seed,
                    "best_episode": best_episode,
                    "validation_cost_bps": validation_cost,
                }
            )

    return checkpoint_records


def main() -> None:
    config, _ = build_evaluation_settings()

    records = audit_checkpoints(
        MODEL_DIRECTORY,
        config,
    )

    print("Final Model Checkpoint Audit")
    print("----------------------------")
    print(f"Training seeds: {len(TRAINING_SEEDS)}")
    print(f"Expected checkpoints: {len(TRAINING_SEEDS) * len(MODEL_CONFIGURATIONS)}")

    for record in records:
        print(
            f"{record['model']} | "
            f"Seed {record['training_seed']} | "
            f"Episode {record['best_episode']} | "
            f"Validation {record['validation_cost_bps']:.4f} bps"
        )

    print(f"\nVerified checkpoints: {len(records)}")
    print("All saved DQN checkpoints passed validation.")


if __name__ == "__main__":
    main()
