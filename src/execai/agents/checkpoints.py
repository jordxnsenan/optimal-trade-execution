import torch

from pathlib import Path
from execai.agents.dqn import DQNAgent, QNetwork
from execai.environment.config import ExecutionConfig


def save_dqn_checkpoint(
    agent: DQNAgent,
    path: Path,
    training_seed: int,
    best_episode: int,
    validation_cost_bps: float,
) -> None:
    """saves a trained DQN and its configuration"""

    checkpoint = {
        "config": {
            "total_shares": agent.config.total_shares,
            "num_time_steps": agent.config.num_time_steps,
            "num_lots": agent.config.num_lots,
        },
        "use_liquidity": agent.use_liquidity,
        "hidden_units": agent.network.hidden_units,
        "training_seed": training_seed,
        "best_episode": best_episode,
        "validation_cost_bps": validation_cost_bps,
        "network_weights": agent.network.state_dict(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_dqn_checkpoint(
    path: Path,
) -> tuple[DQNAgent, dict]:
    """loads a saved DQN for evaluation"""

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    config = ExecutionConfig(**checkpoint["config"])

    network = QNetwork(
        num_lots=config.num_lots,
        hidden_units=checkpoint["hidden_units"],
        use_liquidity=checkpoint["use_liquidity"],
    )

    network.load_state_dict(checkpoint["network_weights"])
    network.eval()

    agent = DQNAgent(
        config,
        network=network,
        use_liquidity=checkpoint["use_liquidity"],
    )

    metadata = {
        "training_seed": checkpoint["training_seed"],
        "best_episode": checkpoint["best_episode"],
        "validation_cost_bps": checkpoint["validation_cost_bps"],
    }

    return agent, metadata
