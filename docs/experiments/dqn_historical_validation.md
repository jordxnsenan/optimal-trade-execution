# DQN Historical Validation Results

## Experimental Setup

I evaluated four independently trained models for each DQN configuration using 250 historical SPY trading sessions from 2025. Both configurations were trained on synthetic market scenarios calibrated using 2023–2024 data. Their checkpoints were selected using separate synthetic validation scenarios before historical evaluation.

Each historical episode simulated a 10,000-share sell order over twelve five-minute intervals. The evaluation used training-derived forecast liquidity and a linear temporary-impact coefficient of 20 basis points. Execution costs were modeled using the same simulator as the classical baselines.

## Results

| Training Seed | Basic DQN (bps) | Liquidity-Aware DQN (bps) |
| ------------- | --------------: | ------------------------: |
| 42            |          6.3213 |                    9.2095 |
| 7             |          5.7589 |                    6.2834 |
| 123           |          6.0310 |                    7.7349 |
| 2026          |          6.9051 |                    5.7711 |
| **Mean**      |      **6.2541** |                **7.2497** |

TWAP achieved 5.6426 basis points, while risk-neutral Almgren–Chriss achieved 5.6250 basis points on the same historical sessions.

The basic DQN performed better than the liquidity-aware configuration in three of the four training runs. It also showed less variation between independently trained models. Neither configuration consistently matched the classical baselines.

## Interpretation

The results suggest that the additional liquidity feature has not improved generalization under the current training setup. During synthetic training, liquidity varied between episodes, while historical evaluation used a fixed, training-derived liquidity forecast. This difference may have contributed to the weaker performance of the liquidity-aware agent.

The basic DQN will serve as the primary reinforcement learning configuration for subsequent evaluation. All four training seeds will be retained, and the liquidity-aware version will remain available for comparison.

These results use historical price movements with hypothetical execution costs. The 2025 data has been used for model development and comparison. The 2026 historical test period remains reserved for final evaluation after the experimental settings have been fixed.