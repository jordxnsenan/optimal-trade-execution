# DQN Experimental Results

## Deterministic Benchmark

The first benchmark uses a 400-share sell order divided into four inventory lots and executed over three decisions. 
Market prices remain constant at $100. The known optimal implementation shortfall is 13.5 basis points.
The DQN agent was trained for 2,000 episodes under four random seeds.

| Seed | Early training cost | Late training cost | Greedy cost |
| ---- | ------------------: | -----------------: | ----------: |
| 42   |             19.4750 |            14.2250 |     13.5000 |
| 7    |             19.1750 |            14.2500 |     13.5000 |
| 123  |             19.1250 |            14.2750 |     13.5000 |
| 2026 |             19.7000 |            13.9750 |     13.5000 |

All four agents learned the optimal schedule [1, 1, 2].

Training costs include exploratory actions. The greedy policy was evaluated with exploration disabled. This verifies learning on the deterministic benchmark, but does not establish generalization to stochastic market prices.

### Full-Sized Deterministic Benchmark

I extended the deterministic benchmark to the project's full execution configuration of 10,000 shares, 20 inventory lots and twelve five-minute decisions.

With 2,000 training episodes and random seed 42, DQN achieved a greedy execution cost of 7.8 basis points, matching TWAP and the known minimum under constant prices and linear market impact. Average training costs decreased from 16.3920 basis points during the first 100 episodes to 9.2340 basis points during the final 100 episodes.

The result shows that DQN can learn an optimal execution schedule in the larger action space. The benchmark does not yet test performance under changing market prices.

### Stochastic Synthetic Market Evaluation

I trained DQN for 2,000 episodes using synthetic price paths generated from volatility estimated from the 2023–2024 SPY training dataset. 
The experiment used the full 10,000-share execution problem, with constant liquidity and linear market impact.

The trained policy was evaluated without exploration on 250 new synthetic price paths. All strategies used the same evaluation scenarios.

| Strategy          | Mean shortfall (bps) |
| ----------------- | -------------------: |
| Immediate         |              26.0000 |
| TWAP              |               6.9296 |
| VWAP              |               6.8811 |
| AC Risk-Neutral   |               6.9296 |
| AC Risk-Sensitive |               8.8593 |
| DQN               |               7.1646 |

DQN's mean shortfall was 0.2351 basis points higher than TWAP which suggests that DQN learned a competitive execution policy, although it did not outperform the simpler scheduling strategies.
The current DQN observes only time and remaining inventory.

### Common-Scenario Training-Seed Comparison

I repeated stochastic DQN training using four independent seeds. Each agent was evaluated on the same 250 synthetic price paths to separate differences in training outcomes from variation in evaluation scenarios.

| Training Seed | TWAP (bps) | DQN (bps) | DQN − TWAP (bps) |
| ------------- | ---------: | --------: | ---------------: |
| 42            |     7.1664 |    7.7040 |          +0.5375 |
| 7             |     7.1664 |   10.5545 |          +3.3881 |
| 123           |     7.1664 |    9.1991 |          +2.0327 |
| 2026          |     7.1664 |    7.3844 |          +0.2179 |

DQN was more expensive than TWAP in all four runs. The variation between learned policies suggests that training stability needs further investigation.