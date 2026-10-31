# Tabular Q-Learning: Experimental Results

## Introduction

This report covers the implementation and testing of tabular Q-learning for ExecAI which is focused on optimal trade execution. The idea behind the project is to train an agent to decide how much of a stock order to sell at different points in time, with the goal of keeping execution costs minimal.

The agent learns by interacting with a simulated market. At each time step it chooses how many shares to sell and receives a reward based on the resulting execution cost. Over many episodes, it updates its Q-values and gradually learns which decisions tend to produce better outcomes.

I started with a fairly small execution problem to make sure the learning algorithm worked correctly before moving to the full-sized environment. I was also interested in seeing how well tabular Q-learning would scale, especially since the number of possible trading decisions increases quite a bit as the order size and execution window get larger.

To measure performance, I compared the agent against two basic execution strategies. The first was Immediate Execution which sells the entire order at once. The second was Time-Weighted Average Price (TWAP) which spreads the order across the available time intervals.

Both baselines use the same trading simulator as the Q-learning agent, so execution costs are calculated consistently.

The results discussed here are from the development stage of the project. A separate set of market scenarios will be used for the final evaluation.

## Experimental Setup

### Trading Environment

I used two different execution configurations for these experiments. The smaller version was mainly for checking whether the agent could learn a known optimal solution. The larger version represents the execution problem I intend to use for the main project.

| Parameter      | Small Benchmark | Full-Size Problem |
|----------------|-----------------|-------------------|
| Order size     | 400 shares      | 10,000 shares     |
| Time steps     | 3               | 12                |
| Inventory lots | 4               | 20                |
| Shares per lot | 100             | 500               |
| Starting price | $100            | $100              |

In the full-sized environment each time step represents five minutes which gives the agent one hour to complete the order.

The agent chooses how many inventory lots to sell at each step. It can also choose to wait, although it cannot sell more shares than it has remaining. If the deadline is reached before the order is complete, the simulator automatically executes whatever inventory is left. The purpose of this is to ensures that every strategy completes the same total order, even if its trading decisions are different.

### Execution Costs

The simulator includes a simplified transaction-cost model. Selling shares incurs a half-spread cost and larger trades experience greater temporary market impact. There is also a fixed fee for each share sold.

For the initial experiments, I used the following values:

| Parameter                    | Value           |
|------------------------------|-----------------|
| Half-spread                  | 5 basis points  |
| Temporary impact coefficient | 20 basis points |
| Transaction fee              | $0.01 per share |

Temporary impact depends on how much of the original order is executed in a single trade. This means selling a large portion of the order at once is generally more expensive than dividing it into smaller trades assuming the market price stays the same.

I measured execution performance using implementation shortfall expressed in basis points. It measures the difference between the value of the order at the starting price and the net proceeds received from executing it. A lower value means the strategy achieved a better execution result.

### Q-Learning Agent

For the first version of Q-learning I kept the state representation simple. Each state contains the current time step and the number of inventory lots remaining.

The available actions are the different quantities of inventory the agent can sell. These are restricted to prevent the agent from executing more shares than it has available. At the final step, the agent must sell its remaining inventory.

I implemented the Q-table using Python dictionaries with Q-values initially set to zero. The agent uses epsilon-greedy action selection during training, allowing it to explore different trades while gradually placing more weight on what it has learned.

The initial training settings were:

| Hyperparameter            | Value          |
|---------------------------|----------------|
| Learning rate             | 0.1            |
| Discount factor           | 1.0            |
| Initial exploration rate  | 1.0            |
| Final exploration rate    | 0.1            |
| Exploration schedule      | Linear decay   |
| Action selection          | Epsilon-greedy |

I used a discount factor of 1.0 because the agent has a fixed execution deadline and the objective is to minimize the total cost of completing the order.

One limitation I found with this first implementation is that the state does not contain the current market price or volatility. The agent can learn how to distribute trades based on time and remaining inventory but its decisions cannot directly respond to changes in market conditions. I wanted to establish how well this simpler version worked before adding more information to the state.

### Training and Evaluation

The experiments use both constant-price markets and randomly generated price paths. Synthetic price movements are produced using a seeded geometric Brownian motion model which allows the same experiments to be reproduced.

The constant-price environment is especially useful because the optimal execution schedule can be calculated independently. This gives me a way to check whether the Q-learning agent is learning reasonable decisions.

For the stochastic experiments, I generated separate market paths for training and evaluation. Each strategy was evaluated using the same price paths so that differences in performance could be compared directly.

I also trained multiple agents using different random seeds to investigate whether the results were consistent across training runs.

Since some of these evaluation scenarios have already been used while developing and adjusting the agent, I am treating them as validation data. My final evaluation will use new, untouched scenarios.

## Small Deterministic Benchmark

### Experiment

The first experiment was designed to check whether Q-learning could learn an optimal trading schedule in a simple environment.

I used an order of 400 shares divided into four lots with three time steps available to complete the execution. The market price stayed constant at $100 throughout each episode.

The agent was trained for 5,000 episodes using random seed 42.

Because the price did not change the main factor separating the strategies was how they distributed their trades. Larger trades produced greater temporary impact so spreading execution across the available intervals should reduce costs.

Before training the agent I had already implemented an exhaustive-search test that calculated the cost of every possible complete execution schedule in this environment.

That test established a minimum execution cost of **13.5 basis points**. It also confirmed that several different schedules could achieve the same result.

### Results

After training, I evaluated the learned Q-learning policy against Immediate Execution and TWAP.

| Strategy            | Execution Schedule (Shares) | Shortfall |
|---------------------|-----------------------------|-----------|
| Immediate Execution | [400]                       | 26.0 bps  |
| TWAP                | [200, 100, 100]             | 13.5 bps  |
| Q-Learning          | [100, 100, 200]             | 13.5 bps  |
| Known Optimum       | Multiple valid schedules    | 13.5 bps  |

The Q-learning agent learned to sell 100 shares during each of the first two intervals, leaving 200 shares for the final interval.

This schedule was different from TWAP, which executed 200 shares immediately and then sold 100 shares in each remaining interval. Despite the difference, both strategies achieved the same execution cost.

The learned policy matched the minimum cost that had been established through exhaustive search.

### Discussion

I found thiss initial result to be interesting because it showed that the Q-learning implementation was capable of learning a sensible execution strategy. The agent had no built-in knowledge of TWAP or the optimal schedule. It arrived at its decisions through repeated interaction with the simulator.

The result also gave me more confidence in the learning process. The Q-value updates were working as expected in the unit tests and the agent was now able to produce an optimal policy in a complete execution episode.

A clear limitation that I discovered with this approach was that the problem is relatively predictable with a constant market price and the same liquidity assumptions at every step as the agent only needs to learn how to distribute its inventory efficiently.

Matching TWAP in this setting was useful as a starting point but it does not tell me much about how the agent will behave in a changing market.

For the next experiments I intend on using randomly generated prices and a larger execution problem. I wnat to see whether the agent can maintain similar performance once it had more decisions to make and the consequences of those decisions became harder to learn.

## Full-Size Q-Learning Experiments

### Scaling Up the Execution Problem

After getting Q-learning working in the smaller environment I moved on to the full-sized execution problem. This involved selling 10,000 shares over twelve time steps where the order was divided into twenty inventory lots.

The main difference was the number of possible decisions the agent had to learn. In the smaller environment, there were only four lots and three time steps, meaning that the the agent had relatively few trading schedules to consider. With twenty lots and twelve steps, there are considerably more state-action combinations. The agent also needs to learn how decisions made early in the episode affect the execution cost later on.

I initially trained the agent using 5,000 episodes with randomly generated market prices. I used five different agent seeds to see whether training produced similar results across independent runs. Each agent was evaluated on the same 500 synthetic validation paths.

The results were noticeably worse than those from the small benchmark. Although the agent was able to complete every order, many of its learned schedules involved waiting through most of the execution window before selling large quantities of shares which was expensive because of the temporary market-impact model.

For comparison, TWAP achieved an average implementation shortfall of 8.2047 basis points on these validation paths and Immediate Execution cost 26.0 basis points.

The initial Q-learning agents were consistently worse than TWAP. Some learned schedules placed several thousand shares into a single trading interval, even though spreading those trades would have reduced their impact costs in a constant-price market.

### Increasing the Training Budget

My first thought was that the agent might need more training to learn the larger state-action space properly. I compared training runs using 5,000 and 20,000 episodes per agent. Both experiments used the same initial market-path seed ranges along with the existing learning rate and exploration settings.

| Training Episodes | Mean Difference from TWAP |
|-------------------|---------------------------|
| 5,000             | +8.5917 bps               |
| 20,000            | +8.0659 bps               |

Increasing the number of episodes produced a small improvement in the average result but the learned schedules were still inefficient. There was also considerable variation between agents.

As an example, after 20,000 episodes one agent learned to execute 9,500 of its 10,000 shares in a single interval. Its execution cost in the constant-price reference market was 24.1 basis points, compared with the known optimum of 7.8 basis points.

I had expected the longer training run to produce a more noticeable improvement but instead it suggested that increasing the number of episodes alone was unlikely to solve the problem.

### Training Without Price Volatility

I wanted to check whether the changing market prices were making Q-learning harder to train.

To investigate this I repeated the 20,000-episode experiment with training volatility set to zero. Every training episode now used a constant market price of $100.

I kept the original stochastic validation paths which had a volatility setting of 0.002 per time step which allowed me to examine the learned schedules in a simpler training environment while continuing to compare their performance under changing prices.

Removing price volatility improved the results although none of the five agents reached the optimal flat-market execution cost.

| Agent Seed | Flat-Market Cost | Difference from 7.8 bps Optimum |
|------------|------------------|---------------------------------|
| 42         | 9.3 bps          | +1.5 bps                        |
| 43         | 11.9 bps         | +4.1 bps                        |
| 44         | 12.7 bps         | +4.9 bps                        |
| 45         | 13.7 bps         | +5.9 bps                        |
| 46         | 15.8 bps         | +8.0 bps                        |

As can be seen in the results table, seed 42 produced the best result, but even this policy still concentrated some trades into relatively large quantities.

I found this experiment to be useful because it showed that unpredictable price movements were contributing to the learning difficulty. At the same time, the agent continued to struggle when those movements were removed.

I decided to look more closely at the Q-table before making further changes to the training algorithm.

### Investigating the Q-Values

I added diagnostic counters to record how often each state-action pair was updated during training.

Prior to this I was checking how many states appeared in the Q-table. Most training runs had visited around 220 or 221 states which initially made it seem like the agent had explored nearly everything.

The new diagnostics showed that all 2,341 recorded state-action pairs had received at least one update in the deterministic training experiment but some actions were still being updated much more frequently than others.

At the initial state the action corresponding to selling zero lots received roughly 9,000 updates in a 20,000-episode run. Selling two lots received only around 600 updates.

I also noticed that the estimated Q-values for waiting were much higher than their actual optimal values.

For example, in seed 42, the estimated Q-value for selling zero lots at the initial step was approximately -1.60. In the constant-price environment, the best possible total reward after choosing that action is -7.9.

The Q-value for selling the entire order immediately was correctly estimated at -26.0. That action is easier to learn because the episode ends immediately and the agent receives the full execution cost.

Waiting is more complicated. Its immediate reward is zero, and the cost of completing the order comes later. The agent needs to learn those future costs through repeated Q-value updates.

The results suggested that the estimated future costs were not being propagated accurately enough through the longer execution horizon.

Another possible contributor was the way the agent handled actions with equal Q-values. Initially, every Q-value was zero, and the greedy action-selection method always selected the smallest action when values were tied. In our environment, the smallest action is often to sell nothing.

I wanted to see whether removing that preference would improve learning.

### Randomized Tie-Breaking

I modified the Q-learning agent to support randomized greedy tie-breaking.

Previously, when several actions had the same highest Q-value, the agent always selected the smallest action. With the new option enabled, it randomly selects one of the equally valued actions.

I kept the original behavior available so that the two approaches could be compared using the same training configuration.

Both experiments used 20,000 episodes per agent with zero training volatility. Evaluation was performed on the same stochastic validation paths.

| Agent Seed  | Original Difference from TWAP | Randomized Tie-Breaking |
|-------------|-------------------------------|-------------------------| 
| 42          | +1.9412 bps                   | +4.5645 bps             |
| 43          | +4.2885 bps                   | +1.2646 bps             |
| 44          | +5.0068 bps                   | +1.7981 bps             |
| 45          | +6.2274 bps                   | +2.7027 bps             |
| 46          | +7.9187 bps                   | +1.9077 bps             |
| **Average** | **+5.0765 bps**               | **+2.4475 bps**         |

Randomized tie-breaking improved the average difference from TWAP by approximately 2.63 basis points.

The improvement was apparent in the learned schedules as well. Several agents began spreading their trades across more intervals instead of leaving most of the inventory until near the deadline, however this improvement was not consistent for every seed. Seed 42 actually performed worse with randomized tie-breaking, although the other four agents improved.

This was an interesting result because the underlying Q-learning update equation had not changed. The improvement came from changing how the agent selected between actions with identical estimated values.

It also showed how much the learning process could depend on relatively small implementation choices.


## Reproducing the Experiments

I created Python scripts for running the Q-learning experiments so that I could repeat tests without having to modify the source code each time. All experiments were run using the ExecAI simulator and its existing transaction-cost model. The scripts use fixed random seeds to make the results reproducible.

### Development Environment

The project was developed using Python 3.14.3 on macOS with a virtual environment for managing Python packages.

To set up the project from the repository root, run the following:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The unit tests can be run using:

```bash
python -m unittest discover -s tests -v
```

These tests cover the simulator and execution-cost calculations as well as the Q-learning implementation.

### Small Benchmark

The original deterministic experiment can be reproduced using:

```bash
python scripts/compare_q_learning.py
```

This trains a Q-learning agent for 5,000 episodes in the small constant-price environment. The script prints the learned execution schedule and compares its cost against Immediate Execution and TWAP. The expected optimal execution cost is 13.5 basis points.

### Full-Size Experiments

The full-sized experiments use 10,000 shares divided into twenty inventory lots with twelve decision intervals.

To run the original experiment with 5,000 training episodes:

```bash
python scripts/evaluate_q_learning.py --full-size --training-episodes 5000
```

To repeat it with 20,000 training episodes:

```bash
python scripts/evaluate_q_learning.py --full-size --training-episodes 20000
```

For the experiment with zero training volatility:

```bash
python scripts/evaluate_q_learning.py --full-size --training-episodes 20000 --training-volatility 0
```

To enable randomized greedy tie-breaking:

```bash
python scripts/evaluate_q_learning.py --full-size --training-episodes 20000 --training-volatility 0 --random-tie-breaking
```

The evaluation script trains five agents using different seeds. It reports each agent's learned execution schedule and its performance against the existing baselines.

All agents are evaluated on the same 500 stochastic validation paths, allowing their execution costs to be compared using the same market scenarios.