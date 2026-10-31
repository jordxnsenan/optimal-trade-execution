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