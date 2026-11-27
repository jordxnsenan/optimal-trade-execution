# Historical Market Data

## Data Source

I collected historical SPY market data from Alpaca's Historical Market Data API. I chose Alpaca because its SIP feed provides five-minute price and volume data covering the period needed for my execution experiments.

The requests use the following settings:

| Parameter        | Value                         |
| ---------------- | ----------------------------- |
| Symbol           | SPY                           |
| Timeframe        | 5 minutes                     |
| Feed             | SIP                           |
| Price adjustment | Raw                           |
| Market session   | 9:30 a.m. – 4:00 p.m. Eastern |
| Timestamp format | UTC                           |

Each observation contains a timestamp and the opening, highest, lowest and closing prices for that interval, aswell as trading volume.

I used the price data to study how execution strategies perform when market prices change. The volume data was useful for estimating when trading activity is typically higher or lower throughout the day.

## Data Collection and Validation

Before downloading the full dataset, I tested the API using trading sessions from different years to confirm that I could retrieve the expected data.

| Trading Date      | Five-Minute Bars Returned |
| ----------------- | ------------------------: |
| February 15, 2023 |                        78 |
| July 17, 2024     |                        78 |
| October 15, 2025  |                        78 |

Each request returned the expected 78 bars for a regular trading session. I also checked that the timestamps aligned with the US market's trading hours.

I then collected data from January 2023 through September 2026. The completed dataset contains **939 trading sessions and 72,954 five-minute bars**.

| Dataset    | Period                 | Trading Sessions | Five-Minute Bars |
| ---------- | ---------------------- | ---------------: | ---------------: |
| Training   | 2023–2024              |              502 |           38,976 |
| Validation | 2025                   |              250 |           19,392 |
| Final Test | January–September 2026 |              187 |           14,586 |
| **Total**  |                        |          **939** |       **72,954** |

I checked the completed dataset against the NYSE trading calendar. This included accounting for shortened trading sessions which contain fewer observations than a regular trading day. The final audit reported 100% coverage with no missing expected sessions or bars.

I also implemented checkpointing during data collection. Each completed trading session is saved as a separate CSV file and the downloader checks existing files before requesting additional data. This allowed me to resume interrupted downloads without collecting the same sessions again. The raw data is stored locally under `data/raw/SPY_5min_SIP/` and is excluded from Git.

## Historical Market Calibration

I used the 2023–2024 training data to estimate the market conditions needed for the trading simulator.

The estimated standard deviation of five-minute intraday log returns was approximately **0.000721**. I used this value to set the volatility of the synthetic price paths generated during reinforcement learning training.

I also calculated an average intraday trading-volume profile. This calculation used 497 regular trading sessions. I excluded five shortened sessions because their different trading hours would affect the full-day volume distribution.

The resulting profile contains 78 volume fractions, one for each five-minute interval in a regular session. The first interval accounted for approximately 3.03% of average daily volume and the final interval accounted for approximately 8.91%. These estimates were saved in `configs/spy_training_calibration.json`.

The volume profile was later used to construct VWAP schedules and estimate relative liquidity during the execution window. All calibration parameters were calculated from the training period before examining the final test data.

## Data Separation and Evaluation

I divided the historical data chronologically to keep model development separate from final testing. The 2023–2024 data was used for market calibration and synthetic training scenarios. I used the 2025 sessions to evaluate execution strategies during development and investigate problems with the trained agents.

I reserved the January–September 2026 sessions for the final evaluation. Once the DQN checkpoints and experimental settings were finalized, I evaluated the saved models on all 187 sessions without retraining them.

The historical experiments use observed SPY price movements while the hypothetical trades are executed through the simulator. Transaction costs and market impact are calculated using the project's assumed execution-cost models. The historical data does not directly measure how my simulated orders would have affected actual market prices.

## Reproducing the Data Preparation

The data collection scripts are located under `scripts/data/`. Downloading historical data requires Alpaca API credentials with access to the SIP feed.

After installing the project dependencies, credentials can be added to a local `.env` file using `.env.example` as a reference.

To download the historical dataset:

```bash
python -m scripts.data.download_market_data \
  --start 2023-01-01 \
  --end 2026-09-30
```

The downloader checks for previously completed sessions so the command can also be used to resume an interrupted collection.

To audit the dataset against the expected trading calendar:

```bash
python -m scripts.data.audit_market_data \
  --start 2023-01-01 \
  --end 2026-09-30
```

Once the training data is available, the calibration can be regenerated using:

```bash
python -m scripts.data.calibrate_market
```

This recalculates the training-period market estimates and writes them to `configs/spy_training_calibration.json`. The repository already includes the calibration file used for the reported experiments.