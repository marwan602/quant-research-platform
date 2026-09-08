# S&P 500 Quantitative Research Platform

Research platform for predicting 5-day forward returns on the S&P 500 using point-in-time constituent membership and Microsoft Qlib's Alpha158 factor suite.

## Data & Survivorship Bias Prevention

The underlying price and index membership data spans January 2015 through August 2026 across 745 unique historical constituents:

- Index composition is reconstructed daily from historical change records so acquired, merged, and delisted stocks are kept during their active index years (avoiding survivorship bias).
- Daily OHLCV data was acquired from Yahoo Finance and backfilled for delisted/acquired tickers using the Tiingo API (98.5% historical ticker coverage).
- Raw data lives in `data/raw/` (`s_and_p_500_daily_composition.parquet` and `s_and_p_500_prices.parquet`).

## Features & Target

- **Features (`src/features.py`)**: Reimplementation of Microsoft Qlib's Alpha158 factor suite (9 K-bar price-action features, 4 normalized price ratios, and 145 rolling metrics across 5, 10, 20, 30, and 60-day windows). Rolling windows require complete histories (`min_periods=w`); the initial 60-day warm-up period is cleanly dropped rather than zero-filled.
- **Target**: 5-day forward return:
  $$R_{t \rightarrow t+5} = \frac{\text{Close}_{t+5}}{\text{Close}_t} - 1$$

## Dataset Splits

Chronological boundaries enforced in `src/dataset.py`:
- **Train**: 2015-01-01 to 2021-12-31 (826,931 observations)
- **Validation**: 2022-01-01 to 2023-12-31 (249,959 observations)
- **Test**: 2024-01-01 to 2026-08-31 (332,279 out-of-sample observations)

## Evaluation Engine (`src/evaluate.py`)

Models are evaluated across both signal quality and economic performance:
- **Daily Cross-Sectional Information Coefficient (IC & Rank IC)**: Spearman rank correlation between predictions and realized forward returns computed per trading day.
- **5-Day Holding Backtest**: Non-overlapping 5-day rebalance cycle matching the forecast horizon, accounting for one-way transaction costs (default 10 bps) derived from signed portfolio weight turnover:
  - **Market-Neutral Long-Short**: +50% long top decile, -50% short bottom decile ($1.00 gross exposure).
  - **Long-Only Top Decile**: Top 10% highest predicted return stocks.
  - **Benchmark**: Equal-weighted universe 5-day return on the same rebalance schedule.

## Baseline Results: LightGBM (Test Set: 2024–2026)

Trained with early stopping on validation RMSE and evaluated out-of-sample:

| Metric | Long-Short (50/50 Net) | Long-Only (Net) | Equal-Weighted Benchmark |
| :--- | :--- | :--- | :--- |
| **Annualized Return** | +5.71% | +25.21% | +13.95% |
| **Annualized Volatility** | 9.05% | 22.75% | 12.98% |
| **Sharpe Ratio** | 0.63 | 1.11 | 1.07 |
| **Max Drawdown** | -11.09% | -24.33% | -15.77% |
| **Mean Turnover** | 58.07% | 46.20% | — |
| **Excess Return vs Benchmark** | — | +11.26% | — |

- **Mean Rank IC**: 0.0118 (annualized Rank ICIR: 1.16, naive t-stat: 1.88, p = 0.060)
- **Daily Win Rate**: 52.94% positive Rank IC days over 663 test trading days
- **Point Metrics**: RMSE = 0.0472, MAE = 0.0325, R² = 0.071%
- **Top Predictive Factors**: Multi-period price momentum (`ROC30`, `ROC5`), volume-price correlation (`CORR20`), and trend exhaustion (`IMIN60`, `IMAX20`).

Figures and detailed logs are saved in `reports/` and `reports/figures/`.

## Usage

Run tests:
```bash
uv run pytest
```

Train LightGBM baseline:
```bash
uv run python -m src.models.lightgbm_model
```

Generate evaluation figures:
```powershell
$env:PYTHONPATH="."; uv run python scripts/generate_figures.py
```
