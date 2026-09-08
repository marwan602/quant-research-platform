# S&P 500 Quantitative Research Platform

An institutional-grade research platform for predicting 5-day forward returns across S&P 500 constituents using point-in-time index membership, Microsoft Qlib's Alpha158 factor suite, and reproducible machine learning & deep learning models.

---

## Data & Survivorship Bias Prevention

The underlying price and membership dataset spans January 2015 through August 2026 across 745 unique historical constituents:

- **Point-in-Time Index Composition**: Reconstructed daily from corporate actions and historical index changes. Acquired, merged, and delisted companies are preserved throughout their active index tenure, preventing survivorship bias.
- **Price History & Backfilling**: OHLCV data was acquired from Yahoo Finance and backfilled for delisted/acquired tickers via the Tiingo API (achieving 98.5% historical constituent price coverage).
- **Storage**: Raw data resides in `data/raw/` (`s_and_p_500_daily_composition.parquet` and `s_and_p_500_prices.parquet`).

---

## Features & Target

- **Features (`src/features.py`)**: Exact reimplementation of Microsoft Qlib's Alpha158 factor suite:
  - 9 K-bar price-action features (open, high, low, close relationships).
  - 4 normalized price ratios and moving trends.
  - 145 rolling technical and statistical metrics across 5, 10, 20, 30, and 60-day windows (momentum `ROC`, volatility `STD`, linear trend `BETA`/`RSQR`/`RESI`, volume dynamics `VMA`/`VSTD`/`WVMA`, and volume-price correlation `CORR`/`CORD`).
  - Rolling windows require full observation histories (`min_periods=w`); the initial 60-day warm-up period is cleanly dropped rather than zero-filled, leaving 1,409,169 clean rows in `data/processed/sp500_alpha158.parquet`.
- **Target**: 5-day forward arithmetic return:
  ```
  R(t -> t+5) = (Close[t+5] - Close[t]) / Close[t]
  ```

---

## Chronological Dataset Splits

Strict chronological boundaries are enforced in `src/dataset.py` with zero forward leakage:

- **Train**: 2015-01-01 to 2021-12-31 (826,931 tabular rows; 790,996 60-day sequences)
- **Validation**: 2022-01-01 to 2023-12-31 (249,959 tabular rows; 248,326 60-day sequences)
- **Test (Out-of-Sample)**: 2024-01-01 to 2026-08-31 (332,279 tabular rows; 329,522 60-day sequences across 663 trading days)

In the sequence pipeline, validation and test samples legitimately draw their preceding 60-day historical window from late 2021/2023, while sample targets remain strictly restricted to the evaluated period.

---

## Evaluation Engine (`src/evaluate.py`)

Models are evaluated across statistical signal quality and economic portfolio simulation:

- **Cross-Sectional Information Coefficient (Daily)**:
  - **Pearson IC**: Linear predictive correlation computed across universe constituents each trading day.
  - **Spearman Rank IC**: Monotonic ranking correlation computed daily.
  - **Information Ratio (ICIR)**: Annualized signal stability (`mean(IC) / std(IC) * sqrt(252)`).
  - **Hypothesis Testing**: Two-sided t-statistic and p-value for null hypothesis IC = 0.
- **5-Day Holding Period Portfolio Simulation**:
  - Rebalanced every 5 trading days matching the forecast horizon.
  - **Market-Neutral Long-Short**: +50% top decile, -50% bottom decile ($1.00 gross exposure).
  - **Long-Only Top Decile**: Top 10% highest predicted return stocks.
  - **Universe Benchmark**: Equal-weighted universe 5-day return on identical rebalancing dates.
  - **Transaction Costs**: Realistic 10 bps (0.10%) one-way transaction fee applied against portfolio weight turnover.

---

## Model Benchmark & Empirical Results

Evaluated out-of-sample on identical test data (January 2024 through August 2026, 663 trading days):

| Metric | Model 1: LightGBM | Model 2: ALSTM (Attentive LSTM) | Equal-Weighted Benchmark |
| :--- | :---: | :---: | :---: |
| **Mean IC (Pearson)** | 0.0152 | **0.0260** (+71.2%) | — |
| **IC Information Ratio (ICIR)** | 1.831 | **2.454** | — |
| **IC t-statistic / p-value** | 2.970 (p = 0.0031) | **3.981 (p = 7.6e-5)** | — |
| **IC Positive Days Win Rate** | 52.19% | **55.35%** | — |
| **Mean Rank IC (Spearman)** | 0.0118 | **0.0123** | — |
| **Rank ICIR** | 1.161 | **1.265** | — |
| **Rank IC t-stat / p-value** | 1.883 (p = 0.060) | **2.052 (p = 0.0405)** | — |
| **Rank IC Positive Days Win Rate** | 52.94% | **51.73%** | — |
| **Long-Only Annualized Return (Net)** | **+25.21%** | **+24.58%** | +13.69% |
| **Long-Only Excess Return vs Bench** | **+11.26%** | **+10.90%** | — |
| **Long-Only Net Sharpe Ratio** | **1.108** | **1.065** | 1.058 |
| **Long-Only Max Drawdown (Net)** | 24.37% | **23.88%** | 15.79% |
| **Long-Only Mean 5-Day Turnover** | 46.56% | 53.05% | — |
| **Long-Short Annualized Return (Net)** | **+5.71%** | **+2.42%** | — |
| **Long-Short Net Sharpe Ratio** | **0.629** | **0.274** | — |
| **Long-Short Max Drawdown (Net)** | 11.09% | **11.41%** | — |
| **Point Prediction RMSE / MAE** | 0.0472 / 0.0325 | 0.0472 / 0.0326 | — |

---

## Model Architecture & Details

### Model 1: LightGBM (`src/models/lightgbm_model.py`)
- **Type**: Gradient-boosted decision trees (`LGBMRegressor`, 1000 estimators, learning rate 0.03, 31 leaves).
- **Objective**: Direct contemporaneous mapping from 158 features at date t to 5-day forward return.
- **Top Predictive Features**: Multi-period momentum (`ROC30`, `ROC5`), volume-price correlation (`CORR20`), and trend exhaustion indicators (`IMIN60`, `IMAX20`).
- **Strengths**: Excels at isolating tail quintiles for market-neutral long-short spread generation (+5.71% net return, 0.63 Sharpe).

### Model 2: Attentive LSTM / ALSTM (`src/models/alstm_model.py`)
- **Type**: Stacked 2-layer unidirectional LSTM with temporal attention:
  - Input: `(batch_size, 60, 158)`
  - Backbone: 2-layer stacked unidirectional LSTM (`hidden_size=64`, `dropout=0.2`) producing hidden sequence `(batch_size, 60, 64)` and final step state `(batch_size, 64)`.
  - Temporal Attention: Computes attention weights across the 60 days via `softmax(v^T * tanh(W * h_t + b))` to yield a weighted context vector `(batch_size, 64)`.
  - Feature Concatenation: `[context_vector ; last_hidden_state]` -> combined representation `(batch_size, 128)`.
  - Prediction Head: MLP `128 -> 64 -> ReLU -> Dropout(0.2) -> 1` producing the scalar 5-day return prediction.
- **Normalization**: `RobustStandardScaler` fitted strictly on train data with 0.05% and 99.95% percentile winsorization to safely handle rolling volume spikes and zero-variance divisions.
- **Training**: Adam optimizer, MSE loss, gradient clipping (1.0), early stopping patience 7 on validation loss. Best model checkpoint restored from Epoch 2 (validation loss `0.002237`).
- **Strengths**: Temporal modeling of continuous 60-day feature trajectories drives a **+71% jump in Pearson IC (0.0260, p = 7.6e-5)** and achieves **statistically significant Rank IC (0.0123, p = 0.0405)**, outperforming the universe benchmark by **+10.90% net annualized return**.

---

## Research Figures

Detailed plots are saved in `reports/figures/`:

- `model_comparison_cumulative_returns.png`: Head-to-head out-of-sample cumulative performance comparing LightGBM, ALSTM, and the Universe Benchmark.
- `alstm_cumulative_returns.png`: ALSTM Long-Only, Long-Short, and Benchmark cumulative net returns.
- `alstm_cumulative_rank_ic.png`: Daily and cumulative Spearman Rank IC trajectory for ALSTM.
- `alstm_underwater_drawdown.png`: Underwater drawdown curves over the 2024–2026 test period.
- `alstm_training_loss.png`: Training vs validation loss convergence across epochs.
- `cumulative_returns.png`, `cumulative_rank_ic.png`, `underwater_drawdown.png`, `factor_importance.png`: LightGBM baseline figures.

---

## Reproduction & Usage

### 1. Run Complete Test Suite
```bash
uv run pytest
```
*(35 unit tests covering dataset splitting, sequence preparation, factor formulas, portfolio math, LightGBM, and ALSTM architecture).*

### 2. Train Model 1 (LightGBM)
```bash
uv run python -m src.models.lightgbm_model
```

### 3. Train Model 2 (ALSTM)
```powershell
py -3.11 -m src.models.alstm_model
```

### 4. Generate Research Figures
```powershell
py -3.11 scripts/generate_alstm_figures.py
```

---

## Acknowledgments & Licensing

This project builds upon and credits the work of the following open-source projects:

1. **[sp500-quantitative-dataset](https://github.com/K0D1Z/sp500-quantitative-dataset)**:
   - Created by **Konrad Zatorski** under the **MIT License**.
   - Provided the point-in-time S&P 500 daily index composition reconstruction, delisting adjustment methodologies, and OHLCVA backfilling infrastructure.
2. **[Microsoft Qlib](https://github.com/microsoft/qlib)**:
   - Developed by **Microsoft Research Asia** under the **MIT License**.
   - Provided the reference formulations for the Alpha158 factor suite and classical attentive recurrent architectures in quantitative finance.

This codebase is licensed under the [MIT License](LICENSE).
