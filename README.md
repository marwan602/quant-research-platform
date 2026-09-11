# S&P 500 Quantitative Research Platform

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-blue?style=flat-square)](https://marwan602.github.io/quant-research-platform/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A quantitative research platform for 5-day cross-sectional S&P 500 return prediction using point-in-time index membership, Microsoft Qlib's Alpha158 factor suite adapted to US equities, and reproducible machine learning & deep learning architectures.

🌐 **Live Web Dashboard**: [https://marwan602.github.io/quant-research-platform/](https://marwan602.github.io/quant-research-platform/)

---

## Data & Universe Construction

The dataset covers daily trading activity from January 2015 onwards across historical and active constituents:

- **Point-in-Time Index Composition**: Reconstructed daily from historical index changes. Acquired, merged, and delisted companies are tracked throughout their active index tenure to address survivorship bias.
- **Price History**: Daily OHLCV bars stored locally in `data/raw/` (`s_and_p_500_daily_composition.parquet` and `s_and_p_500_prices.parquet`).

---

## Features & Target

- **Features (`src/features.py`)**: Implementation of Qlib's Alpha158 feature definitions adapted to the S&P 500 OHLCV dataset:
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

| Metric | Model 1: LightGBM | Model 2: ALSTM (Attentive LSTM) | Model 3: Transformer (Attentive) | Equal-Weighted Benchmark |
| :--- | :---: | :---: | :---: | :---: |
| **Mean IC (Pearson)** | 0.0152 | 0.0260 | **0.0269** (+77.0%) | — |
| **IC Information Ratio (ICIR)** | 1.831 | **2.454** | 2.409 | — |
| **IC t-statistic / p-value** | 2.970 (p = 0.0031) | 3.981 (p = 7.6e-5) | **3.908 (p = 1.0e-4)** | — |
| **IC Positive Days Win Rate** | 52.19% | **55.35%** | 53.39% | — |
| **Mean Rank IC (Spearman)** | 0.0118 | 0.0123 | **0.0227** (+84.5% vs ALSTM) | — |
| **Rank ICIR** | 1.161 | 1.265 | **2.058** | — |
| **Rank IC t-stat / p-value** | 1.883 (p = 0.060) | 2.052 (p = 0.0405) | **3.338 (p = 0.00089)** | — |
| **Rank IC Positive Days Win Rate** | 52.94% | 51.73% | **53.54%** | — |
| **Long-Only Annualized Return (Net)** | +25.21% | +24.58% | **+29.05%** | +13.69% |
| **Long-Only Excess Return vs Bench** | +11.26% | +10.90% | **+15.37%** | — |
| **Long-Only Net Sharpe Ratio** | 1.108 | 1.065 | **1.188** | 1.058 |
| **Long-Only Max Drawdown (Net)** | 24.37% | **23.88%** | 25.15% | 15.79% |
| **Long-Only Mean 5-Day Turnover** | 46.56% | 53.05% | **42.31%** | — |
| **Long-Short Annualized Return (Net)** | +5.71% | +2.42% | **+8.26%** | — |
| **Long-Short Net Sharpe Ratio** | 0.629 | 0.274 | **0.811** | — |
| **Long-Short Max Drawdown (Net)** | **11.09%** | 11.41% | 12.41% | — |
| **Long-Short Mean 5-Day Turnover** | **46.56%** | 53.05% | 55.55% | — |
| **Point Prediction RMSE / MAE** | 0.0472 / 0.0325 | 0.0472 / 0.0326 | **0.0469 / 0.0324** | — |

---

## Model Architecture & Details

### Model 1: LightGBM (`src/models/lightgbm_model.py`)
- **Type**: Gradient-boosted decision trees (`LGBMRegressor`, 1000 estimators, learning rate 0.03, 31 leaves).
- **Objective**: Direct contemporaneous mapping from 158 features at date t to 5-day forward return.
- **Top Predictive Features**: Multi-period momentum (`ROC30`, `ROC5`), volume-price correlation (`CORR20`), and trend exhaustion indicators (`IMIN60`, `IMAX20`).
- **Strengths**: Isolates non-linear cross-sectional feature interactions with fast training and deterministic evaluation.

### Model 2: Attentive LSTM / ALSTM (`src/models/alstm_model.py`)
- **Type**: Stacked 2-layer unidirectional LSTM with temporal attention:
  - Input: `(batch_size, 60, 158)`
  - Backbone: 2-layer stacked unidirectional LSTM (`hidden_size=64`, `dropout=0.2`) producing hidden sequence `(batch_size, 60, 64)` and final step state `(batch_size, 64)`.
  - Temporal Attention: Computes attention weights across the 60 days via `softmax(v^T * tanh(W * h_t + b))` to yield a weighted context vector `(batch_size, 64)`.
  - Feature Concatenation: `[context_vector ; last_hidden_state]` -> combined representation `(batch_size, 128)`.
  - Prediction Head: MLP `128 -> 64 -> ReLU -> Dropout(0.2) -> 1` producing the scalar 5-day return prediction.
- **Normalization**: `RobustStandardScaler` fitted strictly on train data with 0.05% and 99.95% percentile winsorization.
- **Training**: Adam optimizer, MSE loss, gradient clipping (1.0), early stopping patience 7 on validation loss. Best model checkpoint restored from Epoch 2 (validation loss `0.002237`).

### Model 3: Time-Series Transformer (`src/models/transformer_model.py`)
- **Type**: Multi-head self-attention Transformer encoder with temporal attention pooling:
  - Input: `(batch_size, 60, 158)`
  - Linear Feature Projection: `nn.Linear(158, 64)` mapping factor dimensions into model space.
  - Positional Encoding: Sinusoidal positional embeddings with input dropout (`dropout=0.2`) across the 60-day sequence.
  - Backbone: 2-layer `nn.TransformerEncoder` (`d_model=64`, `nhead=4`, `dim_feedforward=128`, `dropout=0.2`, `batch_first=True`).
  - Temporal Attention Pooling: Learns dynamic attention weights over all 60 contextualized states via `softmax(v^T * tanh(W * h_t + b))` to produce sequence context vector `c` `(batch_size, 64)`.
  - State Concatenation: `[context_vector ; last_state]` -> combined representation `(batch_size, 128)`, fusing multi-week trajectory context with the latest day t state.
  - Prediction Head: MLP `128 -> 64 -> ReLU -> Dropout(0.2) -> 1` producing the scalar 5-day return prediction.
- **Training**: Adam (`lr=0.0005`, `weight_decay=1e-5`), MSE loss, gradient clipping (1.0), early stopping patience 7. Restored from Epoch 2 best checkpoint (validation loss `0.002233`, the lowest validation loss in the project).
- **Performance**: Yields the highest rank correlation and risk-adjusted excess returns among the evaluated models, delivering **0.0227 Rank IC (p = 0.00089)**, **+29.05% net long-only return (+15.37% excess over benchmark, 1.188 Sharpe)**, **+8.26% net market-neutral long-short return (0.811 Sharpe)**, and a 42.31% 5-day turnover.

---

## Research Figures

Detailed plots are saved in `reports/figures/`:

- `model_comparison_cumulative_returns.png`: Three-model head-to-head out-of-sample cumulative performance comparing LightGBM, ALSTM, Transformer, and the Universe Benchmark.
- `transformer_cumulative_returns.png`: Transformer Long-Only, Long-Short, and Benchmark cumulative net returns.
- `transformer_cumulative_rank_ic.png`: Daily and cumulative Spearman Rank IC trajectory for Transformer.
- `transformer_underwater_drawdown.png`: Underwater drawdown curves over the 2024–2026 test period.
- `transformer_training_loss.png`: Training vs validation loss convergence across epochs.
- `alstm_cumulative_returns.png`, `alstm_cumulative_rank_ic.png`, `alstm_underwater_drawdown.png`, `alstm_training_loss.png`: ALSTM research figures.
- `cumulative_returns.png`, `cumulative_rank_ic.png`, `underwater_drawdown.png`, `factor_importance.png`: LightGBM baseline figures.

---

## Reproduction & Usage

### 1. Run Complete Test Suite
```bash
uv run pytest
```
*(51 unit and regression tests covering dataset splitting, sequence preparation, factor formulas, portfolio math, model architectures, data providers, and the inference pipeline).*

### 2. Train Model 1 (LightGBM)
```bash
uv run python -m src.models.lightgbm_model
```

### 3. Train Model 2 (ALSTM)
```powershell
py -3.11 -m src.models.alstm_model
```

### 4. Train Model 3 (Transformer)
```powershell
py -3.11 -m src.models.transformer_model
```

### 5. Generate Research Figures
```powershell
py -3.11 scripts/generate_transformer_figures.py
```

### 6. Real-Time Inference
Generate 5-day forward return forecasts and portfolio allocations for current index constituents:
```powershell
py -3.11 -m src.inference --as-of-date 2026-09-10 --top-n 10
```
The inference pipeline loads trailing 120-day constituent bars, computes Alpha158 factor sequences, evaluates the trained Transformer checkpoint, and outputs top-decile long targets alongside portfolio rebalance weights.

### 7. Daily Production Runner
Automate the post-close market ingestion (via Polygon.io or local store), factor engineering, CPU Transformer inference (~15 seconds), and state payload serialization:
```powershell
py -3.11 scripts/daily_update.py --provider polygon
```
This updates the price store and outputs live JSON payloads to `reports/live/`:
* `rankings.json`: Ranked forecasts and deciles for all active S&P 500 constituents.
* `portfolio.json`: Top-decile target weights and long/short constituent buckets.
* `system_status.json`: Sync timestamp, active constituent count, and pipeline execution health.

### 8. Production REST API
Serve real-time research outputs, portfolio rebalancing simulations, and cross-model performance benchmarks:
```powershell
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Available endpoints:
* `GET /health`: Health probe.
* `GET /api/v1/system/status`: Pipeline health, active model, and last synchronization date.
* `GET /api/v1/rankings/latest?top_k=10&bottom_k=10`: Top-K and bottom-K ranked stock return predictions.
* `GET /api/v1/portfolio/current`: Current top-decile target portfolio weights.
* `POST /api/v1/portfolio/rebalance`: Calculates execution orders (BUY/SELL, trade dollar values) from current client holdings.
* `GET /api/v1/models/benchmark`: Out-of-sample Sharpe ratio, Rank IC, and drawdown metrics across LightGBM, ALSTM, and Transformer.

### 9. Public Web Dashboard & Cloud Automation
The research platform is deployed as a zero-maintenance static web application on GitHub Pages:
🔗 **[https://marwan602.github.io/quant-research-platform/](https://marwan602.github.io/quant-research-platform/)**

Features:
* **Daily Forecast**: Complete 501-stock cross-sectional return forecasts, top/bottom ranked signals, and searchable universe table.
* **Model Portfolio & Rebalance**: 50-stock target portfolio, dynamic sector distribution, and interactive rebalance execution worksheet.
* **Forward Performance**: Out-of-sample historical evidence and live forward tracking curve with prediction audit log.
* **Empirical Benchmarks**: Head-to-head empirical validation matrix and comparative trajectories across 663 trading sessions.
* **Technical Appendix**: Specification of point-in-time universe, 158 Alpha factors, temporal self-attention, and cloud pipeline.

To run the dashboard locally:
```powershell
uv run python -m http.server 8000 --directory docs
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
