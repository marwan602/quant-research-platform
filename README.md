# S&P 500 Quantitative Research Platform

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-GitHub_Pages-blue?style=flat-square)](https://marwan602.github.io/quant-research-platform/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

A production quantitative research platform predicting 5-day cross-sectional S&P 500 equity returns using point-in-time constituent membership, Microsoft Qlib's Alpha158 factor suite, and deep learning architectures (Transformer, Attentive LSTM, LightGBM).

🌐 **Live Research Dashboard**: [https://marwan602.github.io/quant-research-platform/](https://marwan602.github.io/quant-research-platform/)

---

## 1. Data, Universe & Replication

### Dataset Attribution & Reference Pipeline
This project's historical universe and price data pipeline is adapted from the open-source pipeline by **Konrad Zatorski**:
👉 **Reference Repository**: [https://github.com/K0D1Z/sp500-quantitative-dataset](https://github.com/K0D1Z/sp500-quantitative-dataset)

Specifically, this project adapts **Steps 1 through 4** of that pipeline:
- **Step 1**: Point-in-time daily S&P 500 composition scraping (survivorship-bias-free tracking of index additions and deletions).
- **Step 2**: Corporate events ledger construction (mergers, spin-offs, ticker renames e.g. FB → META).
- **Step 3**: Historical OHLCV bar downloading via Yahoo Finance.
- **Step 4**: Gap backfilling via Tiingo.

*(Note: Steps 5 & 6 of the reference repo—SEC EDGAR fundamental extraction and fundamental ratio merges—are intentionally omitted here, as our models predict cross-sectional returns using the technical Alpha158 factor suite).*

### Data Privacy & Commercial Redistribution Policy
Under the Terms of Service of commercial market data providers (**Polygon.io** and **Yahoo Finance / yfinance**), raw proprietary OHLCV market bars cannot be redistributed publicly in this repository. Raw price stores are maintained in a private repository (`quant-data-private`) for production runs.

### How to Replicate the Dataset
To generate the raw market files and compute the factor matrix from scratch:

1. **Clone the reference dataset repository**:
   ```bash
   git clone https://github.com/K0D1Z/sp500-quantitative-dataset.git
   cd sp500-quantitative-dataset
   uv sync
   ```
2. **Execute Steps 1–4 of the ETL pipeline**:
   ```bash
   uv run python src/sp500_quantitative_dataset/generate_daily_composition.py
   uv run python src/sp500_quantitative_dataset/generate_corporate_events.py
   uv run python src/sp500_quantitative_dataset/download_historical_prices.py
   # Optional / recommended for delisted stock backfill (requires free TIINGO_API_KEY):
   uv run python src/sp500_quantitative_dataset/backfill_missing_prices.py
   ```
3. **Copy the generated parquet files into this project's `data/raw/` directory**:
   *(The reference pipeline outputs files to `data/datasets/`)*
   ```bash
   mkdir -p /path/to/quant-research-platform/data/raw
   cp data/datasets/daily_composition/s_and_p_500_daily_composition.parquet /path/to/quant-research-platform/data/raw/
   cp data/datasets/historical_prices/s_and_p_500_prices.parquet /path/to/quant-research-platform/data/raw/
   ```
4. **Compute the Alpha158 factor matrix**:
   Navigate to this project (`quant-research-platform`) and run:
   ```bash
   uv run python -m src.features
   ```
   This generates the processed feature matrix at `data/processed/sp500_alpha158.parquet` (1,409,169 rows across 158 factors).

---

## 2. Research Methodology

- **Alpha158 Factor Suite (`src/features.py`)**: 158 normalized factors adapted from Microsoft Qlib:
  - 9 K-bar price-action features (open, high, low, close relationships).
  - 4 normalized price ratios and moving trends.
  - 145 rolling technical and statistical metrics across 5, 10, 20, 30, and 60-day windows (momentum `ROC`, volatility `STD`, linear trend `BETA`/`RSQR`/`RESI`, volume dynamics `VMA`/`VSTD`/`WVMA`, volume-price correlation `CORR`/`CORD`).
  - Warm-up period (60 days) cleanly dropped (`min_periods=w`), yielding 1,409,169 observation rows.
- **Prediction Target**: 5-day forward arithmetic return:
  $$R_{t \to t+5} = \frac{\text{Close}_{t+5} - \text{Close}_t}{\text{Close}_t}$$
- **Chronological Splits (Zero Forward Leakage)**:
  - **Train**: 2015-01-01 to 2021-12-31 (826,931 tabular rows; 790,996 60-day sequences)
  - **Validation**: 2022-01-01 to 2023-12-31 (249,959 tabular rows; 248,326 60-day sequences)
  - **Test (Out-of-Sample)**: 2024-01-01 to 2026-08-31 (332,279 tabular rows; 329,522 sequences across 663 trading days)
- **Portfolio Simulation & Friction**:
  - Rebalanced every 5 trading days matching forecast maturity.
  - Long top decile (top 10%), short bottom decile (bottom 10%), or equal-weighted universe benchmark.
  - 10 bps (0.10%) one-way transaction fee applied against portfolio weight turnover.

---

## 3. Empirical Results (Out-of-Sample: 2024–2026)

Evaluated across 663 trading sessions on unseen test data:

| Metric | Model 1: LightGBM | Model 2: ALSTM | Model 3: Transformer | Benchmark (S&P 500 EW) |
| :--- | :---: | :---: | :---: | :---: |
| **Mean IC (Pearson)** | 0.0152 | 0.0260 | **0.0269** (+77.0%) | — |
| **IC Information Ratio (ICIR)** | 1.831 | **2.454** | 2.409 | — |
| **IC t-stat / p-value** | 2.970 (p=0.0031) | 3.981 (p=7.6e-5) | **3.908 (p=1.0e-4)** | — |
| **IC Win Rate (% Positive Days)** | 52.19% | **55.35%** | 53.39% | — |
| **Mean Rank IC (Spearman)** | 0.0118 | 0.0123 | **0.0227** (+84.5%) | — |
| **Rank ICIR** | 1.161 | 1.265 | **2.058** | — |
| **Rank IC t-stat / p-value** | 1.883 (p=0.060) | 2.052 (p=0.0405) | **3.338 (p=0.00089)** | — |
| **Long-Only Annualized Return (Net)** | +25.21% | +24.58% | **+29.05%** | +13.69% |
| **Long-Only Excess Return vs Bench** | +11.26% | +10.90% | **+15.37%** | — |
| **Long-Only Net Sharpe Ratio** | 1.108 | 1.065 | **1.188** | 1.058 |
| **Long-Only Max Drawdown (Net)** | 24.37% | **23.88%** | 25.15% | 15.79% |
| **Long-Only Mean 5-Day Turnover** | 46.56% | 53.05% | **42.31%** | — |
| **Market-Neutral Long-Short Return (Net)**| +5.71% | +2.42% | **+8.26%** | — |
| **Market-Neutral Net Sharpe Ratio** | 0.629 | 0.274 | **0.811** | — |
| **Point Prediction RMSE / MAE** | 0.0472 / 0.0325 | 0.0472 / 0.0326 | **0.0469 / 0.0324** | — |

*Saved figures in `reports/figures/`: `model_comparison_cumulative_returns.png`, `transformer_cumulative_returns.png`, `transformer_underwater_drawdown.png`.*

---

## 4. Automated Daily Forward Pipeline

The live forward tracker operates via a decoupled two-repository architecture:

- **Private Storage (`quant-data-private`)**: Stores proprietary raw OHLCV market bars and point-in-time constituent tables.
- **Public Platform (`quant-research-platform`)**: Stores model weights, inference scripts, docs, and public research feeds.
- **Scheduled GitHub Action (`.github/workflows/daily_runner.yml`)**:
  - Runs weekdays at 22:00 UTC (after US equity close).
  - Checks out `quant-data-private` using `DATA_REPO_TOKEN`.
  - Ingests latest market close bars via Polygon.io API (`POLYGON_API_KEY`).
  - Computes Alpha158 sequence tensors `[N, 60, 158]` and runs Transformer inference (~15s on CPU).
  - Resolves mature 5-day return cycles applying 10 bps turnover friction.
  - Commits updated raw prices to `quant-data-private` and public JSON feeds (`docs/data/`) to this repo.
  - Automatically updates the [GitHub Pages live dashboard](https://marwan602.github.io/quant-research-platform/).

---

## 5. Quickstart & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/marwan602/quant-research-platform.git
cd quant-research-platform

# Install dependencies (Python >= 3.11)
uv sync
```

### 2. Run Test Suite
```bash
uv run pytest
```

### 3. Feature Generation & Model Training
```bash
# Compute Alpha158 factor matrix (requires data/raw/ files)
uv run python -m src.features

# Train Model 1 (LightGBM baseline)
uv run python -m src.models.lightgbm_model

# Train Model 2 (Attentive LSTM / ALSTM)
uv run python -m src.models.alstm_model

# Train Model 3 (Time-Series Transformer)
uv run python -m src.models.transformer_model
```

### 4. Inference & Daily Ingestion
```bash
# Run one-off inference on active universe
uv run python -m src.inference --as-of-date 2026-09-10 --top-n 10

# Execute daily runner (ingest Polygon bars, score universe, export live JSONs)
uv run python scripts/daily_update.py --provider polygon
```

### 5. Production REST API
```bash
uv run uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
Key endpoints:
- `GET /health` & `GET /api/v1/system/status`: System status and sync dates.
- `GET /api/v1/rankings/latest?top_k=10`: Top/bottom predicted returns.
- `GET /api/v1/portfolio/current`: Current top-decile target allocations.
- `POST /api/v1/portfolio/rebalance`: Execution order sheet (BUY/SELL tickets).
- `GET /api/v1/models/benchmark`: Out-of-sample benchmark metrics across all models.

### 6. Local Dashboard Preview
```bash
uv run python -m http.server 8000 --directory docs
```
Navigate to `http://localhost:8000` to inspect the dashboard locally.

---

## 6. Acknowledgments & Licensing

- **[sp500-quantitative-dataset](https://github.com/K0D1Z/sp500-quantitative-dataset)** by **Konrad Zatorski** (MIT License): Point-in-time S&P 500 daily index composition reconstruction, delisting adjustment methodologies, and price backfilling pipeline.
- **[Microsoft Qlib](https://github.com/microsoft/qlib)** by **Microsoft Research** (MIT License): Formulations for the Alpha158 factor suite and attentive time-series deep architectures.

Licensed under the [MIT License](LICENSE).
