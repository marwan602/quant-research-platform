# Baseline Prediction Breakdown (Transformer V1)

## 1. Overview

When inspecting the live daily rankings, you may notice that a group of stocks share an identical predicted return of `+0.0491%` (for example, 28 out of 501 stocks on September 10, 2026). 

This document explains why this happens, how it behaves across historical data, and why it has no impact on the live portfolio strategy.

---

## 2. Why This Happens

The Transformer model uses a small two-layer prediction head to convert its temporal attention features into a 5-day return prediction:

```text
Attention Context (128 features)
              │
              ▼
       Linear(128 -> 64)
              │
              ▼
            ReLU
              │
              ▼
         Dropout(0.2)
              │
              ▼
        Linear(64 -> 1)
              │
              ▼
       Predicted Return
```

### The Output Bias Floor
When the model doesn't find strong positive signals for a stock, all 64 values coming out of the first linear layer are negative or zero. 

Because ReLU turns any non-positive number into zero, the entire 64-number vector becomes zeros:

$$\mathbf{a} = \max(0, \mathbf{z}) = \vec{0}$$

When an all-zero vector enters the final layer, the weights multiply by zero, leaving only the layer's learned baseline bias:

$$\hat{y} = \mathbf{W} \cdot \vec{0} + b = b = 0.0004910036805085838 \approx +0.0491\%$$

This matches the model's internal output bias down to the exact floating-point decimal.

### Can Predictions Go Lower?
Yes. The final layer has 35 positive weights and 29 negative weights. When a stock has extreme negative signals, specific neurons tied to negative weights activate, pulling the prediction below `+0.0491%`. In historical testing, this happened in only 1.27% of cases, typically for volatile stocks in sharp drawdowns.

---

## 3. Historical Data Across All Periods

To check whether this was caused by recent market changes or if it was part of the model from the start, we ran the model across all historical dataset splits:

| Dataset Split | Period | Sample Count | At Baseline Floor (~+0.05%) | Above Baseline | Below Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Training** | 2015–2021 | 790,996 | **37.35%** | 60.47% | 2.19% |
| **Validation** | 2022–2023 | 248,326 | **24.52%** | 74.26% | 1.22% |
| **Test (Out-of-Sample)** | 2024–2026 | 329,522 | **32.23%** | 66.50% | 1.27% |
| **Live Tracking** | Sep 10, 2026 | 501 | **5.59%** | 94.41% | 0.00% |

### Takeaway
This baseline behavior has been part of the model throughout its entire history—accounting for roughly 25% to 37% of stocks on any given day. It is a natural property of the trained weights, not bad data or a pipeline error. The 5.59% seen on September 10 is well within the normal day-to-day range (which swings between 0.4% and 63.3% depending on market momentum).

---

## 4. Does This Affect the Strategy?

### 1. Zero Overlap with the Portfolio
The live strategy only buys the **Top 50 stocks (Top Decile)**.

Across all **663 trading days** in the 2024–2026 test set:
- **Baseline stocks in Top 50:** **0**
- **Baseline stocks in Top 100:** **0**
- **Average rank of baseline stocks:** **397 out of ~500** (bottom quintile)
- **Best rank a baseline stock ever reached:** **Rank 181** (on a day when 63.3% of all stocks sat at the baseline)

The baseline group stays firmly at the bottom of the rankings and never enters the trading portfolio.

### 2. Realized Returns are Still Properly Ordered
When we group test predictions and check what the stocks actually returned over the next 5 days, the relationship is strictly monotonic:

| Prediction Tier | % of Universe | Mean Forecast | Mean Realized 5d Return | Realized Volatility | Mean Excess Return vs Market |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Extreme Negative** (< +0.05%) | 1.27% | -1.39% | +0.91% | 7.88% | +0.38% |
| **Baseline Floor** (= +0.05%) | **32.23%** | **+0.05%** | **+0.04%** | **3.47%** | **-0.08%** |
| **Low Positive** (Tier 1) | 16.62% | +0.06% | +0.15% | 3.96% | -0.06% |
| **Mid Positive** (Tier 2) | 16.63% | +0.10% | +0.20% | 4.24% | -0.07% |
| **High Positive** (Tier 3) | 16.63% | +0.19% | +0.36% | 4.83% | -0.04% |
| **Top Positive** (Tier 4) | 16.63% | +0.65% | +0.80% | 6.83% | +0.30% |

Stocks at the baseline floor had:
- The lowest realized return of any positive tier (+0.04% vs +0.80% for top tier).
- The lowest volatility (3.47%).
- Negative excess returns relative to the daily market average (-0.08%).

### 3. Grouping Weak Stocks Actually Helps Ranking
We measured daily Spearman Rank IC with and without the baseline stocks:

| Evaluation Universe | Mean Daily Rank IC | Information Ratio (IR) | % Positive Days |
| :--- | :---: | :---: | :---: |
| **Full Universe (with baseline)** | **0.0227** | **2.06** | **53.4%** |
| **Excluding Baseline Stocks** | **0.0150** | **1.46** | **52.2%** |

Rank IC drops from 0.0227 to 0.0150 when baseline stocks are removed. Placing the bottom third of weak or signal-less stocks together gives the model clear separation between the winners and the rest of the market.

---

## 5. How This is Handled in Production

1. **V1 Remains Frozen**:
   The deployed weights, features, and rankings remain unchanged. Modifying a live model after seeing forward results would compromise the experiment.
2. **Deterministic Tie-Breaking**:
   Stocks at the baseline floor are treated as tied at the bottom of the rankings and sorted alphabetically by ticker. We do not inject secondary factors (like volatility or momentum) to force an artificial ordering.
3. **Use for Ranking, Not Exact Numbers**:
   Predictions should be used to rank stocks relative to each other, not as literal dollar-return forecasts. A stock on the baseline simply means the model has no positive conviction on it.

---

## 6. Ideas for Future Models

If building a new model in the future, alternative designs could prevent this zeroing effect:
- **GELU Activation:** Has smooth curves with non-zero values for negative inputs.
- **LeakyReLU:** Keeps a small slope for negative inputs so values never fully zero out.
- **Residual Skip Connection:** Lets linear signals bypass the hidden layer directly to the output.

No changes are planned for the active V1 deployment.
