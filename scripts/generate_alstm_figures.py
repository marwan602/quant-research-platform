from pathlib import Path
import sys
sys.path.insert(0, ".")
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from src.evaluate import compute_daily_rank_ic

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

out_dir = Path("reports/figures")
out_dir.mkdir(parents=True, exist_ok=True)

df_alstm = pd.read_parquet("reports/alstm_test_predictions.parquet")
df_alstm["Date"] = pd.to_datetime(df_alstm["Date"])
clean_alstm = df_alstm.dropna(subset=["Date", "Ticker", "target", "pred"]).copy()
unique_dates = sorted(clean_alstm["Date"].unique())

df_lgb = pd.read_parquet("reports/lightgbm_test_predictions.parquet")
df_lgb["Date"] = pd.to_datetime(df_lgb["Date"])
clean_lgb = df_lgb.dropna(subset=["Date", "Ticker", "target", "pred"]).copy()

holding_period = 5
cost_rate = 10.0 / 10000.0
rebalance_dates = unique_dates[::holding_period]

def simulate_portfolio(clean_df):
    lo_net = []
    ls_net = []
    bench = []
    dates_used = []
    prev_lo = {}
    prev_ls = {}

    for d in rebalance_dates:
        day_slice = clean_df[clean_df["Date"] == d]
        n_stocks = len(day_slice)
        if n_stocks < 2:
            continue
        n_top = max(1, int(np.floor(n_stocks * 0.1)))
        n_bottom = max(1, int(np.floor(n_stocks * 0.1)))

        sorted_slice = day_slice.sort_values("pred", ascending=False)
        top_slice = sorted_slice.iloc[:n_top]
        bottom_slice = sorted_slice.iloc[n_stocks - n_bottom:]

        lo_tickers = set(top_slice["Ticker"])
        curr_lo = {t: 1.0 / len(lo_tickers) for t in lo_tickers}
        all_lo = set(curr_lo.keys()) | set(prev_lo.keys())
        lo_turnover = 0.5 * sum(abs(curr_lo.get(t, 0.0) - prev_lo.get(t, 0.0)) for t in all_lo)
        r_long = float(top_slice["target"].mean())
        lo_net.append(r_long - lo_turnover * cost_rate)
        prev_lo = curr_lo

        bot_tickers = set(bottom_slice["Ticker"])
        curr_ls = {t: 0.5 / len(lo_tickers) for t in lo_tickers}
        for t in bot_tickers:
            curr_ls[t] = curr_ls.get(t, 0.0) - (0.5 / len(bot_tickers))
        all_ls = set(curr_ls.keys()) | set(prev_ls.keys())
        ls_turnover = 0.5 * sum(abs(curr_ls.get(t, 0.0) - prev_ls.get(t, 0.0)) for t in all_ls)
        r_short = float(bottom_slice["target"].mean())
        r_ls = 0.5 * r_long - 0.5 * r_short
        ls_net.append(r_ls - ls_turnover * cost_rate)
        prev_ls = curr_ls

        bench.append(float(day_slice["target"].mean()))
        dates_used.append(d)

    dt_idx = pd.to_datetime(dates_used)
    w_lo = (1.0 + pd.Series(lo_net, index=dt_idx)).cumprod() - 1.0
    w_ls = (1.0 + pd.Series(ls_net, index=dt_idx)).cumprod() - 1.0
    w_bench = (1.0 + pd.Series(bench, index=dt_idx)).cumprod() - 1.0
    return dt_idx, lo_net, ls_net, bench, w_lo, w_ls, w_bench

dt_idx, alstm_lo_net, alstm_ls_net, alstm_bench, w_lo_alstm, w_ls_alstm, w_bench = simulate_portfolio(clean_alstm)
_, lgb_lo_net, lgb_ls_net, _, w_lo_lgb, w_ls_lgb, _ = simulate_portfolio(clean_lgb)

# ALSTM Cumulative Returns Plot
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.plot(dt_idx, w_lo_alstm * 100, label=f"ALSTM Long-Only Top Decile Net (+{w_lo_alstm.iloc[-1]*100:.1f}%)", color="#8b5cf6", lw=2.2)
ax.plot(dt_idx, w_bench * 100, label=f"Equal-Weighted Universe Benchmark (+{w_bench.iloc[-1]*100:.1f}%)", color="#64748b", lw=1.8, linestyle="--")
ax.plot(dt_idx, w_ls_alstm * 100, label=f"ALSTM Market-Neutral Long-Short 50/50 Net (+{w_ls_alstm.iloc[-1]*100:.1f}%)", color="#06b6d4", lw=2.0)
ax.set_title("ALSTM Alpha158: Cumulative Out-of-Sample Performance (2024 - 2026)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Cumulative Return (%)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper left", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(out_dir / "alstm_cumulative_returns.png", dpi=300)
plt.close(fig)

# ALSTM Cumulative Rank IC
rank_ic_alstm = compute_daily_rank_ic(clean_alstm)
cum_rank_ic_alstm = rank_ic_alstm.cumsum()
ic_dates = pd.to_datetime(rank_ic_alstm.index)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=300, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
ax1.plot(ic_dates, cum_rank_ic_alstm, color="#8b5cf6", lw=2.0)
ax1.set_title("ALSTM: Daily Cross-Sectional Spearman Rank IC (p = 0.0405, t = 2.052)", fontsize=13, fontweight="bold", pad=12)
ax1.set_ylabel("Cumulative Rank IC", fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)

bar_colors = ["#10b981" if x >= 0 else "#ef4444" for x in rank_ic_alstm.values]
ax2.bar(ic_dates, rank_ic_alstm.values, width=1.8, color=bar_colors, alpha=0.7)
ax2.axhline(0, color="black", lw=0.8, linestyle="-")
ax2.set_ylabel("Daily Rank IC", fontsize=10)
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax2.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(out_dir / "alstm_cumulative_rank_ic.png", dpi=300)
plt.close(fig)

# ALSTM Underwater Drawdown Plot
def calc_dd_series(s):
    w = (1.0 + s).cumprod()
    p = w.cummax()
    return (w - p) / p

dd_lo_alstm = calc_dd_series(pd.Series(alstm_lo_net, index=dt_idx)) * 100
dd_ls_alstm = calc_dd_series(pd.Series(alstm_ls_net, index=dt_idx)) * 100
dd_bench = calc_dd_series(pd.Series(alstm_bench, index=dt_idx)) * 100

fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
ax.plot(dt_idx, dd_lo_alstm, label=f"ALSTM Long-Only Top Decile Net (Max {dd_lo_alstm.min():.1f}%)", color="#8b5cf6", lw=1.6)
ax.plot(dt_idx, dd_bench, label=f"Benchmark (Max {dd_bench.min():.1f}%)", color="#64748b", lw=1.4, linestyle="--")
ax.plot(dt_idx, dd_ls_alstm, label=f"ALSTM Long-Short 50/50 Net (Max {dd_ls_alstm.min():.1f}%)", color="#06b6d4", lw=2.0)
ax.fill_between(dt_idx, dd_ls_alstm, 0, color="#06b6d4", alpha=0.15)
ax.set_title("ALSTM: Underwater Portfolio Drawdown Curves (2024 - 2026)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Drawdown (%)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="lower left", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(out_dir / "alstm_underwater_drawdown.png", dpi=300)
plt.close(fig)

# Head-to-Head Model Comparison Plot
fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
ax.plot(dt_idx, w_lo_lgb * 100, label=f"Model 1 (LightGBM) Long-Only Net (+{w_lo_lgb.iloc[-1]*100:.1f}%)", color="#2563eb", lw=2.2)
ax.plot(dt_idx, w_lo_alstm * 100, label=f"Model 2 (ALSTM) Long-Only Net (+{w_lo_alstm.iloc[-1]*100:.1f}%)", color="#8b5cf6", lw=2.2)
ax.plot(dt_idx, w_bench * 100, label=f"Equal-Weighted Universe Benchmark (+{w_bench.iloc[-1]*100:.1f}%)", color="#64748b", lw=1.8, linestyle="--")
ax.plot(dt_idx, w_ls_lgb * 100, label=f"Model 1 (LightGBM) Long-Short Net (+{w_ls_lgb.iloc[-1]*100:.1f}%)", color="#10b981", lw=1.8)
ax.plot(dt_idx, w_ls_alstm * 100, label=f"Model 2 (ALSTM) Long-Short Net (+{w_ls_alstm.iloc[-1]*100:.1f}%)", color="#06b6d4", lw=1.8)
ax.set_title("Model Comparison: LightGBM vs ALSTM Out-of-Sample Performance (2024 - 2026)", fontsize=13, fontweight="bold", pad=12)
ax.set_ylabel("Cumulative Return (%)", fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:+.0f}%"))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper left", fontsize=9.5)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(out_dir / "model_comparison_cumulative_returns.png", dpi=300)
plt.close(fig)

# Training History Loss Plot
with open("reports/alstm_training_history.json", "r", encoding="utf-8") as f:
    history = json.load(f)

epochs = [h["epoch"] for h in history]
train_losses = [h["train_loss"] for h in history]
val_losses = [h["val_loss"] for h in history]

fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
ax.plot(epochs, train_losses, label="Training Loss (MSE)", color="#3b82f6", lw=2.0, marker="o", markersize=4)
ax.plot(epochs, val_losses, label="Validation Loss (MSE)", color="#f97316", lw=2.0, marker="s", markersize=4)
ax.axvline(2, color="#10b981", linestyle="--", lw=1.5, label="Best Checkpoint (Epoch 2, Val Loss = 0.002237)")
ax.set_title("ALSTM Training & Validation Convergence Curve", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Epoch", fontsize=11)
ax.set_ylabel("Mean Squared Error (MSE)", fontsize=11)
ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0", loc="upper right", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
fig.savefig(out_dir / "alstm_training_loss.png", dpi=300)
plt.close(fig)

print("Generated all ALSTM and comparison figures successfully!")
