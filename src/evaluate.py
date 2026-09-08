from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_daily_ic(df: pd.DataFrame) -> pd.Series:
    clean_df = df.dropna(subset=["Date", "target", "pred"]).copy()
    ics = {}
    for date, group in clean_df.groupby("Date", sort=True):
        if len(group) < 2:
            continue
        if group["target"].std() == 0 or group["pred"].std() == 0:
            ics[date] = np.nan
        else:
            ics[date] = group["target"].corr(group["pred"], method="pearson")
    return pd.Series(ics, name="ic")


def compute_daily_rank_ic(df: pd.DataFrame) -> pd.Series:
    clean_df = df.dropna(subset=["Date", "target", "pred"]).copy()
    rank_ics = {}
    for date, group in clean_df.groupby("Date", sort=True):
        if len(group) < 2:
            continue
        if group["target"].std() == 0 or group["pred"].std() == 0:
            rank_ics[date] = np.nan
        else:
            rank_ics[date] = group["target"].corr(group["pred"], method="spearman")
    return pd.Series(rank_ics, name="rank_ic")


def compute_ic_metrics(ic_series: pd.Series, prefix: str = "ic") -> dict[str, float]:
    valid_ic = ic_series.dropna()
    n = len(valid_ic)
    if n == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}ir": 0.0,
            f"{prefix}_naive_tstat": 0.0,
            f"{prefix}_naive_pvalue": 1.0,
            f"{prefix}_pct_positive": 0.0,
        }

    mean = float(valid_ic.mean())
    std = float(valid_ic.std(ddof=1)) if n > 1 else 0.0
    icir = (mean / std * np.sqrt(252.0)) if std > 1e-12 else 0.0

    if std > 1e-12 and n > 1:
        se = std / np.sqrt(n)
        tstat = mean / se
        pvalue = float(stats.t.sf(np.abs(tstat), df=n - 1) * 2.0)
    else:
        tstat = 0.0
        pvalue = 1.0

    pct_positive = float((valid_ic > 0.0).mean())

    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}ir": icir,
        f"{prefix}_naive_tstat": tstat,
        f"{prefix}_naive_pvalue": pvalue,
        f"{prefix}_pct_positive": pct_positive,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) == 0:
        return {"rmse": 0.0, "mae": 0.0, "r2": 0.0}

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _calc_max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = (peak - wealth) / peak
    return float(drawdown.max())


def compute_portfolio_backtest(
    df: pd.DataFrame,
    top_quantile: float = 0.1,
    bottom_quantile: float = 0.1,
    holding_period: int = 5,
    cost_bps: float = 10.0,
) -> dict[str, float]:
    clean_df = df.dropna(subset=["Date", "Ticker", "target", "pred"]).copy()
    unique_dates = sorted(clean_df["Date"].unique())

    rebalance_dates = unique_dates[::holding_period]
    cost_rate = cost_bps / 10000.0
    periods_per_year = 252.0 / holding_period

    lo_returns_gross = []
    lo_returns_net = []
    lo_turnovers = []

    ls_returns_gross = []
    ls_returns_net = []
    ls_turnovers = []

    bench_returns = []

    prev_lo_weights = {}
    prev_ls_weights = {}

    for date in rebalance_dates:
        day_slice = clean_df[clean_df["Date"] == date]
        n_stocks = len(day_slice)
        if n_stocks < 2:
            continue

        n_top = max(1, int(np.floor(n_stocks * top_quantile)))
        n_bottom = max(1, int(np.floor(n_stocks * bottom_quantile)))

        sorted_slice = day_slice.sort_values("pred", ascending=False)
        top_slice = sorted_slice.iloc[:n_top]
        bottom_slice = sorted_slice.iloc[-n_bottom:]

        lo_tickers = set(top_slice["Ticker"])
        curr_lo_weights = {t: 1.0 / len(lo_tickers) for t in lo_tickers}

        all_lo_tickers = set(curr_lo_weights.keys()) | set(prev_lo_weights.keys())
        lo_turnover = 0.5 * sum(
            abs(curr_lo_weights.get(t, 0.0) - prev_lo_weights.get(t, 0.0))
            for t in all_lo_tickers
        )
        if not prev_lo_weights:
            lo_turnover = 1.0

        r_long = float(top_slice["target"].mean())
        lo_cost = lo_turnover * cost_rate
        lo_returns_gross.append(r_long)
        lo_returns_net.append(r_long - lo_cost)
        lo_turnovers.append(lo_turnover)
        prev_lo_weights = curr_lo_weights

        bot_tickers = set(bottom_slice["Ticker"])
        curr_ls_weights = {}
        for t in lo_tickers:
            curr_ls_weights[t] = 0.5 / len(lo_tickers)
        for t in bot_tickers:
            curr_ls_weights[t] = curr_ls_weights.get(t, 0.0) - (0.5 / len(bot_tickers))

        all_ls_tickers = set(curr_ls_weights.keys()) | set(prev_ls_weights.keys())
        ls_turnover = 0.5 * sum(
            abs(curr_ls_weights.get(t, 0.0) - prev_ls_weights.get(t, 0.0))
            for t in all_ls_tickers
        )
        if not prev_ls_weights:
            ls_turnover = 1.0

        r_short = float(bottom_slice["target"].mean())
        r_ls_gross = 0.5 * r_long - 0.5 * r_short
        ls_cost = ls_turnover * cost_rate
        ls_returns_gross.append(r_ls_gross)
        ls_returns_net.append(r_ls_gross - ls_cost)
        ls_turnovers.append(ls_turnover)
        prev_ls_weights = curr_ls_weights

        bench_returns.append(float(day_slice["target"].mean()))

    s_lo_gross = pd.Series(lo_returns_gross)
    s_lo_net = pd.Series(lo_returns_net)
    s_ls_gross = pd.Series(ls_returns_gross)
    s_ls_net = pd.Series(ls_returns_net)
    s_bench = pd.Series(bench_returns)

    def _stats(s: pd.Series) -> tuple[float, float, float, float]:
        if len(s) == 0:
            return 0.0, 0.0, 0.0, 0.0
        mean = float(s.mean())
        std = float(s.std(ddof=1)) if len(s) > 1 else 0.0
        ann_ret = mean * periods_per_year
        ann_vol = std * np.sqrt(periods_per_year)
        sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0
        mdd = _calc_max_drawdown(s)
        return ann_ret, ann_vol, sharpe, mdd

    lo_g_ret, lo_g_vol, lo_g_sharpe, lo_g_mdd = _stats(s_lo_gross)
    lo_n_ret, lo_n_vol, lo_n_sharpe, lo_n_mdd = _stats(s_lo_net)
    ls_g_ret, ls_g_vol, ls_g_sharpe, ls_g_mdd = _stats(s_ls_gross)
    ls_n_ret, ls_n_vol, ls_n_sharpe, ls_n_mdd = _stats(s_ls_net)
    b_ret, b_vol, b_sharpe, b_mdd = _stats(s_bench)

    return {
        "long_only_annualized_return_gross": lo_g_ret,
        "long_only_annualized_return_net": lo_n_ret,
        "long_only_annualized_vol": lo_g_vol,
        "long_only_sharpe_gross": lo_g_sharpe,
        "long_only_sharpe_net": lo_n_sharpe,
        "long_only_max_drawdown_gross": lo_g_mdd,
        "long_only_max_drawdown_net": lo_n_mdd,
        "long_only_mean_turnover": float(np.mean(lo_turnovers)) if lo_turnovers else 0.0,
        "long_only_excess_return_gross": lo_g_ret - b_ret,
        "long_only_excess_return_net": lo_n_ret - b_ret,
        "long_short_annualized_return_gross": ls_g_ret,
        "long_short_annualized_return_net": ls_n_ret,
        "long_short_annualized_vol": ls_g_vol,
        "long_short_sharpe_gross": ls_g_sharpe,
        "long_short_sharpe_net": ls_n_sharpe,
        "long_short_max_drawdown_gross": ls_g_mdd,
        "long_short_max_drawdown_net": ls_n_mdd,
        "long_short_mean_turnover": float(np.mean(ls_turnovers)) if ls_turnovers else 0.0,
        "benchmark_annualized_return": b_ret,
        "benchmark_annualized_vol": b_vol,
        "benchmark_sharpe": b_sharpe,
        "benchmark_max_drawdown": b_mdd,
    }


def evaluate_predictions(
    df: pd.DataFrame,
    holding_period: int = 5,
    top_quantile: float = 0.1,
    bottom_quantile: float = 0.1,
    cost_bps: float = 10.0,
) -> dict[str, float]:
    daily_ic = compute_daily_ic(df)
    daily_rank_ic = compute_daily_rank_ic(df)

    ic_stats = compute_ic_metrics(daily_ic, prefix="ic")
    rank_ic_stats = compute_ic_metrics(daily_rank_ic, prefix="rank_ic")

    reg_stats = compute_regression_metrics(
        y_true=df["target"].values,
        y_pred=df["pred"].values,
    )

    backtest_stats = compute_portfolio_backtest(
        df=df,
        top_quantile=top_quantile,
        bottom_quantile=bottom_quantile,
        holding_period=holding_period,
        cost_bps=cost_bps,
    )

    results = {}
    results.update(ic_stats)
    results.update(rank_ic_stats)
    results.update(reg_stats)
    results.update(backtest_stats)
    return results
