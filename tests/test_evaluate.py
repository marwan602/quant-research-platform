import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    compute_daily_ic,
    compute_daily_rank_ic,
    compute_ic_metrics,
    compute_regression_metrics,
    compute_portfolio_backtest,
    evaluate_predictions,
    _calc_max_drawdown,
)


def make_synthetic_eval_df(n_days=20, n_tickers=20, perfect=False, inverted=False):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    tickers = [f"TICKER_{i}" for i in range(n_tickers)]
    rows = []
    np.random.seed(42)

    for d in dates:
        for t in tickers:
            target_val = float(np.random.normal(0.002, 0.02))
            if perfect:
                pred_val = target_val
            elif inverted:
                pred_val = -target_val
            else:
                pred_val = float(np.random.normal(0.001, 0.02))

            rows.append({
                "Date": d,
                "Ticker": t,
                "target": target_val,
                "pred": pred_val,
            })

    return pd.DataFrame(rows)


def test_perfect_prediction():
    df = make_synthetic_eval_df(perfect=True)
    results = evaluate_predictions(df, holding_period=5)

    assert np.isclose(results["ic_mean"], 1.0)
    assert np.isclose(results["rank_ic_mean"], 1.0)
    assert np.isclose(results["rmse"], 0.0)
    assert np.isclose(results["mae"], 0.0)
    assert np.isclose(results["r2"], 1.0)
    assert results["long_short_annualized_return_gross"] > 0.0


def test_inverted_prediction():
    df = make_synthetic_eval_df(inverted=True)
    results = evaluate_predictions(df, holding_period=5)

    assert np.isclose(results["ic_mean"], -1.0)
    assert np.isclose(results["rank_ic_mean"], -1.0)
    assert results["long_short_annualized_return_gross"] < 0.0


def test_degenerate_zero_variance():
    df = make_synthetic_eval_df(n_days=5, n_tickers=10)
    first_date = df["Date"].iloc[0]
    df.loc[df["Date"] == first_date, "pred"] = 0.05

    daily_ic = compute_daily_ic(df)
    assert np.isnan(daily_ic.loc[first_date])

    metrics = compute_ic_metrics(daily_ic)
    assert not np.isnan(metrics["ic_mean"])
    assert not np.isnan(metrics["ic_ir"])


def test_ic_metrics_edge_cases():
    empty_s = pd.Series([], dtype=float)
    m_empty = compute_ic_metrics(empty_s)
    assert np.isnan(m_empty["ic_mean"])
    assert np.isnan(m_empty["ic_ir"])
    assert m_empty["ic_n_days"] == 0

    one_s = pd.Series([0.05])
    m_one = compute_ic_metrics(one_s)
    assert np.isclose(m_one["ic_mean"], 0.05)
    assert np.isnan(m_one["ic_std"])
    assert np.isnan(m_one["ic_ir"])
    assert m_one["ic_n_days"] == 1

    const_s = pd.Series([0.05, 0.05, 0.05])
    m_const = compute_ic_metrics(const_s)
    assert np.isclose(m_const["ic_mean"], 0.05)
    assert np.isclose(m_const["ic_std"], 0.0)
    assert np.isnan(m_const["ic_ir"])
    assert m_const["ic_n_days"] == 3


def test_turnover_and_cost_deduction():
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    rebalance_dates = dates[::5]
    rows = []

    for t in ["A", "B", "C", "D"]:
        rows.append({"Date": rebalance_dates[0], "Ticker": t, "target": 0.05, "pred": 0.1})
    for t in ["W", "X", "Y", "Z"]:
        rows.append({"Date": rebalance_dates[1], "Ticker": t, "target": 0.02, "pred": 0.1})

    df = pd.DataFrame(rows)
    bt = compute_portfolio_backtest(df, holding_period=5, cost_bps=10.0, top_quantile=0.5, bottom_quantile=0.5)

    assert bt["long_only_annualized_return_gross"] > bt["long_only_annualized_return_net"]
    assert bt["long_short_annualized_return_gross"] > bt["long_short_annualized_return_net"]
    assert bt["long_only_mean_turnover"] == 1.0


def test_max_drawdown():
    returns = pd.Series([0.10, -0.20, 0.15, -0.40])
    mdd = _calc_max_drawdown(returns)
    wealth = (1.0 + returns).cumprod()
    peak = wealth.cummax()
    expected_mdd = float(((peak - wealth) / peak).max())
    assert np.isclose(mdd, expected_mdd)
    assert mdd > 0.40


def test_composite_evaluation():
    df = make_synthetic_eval_df(n_days=15, n_tickers=30)
    res = evaluate_predictions(df, holding_period=5, cost_bps=10.0)

    expected_keys = [
        "ic_mean", "ic_std", "ic_ir", "ic_naive_tstat", "ic_naive_pvalue", "ic_pct_positive", "ic_n_days",
        "rank_ic_mean", "rank_ic_std", "rank_ic_ir", "rank_ic_naive_tstat", "rank_ic_naive_pvalue", "rank_ic_pct_positive", "rank_ic_n_days",
        "rmse", "mae", "r2",
        "long_only_annualized_return_gross", "long_only_annualized_return_net",
        "long_only_annualized_vol", "long_only_sharpe_gross", "long_only_sharpe_net",
        "long_only_max_drawdown_gross", "long_only_max_drawdown_net",
        "long_only_mean_turnover", "long_only_excess_return_gross", "long_only_excess_return_net",
        "long_short_annualized_return_gross", "long_short_annualized_return_net",
        "long_short_annualized_vol", "long_short_sharpe_gross", "long_short_sharpe_net",
        "long_short_max_drawdown_gross", "long_short_max_drawdown_net",
        "long_short_mean_turnover",
        "benchmark_annualized_return", "benchmark_annualized_vol", "benchmark_sharpe", "benchmark_max_drawdown",
    ]

    for k in expected_keys:
        assert k in res, f"Missing key: {k}"
        assert not np.isnan(res[k]), f"NaN in key: {k}"


def test_quantile_validation_and_disjoint_selection():
    df = make_synthetic_eval_df(n_days=5, n_tickers=2)

    with pytest.raises(ValueError, match="must be strictly between 0 and 1"):
        compute_portfolio_backtest(df, top_quantile=0.0, bottom_quantile=0.5)

    with pytest.raises(ValueError, match="must be strictly between 0 and 1"):
        compute_portfolio_backtest(df, top_quantile=0.5, bottom_quantile=1.2)

    with pytest.raises(ValueError, match="cannot exceed 1.0|must be <= 1.0"):
        compute_portfolio_backtest(df, top_quantile=0.6, bottom_quantile=0.6)

    # 2 tickers: 1 long, 1 short, zero overlap
    bt = compute_portfolio_backtest(df, top_quantile=0.5, bottom_quantile=0.5)
    assert not np.isnan(bt["long_short_annualized_return_gross"])

