import numpy as np
import pandas as pd
from src.features import compute_ticker_alpha158, compute_target


def make_dummy_stock(n_days=100, ticker="TEST"):
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_days)))
    high = close * np.random.uniform(1.001, 1.02, n_days)
    low = close * np.random.uniform(0.98, 0.999, n_days)
    open_ = low + np.random.uniform(0.0, 1.0, n_days) * (high - low)
    volume = np.random.uniform(1e5, 1e6, n_days)

    return pd.DataFrame({
        "Date": dates,
        "Ticker": ticker,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def test_alpha158_feature_count():
    df = make_dummy_stock(120)
    feats = compute_ticker_alpha158(df)
    assert feats.shape[1] == 158
    assert len(feats) == len(df)


def test_qlib_specific_formulas():
    df = make_dummy_stock(30)
    feats = compute_ticker_alpha158(df, windows=[5])

    # Qlib ROC: Ref(close, 5) / close
    expected_roc5 = df["Close"].shift(5) / (df["Close"] + 1e-12)
    assert np.isclose(feats["ROC5"].iloc[10], expected_roc5.iloc[10])

    # Qlib IMAX: (argmax + 1) / 5
    high_win = df["High"].iloc[6:11].values
    expected_imax5 = (np.argmax(high_win) + 1.0) / 5.0
    assert np.isclose(feats["IMAX5"].iloc[10], expected_imax5)

    # Qlib RANK: rolling rank pct
    expected_rank5 = df["Close"].iloc[6:11].rank(pct=True).iloc[-1]
    assert np.isclose(feats["RANK5"].iloc[10], expected_rank5)


def test_target_calculation():
    df = make_dummy_stock(20)
    target = compute_target(df, horizon=5)
    expected_0 = df.loc[5, "Close"] / df.loc[0, "Close"] - 1.0
    assert np.isclose(target.iloc[0], expected_0)
    assert target.iloc[-5:].isna().all()


def test_zero_lookahead_leakage():
    df_original = make_dummy_stock(100)
    feats_original = compute_ticker_alpha158(df_original)

    df_mutated = df_original.copy()
    df_mutated.loc[70:, "Close"] = df_mutated.loc[70:, "Close"] * 2.5
    df_mutated.loc[70:, "High"] = df_mutated.loc[70:, "High"] * 3.0
    df_mutated.loc[70:, "Volume"] = df_mutated.loc[70:, "Volume"] * 10.0

    feats_mutated = compute_ticker_alpha158(df_mutated)
    diff = np.nanmax(np.abs(feats_original.iloc[:70].values - feats_mutated.iloc[:70].values))
    assert diff < 1e-9
