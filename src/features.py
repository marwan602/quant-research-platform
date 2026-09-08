from pathlib import Path
import numpy as np
import pandas as pd


def compute_ticker_alpha158(df: pd.DataFrame, windows=(5, 10, 20, 30, 60)) -> pd.DataFrame:
    close = df["Close"].values
    open_ = df["Open"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values
    eps = 1e-12

    if "VWAP" in df.columns:
        vwap = df["VWAP"].values
    else:
        vwap = (high + low + close) / 3.0

    feats = {}

    # K-bar features
    hl_range = high - low + eps
    feats["KMID"] = (close - open_) / (open_ + eps)
    feats["KLEN"] = (high - low) / (open_ + eps)
    feats["KMID2"] = (close - open_) / hl_range
    feats["KUP"] = (high - np.maximum(open_, close)) / (open_ + eps)
    feats["KUP2"] = (high - np.maximum(open_, close)) / hl_range
    feats["KLOW"] = (np.minimum(open_, close) - low) / (open_ + eps)
    feats["KLOW2"] = (np.minimum(open_, close) - low) / hl_range
    feats["KSFT"] = (2.0 * close - high - low) / (open_ + eps)
    feats["KSFT2"] = (2.0 * close - high - low) / hl_range

    # Price normalized by Close
    feats["OPEN0"] = open_ / (close + eps)
    feats["HIGH0"] = high / (close + eps)
    feats["LOW0"] = low / (close + eps)
    feats["VWAP0"] = vwap / (close + eps)

    s_close = pd.Series(close)
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    s_vol = pd.Series(volume)
    log_vol = np.log(volume + 1.0)
    s_log_vol = pd.Series(log_vol)

    ret1 = s_close.pct_change().fillna(0.0)
    delta_close = s_close.diff().fillna(0.0)
    abs_delta_close = delta_close.abs()
    up_close = delta_close.clip(lower=0.0)
    down_close = (-delta_close).clip(lower=0.0)

    delta_vol = s_vol.diff().fillna(0.0)
    abs_delta_vol = delta_vol.abs()
    up_vol = delta_vol.clip(lower=0.0)
    down_vol = (-delta_vol).clip(lower=0.0)

    vol_shift = s_vol.shift(1).fillna(0.0)
    vol_ret = np.log((s_vol / (vol_shift + eps)) + 1.0).fillna(0.0)
    wv = ret1.abs() * s_vol

    # Rolling window features
    for w in windows:
        # ROC: Ref(close, w) / close in Qlib
        feats[f"ROC{w}"] = (s_close.shift(w) / (s_close + eps)).values
        ma = s_close.rolling(w, min_periods=w).mean()
        std = s_close.rolling(w, min_periods=w).std()
        feats[f"MA{w}"] = (ma / (s_close + eps)).values
        feats[f"STD{w}"] = (std / (s_close + eps)).values

        # Linear regression trend: BETA, RSQR, RESI
        weights = np.arange(1, w + 1, dtype=float)
        x_mean = (w + 1.0) / 2.0
        x_var = np.sum((weights - x_mean) ** 2)

        sum_xy = np.convolve(close, weights[::-1], mode="full")[:len(close)]
        sum_xy[:w - 1] = np.nan
        sum_y = s_close.rolling(w, min_periods=w).sum().values
        cov_xy = sum_xy - x_mean * sum_y
        slope = cov_xy / (x_var + eps)
        var_y = s_close.rolling(w, min_periods=w).var().values * (w - 1)
        rsqr = np.where(var_y > eps, (cov_xy ** 2) / (x_var * var_y + eps), 0.0)
        y_mean = sum_y / w
        fitted_last = y_mean + slope * (w - x_mean)
        resi = close - fitted_last

        feats[f"BETA{w}"] = slope / (close + eps)
        feats[f"RSQR{w}"] = rsqr
        feats[f"RESI{w}"] = resi / (close + eps)

        # High/Low bounds, quantiles, and rank
        max_h = s_high.rolling(w, min_periods=w).max()
        min_l = s_low.rolling(w, min_periods=w).min()
        feats[f"MAX{w}"] = (max_h / (s_close + eps)).values
        feats[f"MIN{w}"] = (min_l / (s_close + eps)).values
        feats[f"QTLU{w}"] = (s_close.rolling(w, min_periods=w).quantile(0.8) / (s_close + eps)).values
        feats[f"QTLD{w}"] = (s_close.rolling(w, min_periods=w).quantile(0.2) / (s_close + eps)).values
        feats[f"RANK{w}"] = s_close.rolling(w, min_periods=w).rank(pct=True).values
        feats[f"RSV{w}"] = ((s_close - min_l) / (max_h - min_l + eps)).values

        # Days since peak / trough: (argmax + 1) / w in Qlib
        feats[f"IMAX{w}"] = s_high.rolling(w, min_periods=w).apply(
            lambda s: (np.argmax(s) + 1.0) / w, raw=True
        ).values
        feats[f"IMIN{w}"] = s_low.rolling(w, min_periods=w).apply(
            lambda s: (np.argmin(s) + 1.0) / w, raw=True
        ).values
        feats[f"IMXD{w}"] = feats[f"IMAX{w}"] - feats[f"IMIN{w}"]

        # Correlations
        feats[f"CORR{w}"] = s_close.rolling(w, min_periods=w).corr(s_log_vol).values
        feats[f"CORD{w}"] = ret1.rolling(w, min_periods=w).corr(vol_ret).values

        # Direction counts
        cnt_pos = (delta_close > 0).astype(float).rolling(w, min_periods=w).mean()
        cnt_neg = (delta_close < 0).astype(float).rolling(w, min_periods=w).mean()
        feats[f"CNTP{w}"] = cnt_pos.values
        feats[f"CNTN{w}"] = cnt_neg.values
        feats[f"CNTD{w}"] = (cnt_pos - cnt_neg).values

        # Price move sums
        sum_abs_c = abs_delta_close.rolling(w, min_periods=w).sum() + eps
        sum_p_c = up_close.rolling(w, min_periods=w).sum()
        sum_n_c = down_close.rolling(w, min_periods=w).sum()
        feats[f"SUMP{w}"] = (sum_p_c / sum_abs_c).values
        feats[f"SUMN{w}"] = (sum_n_c / sum_abs_c).values
        feats[f"SUMD{w}"] = ((sum_p_c - sum_n_c) / sum_abs_c).values

        # Volume dynamics
        v_mean = s_vol.rolling(w, min_periods=w).mean()
        v_std = s_vol.rolling(w, min_periods=w).std()
        feats[f"VMA{w}"] = (v_mean / (s_vol + eps)).values
        feats[f"VSTD{w}"] = (v_std / (s_vol + eps)).values
        feats[f"WVMA{w}"] = (
            wv.rolling(w, min_periods=w).std() / (wv.rolling(w, min_periods=w).mean() + eps)
        ).values

        sum_abs_v = abs_delta_vol.rolling(w, min_periods=w).sum() + eps
        sum_p_v = up_vol.rolling(w, min_periods=w).sum()
        sum_n_v = down_vol.rolling(w, min_periods=w).sum()
        feats[f"VSUMP{w}"] = (sum_p_v / sum_abs_v).values
        feats[f"VSUMN{w}"] = (sum_n_v / sum_abs_v).values
        feats[f"VSUMD{w}"] = ((sum_p_v - sum_n_v) / sum_abs_v).values

    out_df = pd.DataFrame(feats, index=df.index)
    return out_df


def compute_target(df: pd.DataFrame, horizon: int = 5) -> pd.Series:
    return df.groupby("Ticker")["Close"].shift(-horizon) / df["Close"] - 1.0


def compute_alpha158_dataset(
    prices_df: pd.DataFrame,
    composition_df: pd.DataFrame | None = None,
    horizon: int = 5,
    windows=(5, 10, 20, 30, 60),
) -> pd.DataFrame:
    df = prices_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    df["target"] = compute_target(df, horizon=horizon)

    feature_frames = []
    for ticker, group in df.groupby("Ticker", sort=False):
        feats = compute_ticker_alpha158(group, windows=windows)
        feature_frames.append(feats)

    feature_df = pd.concat(feature_frames, axis=0)
    result_df = pd.concat([df[["Date", "Ticker", "target"]], feature_df], axis=1)

    if composition_df is not None:
        comp = composition_df[["Date", "Ticker"]].copy()
        comp["Date"] = pd.to_datetime(comp["Date"])
        result_df = pd.merge(comp, result_df, on=["Date", "Ticker"], how="inner")

    result_df = result_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return result_df


def generate_and_cache_dataset(
    raw_prices_path: str = "data/raw/s_and_p_500_prices.parquet",
    raw_comp_path: str = "data/raw/s_and_p_500_daily_composition.parquet",
    output_path: str = "data/processed/sp500_alpha158.parquet",
    horizon: int = 5,
) -> pd.DataFrame:
    prices_df = pd.read_parquet(raw_prices_path)
    comp_df = pd.read_parquet(raw_comp_path)

    processed_df = compute_alpha158_dataset(
        prices_df=prices_df,
        composition_df=comp_df,
        horizon=horizon,
    )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_parquet(out_file, index=False)
    return processed_df


if __name__ == "__main__":
    generate_and_cache_dataset()
