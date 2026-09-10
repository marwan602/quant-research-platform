import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch

from src.features import compute_ticker_alpha158
from src.models.transformer_model import load_model
from src.providers.base import LocalProvider, UniverseProvider
from src.providers.store import RollingPriceStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return p


def prepare_inference_features(
    trailing_df: pd.DataFrame,
    scaler: object,
    required_feature_steps: int = 60,
    clip_val: float = 5.0,
    as_of_date: str | pd.Timestamp | None = None,
) -> tuple[torch.Tensor, list[str], pd.Timestamp]:
    if trailing_df.empty:
        raise ValueError("trailing_df is empty, cannot generate inference features")

    trailing_df = trailing_df.copy()
    trailing_df["Date"] = pd.to_datetime(trailing_df["Date"])
    target_as_of = pd.to_datetime(as_of_date) if as_of_date is not None else trailing_df["Date"].max()
    tickers = trailing_df["Ticker"].unique()

    feature_slices = []
    valid_tickers = []

    min_raw_bars = 59 + required_feature_steps
    for ticker in sorted(tickers):
        sub = trailing_df[trailing_df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if sub.empty or sub["Date"].iloc[-1] != target_as_of:
            continue
        if len(sub) < min_raw_bars:
            continue

        raw_feats = compute_ticker_alpha158(sub)
        clean_feats = raw_feats.iloc[59:].fillna(0.0)
        if len(clean_feats) < required_feature_steps:
            continue

        seq_feats = clean_feats.iloc[-required_feature_steps:].values.astype(np.float32)
        feature_slices.append(seq_feats)
        valid_tickers.append(ticker)

    if not feature_slices:
        raise ValueError("No tickers had sufficient valid historical bars to form a 60-step sequence")

    arr = np.stack(feature_slices, axis=0)
    n_stocks, seq_len, n_feats = arr.shape

    arr_flat = arr.reshape(-1, n_feats)
    arr_scaled = scaler.transform(arr_flat)
    if clip_val is not None:
        arr_scaled = np.clip(arr_scaled, -clip_val, clip_val)

    scaled_tensor = torch.from_numpy(arr_scaled.reshape(n_stocks, seq_len, n_feats)).float()
    return scaled_tensor, valid_tickers, target_as_of


class PortfolioConstructor:
    @staticmethod
    def construct_portfolios(
        predictions_df: pd.DataFrame,
        top_quantile: float = 0.1,
        bottom_quantile: float = 0.1,
    ) -> dict:
        df = predictions_df.sort_values("pred_return_5d", ascending=False).reset_index(drop=True)
        n = len(df)
        n_top = max(1, int(np.floor(n * top_quantile)))
        n_bottom = max(1, int(np.floor(n * bottom_quantile)))

        df["rank"] = np.arange(1, n + 1)
        df["decile"] = pd.qcut(df["rank"], q=10, labels=list(range(1, 11))).astype(int)

        lo_weights = {}
        for i, row in df.iterrows():
            ticker = row["Ticker"]
            if row["rank"] <= n_top:
                lo_weights[ticker] = 1.0 / n_top
            else:
                lo_weights[ticker] = 0.0

        ls_weights = {}
        for i, row in df.iterrows():
            ticker = row["Ticker"]
            if row["rank"] <= n_top:
                ls_weights[ticker] = 0.5 / n_top
            elif row["rank"] > n - n_bottom:
                ls_weights[ticker] = -0.5 / n_bottom
            else:
                ls_weights[ticker] = 0.0

        df["long_only_weight"] = df["Ticker"].map(lo_weights)
        df["long_short_weight"] = df["Ticker"].map(ls_weights)

        return {
            "rankings": df,
            "long_only_weights": lo_weights,
            "long_short_weights": ls_weights,
            "top_tickers": df.iloc[:n_top]["Ticker"].tolist(),
            "bottom_tickers": df.iloc[n - n_bottom:]["Ticker"].tolist(),
        }

    @staticmethod
    def compute_rebalance_orders(
        target_weights: dict[str, float],
        current_weights: dict[str, float] | None = None,
        portfolio_value: float = 100000.0,
    ) -> pd.DataFrame:
        if current_weights is None:
            current_weights = {}

        all_tickers = sorted(set(target_weights.keys()) | set(current_weights.keys()))
        orders = []

        for t in all_tickers:
            tgt = target_weights.get(t, 0.0)
            cur = current_weights.get(t, 0.0)
            diff = tgt - cur
            if abs(diff) < 1e-5:
                continue

            action = "BUY" if diff > 0 else "SELL"
            dollar_amount = abs(diff) * portfolio_value

            orders.append({
                "Ticker": t,
                "Action": action,
                "TargetWeight": tgt,
                "CurrentWeight": cur,
                "WeightDelta": diff,
                "TradeValue": dollar_amount,
            })

        orders_df = pd.DataFrame(orders)
        if not orders_df.empty:
            orders_df = orders_df.sort_values("TradeValue", ascending=False).reset_index(drop=True)
        return orders_df


class InferenceEngine:
    def __init__(
        self,
        model_path: str = "models/transformer_model.pt",
        scaler_path: str = "models/transformer_scaler.pkl",
        device: str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.scaler = joblib.load(str(_resolve_path(scaler_path)))
        self.model = load_model(str(_resolve_path(model_path)), device=self.device)
        self.model.eval()

    def predict(self, trailing_df: pd.DataFrame, as_of_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
        x_tensor, tickers, evaluated_date = prepare_inference_features(
            trailing_df=trailing_df,
            scaler=self.scaler,
            required_feature_steps=60,
            as_of_date=as_of_date,
        )

        with torch.no_grad():
            x_device = x_tensor.to(self.device)
            preds = self.model(x_device).cpu().numpy().reshape(-1)

        results = pd.DataFrame({
            "Date": evaluated_date,
            "Ticker": tickers,
            "pred_return_5d": preds,
        })
        results = results.sort_values("pred_return_5d", ascending=False).reset_index(drop=True)
        results["rank"] = np.arange(1, len(results) + 1)
        return results


def run_pipeline(
    as_of_date: str | pd.Timestamp | None = None,
    model_path: str = "models/transformer_model.pt",
    scaler_path: str = "models/transformer_scaler.pkl",
    store_path: str = "data/raw/s_and_p_500_prices.parquet",
    composition_path: str = "data/raw/s_and_p_500_daily_composition.parquet",
    device: str | None = None,
) -> dict:
    universe_provider = UniverseProvider(composition_path=str(_resolve_path(composition_path)))
    active_tickers = universe_provider.get_active_tickers(as_of_date=as_of_date)

    store = RollingPriceStore(initial_path=str(_resolve_path(store_path)))
    trailing_df = store.get_trailing_window(
        tickers=active_tickers,
        lookback_days=125,
        as_of_date=as_of_date,
    )

    engine = InferenceEngine(
        model_path=str(_resolve_path(model_path)),
        scaler_path=str(_resolve_path(scaler_path)),
        device=device,
    )
    preds_df = engine.predict(trailing_df, as_of_date=as_of_date)

    portfolio_data = PortfolioConstructor.construct_portfolios(preds_df)
    return portfolio_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of-date", type=str, default="2026-08-31")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    result = run_pipeline(as_of_date=args.as_of_date)
    rankings = result["rankings"]

    print(f"\nTop {args.top_n} Stock Forecasts as of {args.as_of_date}:")
    print(rankings[["rank", "Ticker", "pred_return_5d", "long_only_weight"]].head(args.top_n).to_string(index=False))
