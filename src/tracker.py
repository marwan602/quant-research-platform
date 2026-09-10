from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        candidate = PROJECT_ROOT / p
        if candidate.exists() or candidate.parent.exists():
            return candidate
    return p


def get_realized_forward_returns(
    store_df: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    forward_days: int = 5,
) -> pd.DataFrame:
    target_dt = pd.to_datetime(as_of_date)
    all_dates = sorted(store_df["Date"].unique())
    if target_dt not in all_dates:
        return pd.DataFrame()

    idx = all_dates.index(target_dt)
    if idx + forward_days >= len(all_dates):
        return pd.DataFrame()

    future_dt = all_dates[idx + forward_days]

    base_slice = store_df[store_df["Date"] == target_dt].drop_duplicates(subset=["Ticker"])
    future_slice = store_df[store_df["Date"] == future_dt].drop_duplicates(subset=["Ticker"])

    base_prices = base_slice.set_index("Ticker")["Close"]
    future_prices = future_slice.set_index("Ticker")["Close"]

    common_tickers = base_prices.index.intersection(future_prices.index)
    if common_tickers.empty:
        return pd.DataFrame()

    base_common = base_prices.loc[common_tickers]
    future_common = future_prices.loc[common_tickers]

    returns = (future_common - base_common) / base_common
    res_df = pd.DataFrame({
        "Ticker": common_tickers,
        "base_date": target_dt,
        "resolved_date": future_dt,
        "realized_return_5d": returns.values,
    })
    return res_df


def compute_forward_trading_date(start_date: str | pd.Timestamp, forward_days: int = 5) -> str:
    dt = pd.to_datetime(start_date)
    count = 0
    curr = dt
    while count < forward_days:
        curr += pd.Timedelta(days=1)
        if curr.weekday() < 5:
            count += 1
    return curr.strftime("%Y-%m-%d")


def update_prediction_archive(
    archive_file: str | Path,
    new_rankings: list[dict],
    as_of_date: str,
    store_df: pd.DataFrame | None = None,
    forward_days: int = 5,
) -> dict:
    resolved_path = _resolve_path(archive_file)
    if resolved_path.exists():
        with open(resolved_path, "r", encoding="utf-8") as f:
            archive = json.load(f)
    else:
        archive = {"dates": {}}

    date_key = str(pd.to_datetime(as_of_date).strftime("%Y-%m-%d"))
    target_date_str = compute_forward_trading_date(date_key, forward_days=forward_days)

    if date_key not in archive["dates"]:
        archive["dates"][date_key] = {
            "as_of_date": date_key,
            "target_resolution_date": target_date_str,
            "status": "pending",
            "model": "AttentiveTransformer",
            "total_predictions": len(new_rankings),
            "rank_ic": None,
            "directional_accuracy": None,
            "predictions": [
                {**p, "target_resolution_date": target_date_str} for p in new_rankings
            ],
        }
    else:
        archive["dates"][date_key]["target_resolution_date"] = target_date_str
        for p in archive["dates"][date_key].get("predictions", []):
            if "target_resolution_date" not in p:
                p["target_resolution_date"] = target_date_str

    if store_df is not None and not store_df.empty:
        for d_str, entry in archive["dates"].items():
            if entry.get("status") == "resolved":
                continue

            realized_df = get_realized_forward_returns(store_df, d_str, forward_days=forward_days)
            if realized_df.empty:
                continue

            realized_map = dict(zip(realized_df["Ticker"], realized_df["realized_return_5d"]))

            preds = entry["predictions"]
            matched_pred = []
            matched_real = []

            for p in preds:
                t = p.get("ticker") or p.get("Ticker")
                if t in realized_map:
                    ret = float(realized_map[t])
                    p["realized_return_5d"] = round(ret, 5)
                    matched_pred.append(float(p.get("pred_return_5d", 0.0)))
                    matched_real.append(ret)
                else:
                    p["realized_return_5d"] = None

            if len(matched_pred) >= 10:
                ic_val, _ = spearmanr(matched_pred, matched_real)
                signs_match = np.sign(matched_pred) == np.sign(matched_real)
                dir_acc = float(np.mean(signs_match))

                entry["status"] = "resolved"
                entry["rank_ic"] = round(float(ic_val), 4) if not np.isnan(ic_val) else 0.0
                entry["directional_accuracy"] = round(dir_acc, 4)
                entry["resolved_at"] = datetime.now(timezone.utc).isoformat()

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(archive, f, indent=2)

    return archive


def update_forward_tracking(
    tracking_file: str | Path,
    archive: dict,
    store_df: pd.DataFrame | None = None,
    deployment_date: str = "2026-09-10",
    cost_bps: float = 15.0,
) -> dict:
    resolved_path = _resolve_path(tracking_file)
    dep_dt = pd.to_datetime(deployment_date)

    dates_dict = archive.get("dates", {})
    sorted_dates = sorted([d for d in dates_dict.keys() if pd.to_datetime(d) >= dep_dt])

    total_forecasts = sum(len(dates_dict[d]["predictions"]) for d in dates_dict)
    resolved_dates = [d for d in sorted_dates if dates_dict[d].get("status") == "resolved"]
    pending_dates = [d for d in sorted_dates if dates_dict[d].get("status") == "pending"]

    resolved_forecasts_count = sum(len(dates_dict[d]["predictions"]) for d in resolved_dates)
    pending_forecasts_count = sum(len(dates_dict[d]["predictions"]) for d in pending_dates)

    ic_values = [dates_dict[d]["rank_ic"] for d in resolved_dates if dates_dict[d]["rank_ic"] is not None]
    acc_values = [dates_dict[d]["directional_accuracy"] for d in resolved_dates if dates_dict[d]["directional_accuracy"] is not None]

    mean_rank_ic = round(float(np.mean(ic_values)), 4) if ic_values else None
    mean_dir_acc = round(float(np.mean(acc_values)), 4) if acc_values else None

    equity_curve = []
    current_model_nav = 1.0000
    current_bmark_nav = 1.0000
    peak_model_nav = 1.0000
    max_drawdown = 0.0

    equity_curve.append({
        "date": str(dep_dt.strftime("%Y-%m-%d")),
        "model_nav": 1.0000,
        "benchmark_nav": 1.0000,
        "model_return_pct": 0.0,
        "benchmark_return_pct": 0.0,
        "excess_alpha_pct": 0.0,
    })

    prev_weights = {}
    cost_rate = cost_bps / 10000.0
    period_model_returns = []

    for d in resolved_dates:
        entry = dates_dict[d]
        preds = entry["predictions"]
        top_decile_size = max(1, len(preds) // 10)
        sorted_preds = sorted(preds, key=lambda x: x.get("rank", 999))
        top_picks = sorted_preds[:top_decile_size]

        top_tickers = [p.get("ticker") or p.get("Ticker") for p in top_picks]
        curr_weights = {t: 1.0 / len(top_tickers) for t in top_tickers}

        all_holdings = set(curr_weights.keys()) | set(prev_weights.keys())
        turnover = 0.5 * sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_holdings)
        fee = turnover * cost_rate
        prev_weights = curr_weights

        model_rets = [p["realized_return_5d"] for p in top_picks if p.get("realized_return_5d") is not None]
        all_rets = [p["realized_return_5d"] for p in preds if p.get("realized_return_5d") is not None]

        if model_rets and all_rets:
            gross_ret = float(np.mean(model_rets))
            net_ret = gross_ret - fee
            b_ret = float(np.mean(all_rets))

            period_model_returns.append(net_ret)
            current_model_nav *= (1.0 + net_ret)
            current_bmark_nav *= (1.0 + b_ret)

            if current_model_nav > peak_model_nav:
                peak_model_nav = current_model_nav
            dd = (peak_model_nav - current_model_nav) / peak_model_nav
            if dd > max_drawdown:
                max_drawdown = dd

            equity_curve.append({
                "date": d,
                "model_nav": round(current_model_nav, 4),
                "benchmark_nav": round(current_bmark_nav, 4),
                "model_return_pct": round((current_model_nav - 1.0) * 100.0, 2),
                "benchmark_return_pct": round((current_bmark_nav - 1.0) * 100.0, 2),
                "excess_alpha_pct": round((current_model_nav - current_bmark_nav) * 100.0, 2),
            })

    total_model_ret_pct = round((current_model_nav - 1.0) * 100.0, 2)
    total_bmark_ret_pct = round((current_bmark_nav - 1.0) * 100.0, 2)
    net_alpha_pct = round((current_model_nav - current_bmark_nav) * 100.0, 2)

    if len(period_model_returns) >= 2 and np.std(period_model_returns, ddof=1) > 1e-8:
        mean_ret = np.mean(period_model_returns)
        std_ret = np.std(period_model_returns, ddof=1)
        realized_sharpe = round(float((mean_ret / std_ret) * np.sqrt(252.0 / 5.0)), 2)
    else:
        realized_sharpe = None

    tracking_payload = {
        "deployment_date": str(dep_dt.strftime("%Y-%m-%d")),
        "days_tracked": len(sorted_dates),
        "status": "active",
        "signal_metrics": {
            "mean_rank_ic": mean_rank_ic,
            "mean_directional_accuracy": mean_dir_acc,
            "total_forecasts": total_forecasts,
            "resolved_forecasts": resolved_forecasts_count,
            "pending_forecasts": pending_forecasts_count,
        },
        "portfolio_metrics": {
            "model_cumulative_return_pct": total_model_ret_pct,
            "benchmark_cumulative_return_pct": total_bmark_ret_pct,
            "net_alpha_pct": net_alpha_pct,
            "realized_sharpe": realized_sharpe,
            "max_drawdown_pct": round(max_drawdown * 100.0, 2),
        },
        "equity_curve": equity_curve,
    }

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w", encoding="utf-8") as f:
        json.dump(tracking_payload, f, indent=2)

    return tracking_payload
