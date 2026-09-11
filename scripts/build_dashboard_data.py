from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fetch_or_load_company_meta(cache_path: Path) -> dict[str, dict]:
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            df = pd.read_csv(resp)
    except Exception:
        df = pd.DataFrame()

    meta = {}
    if not df.empty:
        for _, r in df.iterrows():
            sym = str(r["Symbol"]).strip()
            meta[sym] = {
                "ticker": sym,
                "name": str(r["Security"]).strip(),
                "sector": str(r["GICS Sector"]).strip(),
                "sub_industry": str(r["GICS Sub-Industry"]).strip(),
            }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def compute_factor_percentiles(raw_prices_path: Path, target_tickers: list[str]) -> dict[str, dict]:
    if not raw_prices_path.exists():
        return {}

    df = pd.read_parquet(raw_prices_path)
    df["Date"] = pd.to_datetime(df["Date"])
    latest_date = df["Date"].max()
    sub_df = df[df["Date"] >= latest_date - pd.Timedelta(days=130)].copy()

    records = {}
    for ticker in target_tickers:
        ticker_slice = sub_df[sub_df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
        if len(ticker_slice) < 25:
            continue
        c = ticker_slice["Close"].values
        v = ticker_slice["Volume"].values

        mom = float((c[-1] - c[-20]) / (c[-20] + 1e-8)) if len(c) >= 20 else 0.0
        rets = np.diff(c[-21:]) / (c[-21:-1] + 1e-8) if len(c) >= 21 else [0.0]
        vol = float(np.std(rets))
        ma20 = float(np.mean(c[-20:])) if len(c) >= 20 else float(c[-1])
        bias = float((c[-1] - ma20) / (ma20 + 1e-8))
        liq = float(np.mean(v[-20:])) if len(v) >= 20 else float(v[-1])
        ma10 = float(np.mean(c[-10:])) if len(c) >= 10 else float(c[-1])
        ma50 = float(np.mean(c[-50:])) if len(c) >= 50 else float(c[-1])
        trend = float((ma10 / (ma50 + 1e-8)) - 1.0)

        records[ticker] = {
            "momentum_raw": mom,
            "volatility_raw": vol,
            "reversal_raw": bias,
            "liquidity_raw": liq,
            "trend_raw": trend,
        }

    if not records:
        return {}

    feat_df = pd.DataFrame.from_dict(records, orient="index")
    for col in ["momentum_raw", "volatility_raw", "reversal_raw", "liquidity_raw", "trend_raw"]:
        clean_col = col.replace("_raw", "")
        feat_df[clean_col] = (feat_df[col].rank(pct=True) * 100.0).round(1)

    result = {}
    for ticker, row in feat_df.iterrows():
        result[ticker] = {
            "momentum": float(row["momentum"]),
            "volatility": float(row["volatility"]),
            "reversal": float(row["reversal"]),
            "liquidity": float(row["liquidity"]),
            "trend": float(row["trend"]),
        }
    return result


def simulate_model_series(df: pd.DataFrame, rebalance_dates: list, cost_rate: float = 10.0 / 10000.0):
    lo_net, ls_net, bench, dates = [], [], [], []
    prev_lo, prev_ls = {}, {}

    for d in rebalance_dates:
        day_slice = df[df["Date"] == d]
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
        dates.append(d.strftime("%Y-%m-%d"))

    w_lo = ((1.0 + pd.Series(lo_net)).cumprod() - 1.0) * 100.0
    w_ls = ((1.0 + pd.Series(ls_net)).cumprod() - 1.0) * 100.0
    w_bench = ((1.0 + pd.Series(bench)).cumprod() - 1.0) * 100.0

    nav_lo = (1.0 + pd.Series(lo_net)).cumprod()
    peak_lo = nav_lo.cummax()
    dd_lo = ((nav_lo - peak_lo) / peak_lo) * 100.0

    nav_bench = (1.0 + pd.Series(bench)).cumprod()
    peak_bench = nav_bench.cummax()
    dd_bench = ((nav_bench - peak_bench) / peak_bench) * 100.0

    return {
        "dates": dates,
        "lo_net_pct": [round(float(x), 2) for x in w_lo],
        "ls_net_pct": [round(float(x), 2) for x in w_ls],
        "bench_pct": [round(float(x), 2) for x in w_bench],
        "dd_lo_pct": [round(float(x), 2) for x in dd_lo],
        "dd_bench_pct": [round(float(x), 2) for x in dd_bench],
    }


def compute_daily_rank_ic_series(df: pd.DataFrame) -> dict:
    clean_df = df.dropna(subset=["Date", "target", "pred"]).copy()
    rank_ics = []
    dates = []
    for date, group in clean_df.groupby("Date", sort=True):
        if len(group) < 2 or group["target"].nunique() <= 1 or group["pred"].nunique() <= 1:
            continue
        corr = group["target"].corr(group["pred"], method="spearman")
        if not np.isnan(corr):
            rank_ics.append(float(corr))
            dates.append(date.strftime("%Y-%m-%d"))

    cum_ic = list(np.cumsum(rank_ics))
    return {
        "dates": dates,
        "daily_rank_ic": [round(float(x), 4) for x in rank_ics],
        "cumulative_rank_ic": [round(float(x), 3) for x in cum_ic],
    }


def generate_all_dashboard_data(
    project_root: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    root = project_root or PROJECT_ROOT
    docs_data = (output_dir or root / "docs/data").resolve()
    docs_data.mkdir(parents=True, exist_ok=True)

    meta_file = docs_data / "company_meta.json"
    company_meta = fetch_or_load_company_meta(meta_file)

    rankings_src = root / "reports/live/rankings.json"
    raw_prices_path = root / "data/raw/s_and_p_500_prices.parquet"

    target_tickers = []
    if rankings_src.exists():
        with open(rankings_src, "r", encoding="utf-8") as f:
            base_rankings = json.load(f)
        target_tickers = [x["ticker"] for x in base_rankings.get("rankings", [])]

    factor_percentiles = compute_factor_percentiles(raw_prices_path, target_tickers)

    if rankings_src.exists():
        with open(rankings_src, "r", encoding="utf-8") as f:
            rankings_data = json.load(f)

        for item in rankings_data.get("rankings", []):
            t = item["ticker"]
            meta = company_meta.get(t, {})
            item["name"] = meta.get("name", t)
            item["sector"] = meta.get("sector", "Unclassified")
            item["sub_industry"] = meta.get("sub_industry", "General")
            feats = factor_percentiles.get(t, {})
            item["factors"] = {
                "momentum": feats.get("momentum", 50.0),
                "volatility": feats.get("volatility", 50.0),
                "reversal": feats.get("reversal", 50.0),
                "liquidity": feats.get("liquidity", 50.0),
                "trend": feats.get("trend", 50.0),
            }

        with open(docs_data / "rankings.json", "w", encoding="utf-8") as f:
            json.dump(rankings_data, f, indent=2)

    portfolio_src = root / "reports/live/portfolio.json"
    if portfolio_src.exists():
        with open(portfolio_src, "r", encoding="utf-8") as f:
            port_data = json.load(f)

        sector_counts = {}
        total_holdings = len(port_data.get("holdings", []))
        for h in port_data.get("holdings", []):
            t = h["ticker"]
            meta = company_meta.get(t, {})
            h["name"] = meta.get("name", t)
            h["sector"] = meta.get("sector", "Unclassified")
            sec = h["sector"]
            sector_counts[sec] = sector_counts.get(sec, 0) + 1

        port_data["sector_composition"] = [
            {
                "sector": sec,
                "count": count,
                "weight_pct": round((count / max(1, total_holdings)) * 100.0, 1),
            }
            for sec, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        tf_pq = root / "reports/transformer_test_predictions.parquet"
        if tf_pq.exists():
            df_tf_prior = pd.read_parquet(tf_pq)
            df_tf_prior["Date"] = pd.to_datetime(df_tf_prior["Date"])
            last_dt = df_tf_prior["Date"].max()
            prior_slice = df_tf_prior[df_tf_prior["Date"] == last_dt].sort_values("pred", ascending=False).head(50)
            prior_holdings = []
            for _, r in prior_slice.iterrows():
                sym = str(r["Ticker"]).strip()
                meta = company_meta.get(sym, {})
                prior_holdings.append({
                    "ticker": sym,
                    "weight": 0.02,
                    "name": meta.get("name", sym),
                    "sector": meta.get("sector", "Unclassified"),
                })
            port_data["prior_cycle"] = {
                "as_of_date": last_dt.strftime("%Y-%m-%d"),
                "holdings": prior_holdings,
            }

        with open(docs_data / "portfolio.json", "w", encoding="utf-8") as f:
            json.dump(port_data, f, indent=2)

    for copy_name in ["system_status.json", "forward_tracking.json"]:
        src = root / f"reports/live/{copy_name}"
        if src.exists():
            with open(src, "r", encoding="utf-8") as sf:
                payload = json.load(sf)
            with open(docs_data / copy_name, "w", encoding="utf-8") as df:
                json.dump(payload, df, indent=2)

    archive_src = root / "reports/live/prediction_archive.json"
    if archive_src.exists():
        with open(archive_src, "r", encoding="utf-8") as sf:
            arch_payload = json.load(sf)
        for d_str, entry in arch_payload.get("dates", {}).items():
            dt = pd.to_datetime(d_str)
            count = 0
            curr = dt
            while count < 5:
                curr += pd.Timedelta(days=1)
                if curr.weekday() < 5:
                    count += 1
            target_str = curr.strftime("%Y-%m-%d")
            entry["target_resolution_date"] = target_str
            for p in entry.get("predictions", []):
                p["target_resolution_date"] = target_str

        with open(docs_data / "prediction_archive.json", "w", encoding="utf-8") as df:
            json.dump(arch_payload, df, indent=2)
        with open(archive_src, "w", encoding="utf-8") as df:
            json.dump(arch_payload, df, indent=2)

    tf_pq = root / "reports/transformer_test_predictions.parquet"
    alstm_pq = root / "reports/alstm_test_predictions.parquet"
    lgb_pq = root / "reports/lightgbm_test_predictions.parquet"

    if tf_pq.exists() and alstm_pq.exists() and lgb_pq.exists():
        df_tf = pd.read_parquet(tf_pq)
        df_tf["Date"] = pd.to_datetime(df_tf["Date"])
        clean_tf = df_tf.dropna(subset=["Date", "Ticker", "target", "pred"]).copy()

        df_alstm = pd.read_parquet(alstm_pq)
        df_alstm["Date"] = pd.to_datetime(df_alstm["Date"])
        clean_alstm = df_alstm.dropna(subset=["Date", "Ticker", "target", "pred"]).copy()

        df_lgb = pd.read_parquet(lgb_pq)
        df_lgb["Date"] = pd.to_datetime(df_lgb["Date"])
        clean_lgb = df_lgb.dropna(subset=["Date", "Ticker", "target", "pred"]).copy()

        unique_dates = sorted(clean_tf["Date"].unique())
        rebal_dates = unique_dates[::5]

        tf_res = simulate_model_series(clean_tf, rebal_dates)
        alstm_res = simulate_model_series(clean_alstm, rebal_dates)
        lgb_res = simulate_model_series(clean_lgb, rebal_dates)

        tf_ic = compute_daily_rank_ic_series(clean_tf)
        alstm_ic = compute_daily_rank_ic_series(clean_alstm)
        lgb_ic = compute_daily_rank_ic_series(clean_lgb)

        backtest_payload = {
            "period": "2024-01-02 to 2026-08-20",
            "rebalance_dates": tf_res["dates"],
            "transformer_lo": tf_res["lo_net_pct"],
            "transformer_ls": tf_res["ls_net_pct"],
            "alstm_lo": alstm_res["lo_net_pct"],
            "lightgbm_lo": lgb_res["lo_net_pct"],
            "benchmark": tf_res["bench_pct"],
            "drawdowns": {
                "dates": tf_res["dates"],
                "transformer": tf_res["dd_lo_pct"],
                "lightgbm": lgb_res["dd_lo_pct"],
                "alstm": alstm_res["dd_lo_pct"],
                "benchmark": tf_res["dd_bench_pct"],
            },
            "rank_ic": {
                "dates": tf_ic["dates"],
                "transformer_cumulative": tf_ic["cumulative_rank_ic"],
                "alstm_cumulative": alstm_ic["cumulative_rank_ic"],
                "lightgbm_cumulative": lgb_ic["cumulative_rank_ic"],
            },
        }

        with open(docs_data / "backtest_series.json", "w", encoding="utf-8") as f:
            json.dump(backtest_payload, f, indent=2)

    benchmarks_payload = {
        "evaluation_period": "January 2024 to August 2026",
        "holding_period_days": 5,
        "transaction_cost_bps": 10,
        "models": [
            {
                "id": "transformer",
                "name": "Attentive Transformer (Production)",
                "annualized_net_return_pct": 29.05,
                "annualized_net_vol_pct": 24.45,
                "net_sharpe": 1.188,
                "max_drawdown_pct": 25.15,
                "excess_return_pct": 15.37,
                "ic_mean": 0.0269,
                "rank_ic_mean": 0.0227,
                "rank_ic_ir": 2.058,
                "pct_positive_days": 53.54,
                "parameters": 98881,
            },
            {
                "id": "alstm",
                "name": "Attentive LSTM",
                "annualized_net_return_pct": 24.58,
                "annualized_net_vol_pct": 23.08,
                "net_sharpe": 1.065,
                "max_drawdown_pct": 23.88,
                "excess_return_pct": 10.90,
                "ic_mean": 0.0260,
                "rank_ic_mean": 0.0123,
                "rank_ic_ir": 1.265,
                "pct_positive_days": 51.73,
                "parameters": 104513,
            },
            {
                "id": "lightgbm",
                "name": "LightGBM GBDT Baseline",
                "annualized_net_return_pct": 25.21,
                "annualized_net_vol_pct": 22.75,
                "net_sharpe": 1.108,
                "max_drawdown_pct": 24.33,
                "excess_return_pct": 11.26,
                "ic_mean": 0.0152,
                "rank_ic_mean": 0.0118,
                "rank_ic_ir": 1.161,
                "pct_positive_days": 52.94,
                "parameters": 31000,
            },
            {
                "id": "benchmark",
                "name": "S&P 500 Equal-Weighted Universe",
                "annualized_net_return_pct": 13.69,
                "annualized_net_vol_pct": 12.94,
                "net_sharpe": 1.058,
                "max_drawdown_pct": 15.79,
                "excess_return_pct": 0.0,
                "ic_mean": None,
                "rank_ic_mean": None,
                "rank_ic_ir": None,
                "pct_positive_days": None,
                "parameters": 0,
            },
        ],
    }

    with open(docs_data / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmarks_payload, f, indent=2)


if __name__ == "__main__":
    generate_all_dashboard_data()
