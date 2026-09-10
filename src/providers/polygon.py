import os
from pathlib import Path
import time
from dotenv import load_dotenv
import pandas as pd
import requests

from src.providers.base import MarketDataProvider


def _load_env_credentials():
    curr = Path(__file__).resolve().parent
    for _ in range(4):
        cand = curr / ".env"
        if cand.exists():
            load_dotenv(dotenv_path=cand, override=False)
            break
        curr = curr.parent


class PolygonProvider(MarketDataProvider):
    BASE_URL = "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks"

    def __init__(
        self,
        api_key: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
        rate_limit_delay: float = 12.5,
    ):
        _load_env_credentials()
        key = api_key or os.getenv("POLYGON_API_KEY", "").strip()
        if key == "your_polygon_api_key_here":
            key = ""
        self.api_key = key
        self.session = session or requests.Session()
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay

    def _fetch_grouped_daily(self, date_str: str) -> dict:
        if not self.api_key:
            raise ValueError(
                "Polygon API key is required. Set POLYGON_API_KEY in .env, environment variable, or pass api_key."
            )

        url = f"{self.BASE_URL}/{date_str}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"adjusted": "true"}
        resp = self.session.get(url, params=params, headers=headers, timeout=self.timeout)

        if resp.status_code in (401, 403):
            raise PermissionError(f"Polygon authentication failed ({resp.status_code}): {resp.text}")
        if resp.status_code == 429:
            raise RuntimeError(f"Polygon rate limit exceeded: {resp.text}")
        if resp.status_code != 200:
            raise RuntimeError(f"Polygon request failed with status {resp.status_code}: {resp.text}")

        return resp.json()

    def _parse_grouped_results(self, data: dict, date_str: str) -> pd.DataFrame:
        results = data.get("results", [])
        if not results:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"])

        target_dt = pd.to_datetime(date_str)
        records = []
        for item in results:
            ticker = item.get("T")
            if not ticker:
                continue
            open_px = float(item.get("o", 0.0))
            high_px = float(item.get("h", 0.0))
            low_px = float(item.get("l", 0.0))
            close_px = float(item.get("c", 0.0))
            vol = float(item.get("v", 0.0))
            vwap = item.get("vw")
            if vwap is None or pd.isna(vwap) or float(vwap) <= 0.0:
                vwap = (high_px + low_px + close_px) / 3.0 if (high_px + low_px + close_px) > 0.0 else close_px
            else:
                vwap = float(vwap)

            records.append({
                "Date": target_dt,
                "Ticker": ticker,
                "Open": open_px,
                "High": high_px,
                "Low": low_px,
                "Close": close_px,
                "Volume": vol,
                "VWAP": vwap,
            })

        df = pd.DataFrame(records)
        cols_order = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"]
        return df[cols_order]

    def get_latest_bars(
        self,
        tickers: list[str],
        date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        payload = self._fetch_grouped_daily(date_str)
        parsed = self._parse_grouped_results(payload, date_str)

        if parsed.empty:
            return parsed

        target_set = set(tickers)
        filtered = parsed[parsed["Ticker"].isin(target_set)].sort_values(["Date", "Ticker"]).reset_index(drop=True)
        return filtered

    def get_historical_bars(
        self,
        tickers: list[str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        trading_days = [d for d in dates if d.weekday() < 5]

        frames = []
        for i, dt in enumerate(trading_days):
            day_df = self.get_latest_bars(tickers, dt)
            if not day_df.empty:
                frames.append(day_df)
            if self.rate_limit_delay > 0 and i < len(trading_days) - 1:
                time.sleep(self.rate_limit_delay)

        if not frames:
            return pd.DataFrame(columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume", "VWAP"])

        return pd.concat(frames, ignore_index=True).sort_values(["Date", "Ticker"]).reset_index(drop=True)
