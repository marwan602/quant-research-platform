from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    non_feature_cols = {"Date", "Ticker", "target"}
    return [col for col in df.columns if col not in non_feature_cols]


def load_processed_data(file_path: str = "data/processed/sp500_alpha158.parquet") -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    # Drop rows without targets (e.g. final horizon days) or where warm-up features are NaN
    df = df.dropna(subset=["target"]).copy()
    feature_cols = get_feature_columns(df)
    # Fill remaining NaNs (e.g. 0 variance rolling correlations) with 0.0
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df


def split_tabular_data(df: pd.DataFrame, config: dict | None = None):
    if config is None:
        config = load_config()

    split_cfg = config["split"]
    feature_cols = get_feature_columns(df)

    train_mask = (df["Date"] >= split_cfg["train_start"]) & (df["Date"] <= split_cfg["train_end"])
    val_mask = (df["Date"] >= split_cfg["val_start"]) & (df["Date"] <= split_cfg["val_end"])
    test_mask = (df["Date"] >= split_cfg["test_start"]) & (df["Date"] <= split_cfg["test_end"])

    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["target"].values.astype(np.float32)
    meta_train = train_df[["Date", "Ticker"]]

    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["target"].values.astype(np.float32)
    meta_val = val_df[["Date", "Ticker"]]

    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["target"].values.astype(np.float32)
    meta_test = test_df[["Date", "Ticker"]]

    return (
        (X_train, y_train, meta_train),
        (X_val, y_val, meta_val),
        (X_test, y_test, meta_test),
        feature_cols,
    )


class StockSequenceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        dates: np.ndarray,
        tickers: np.ndarray,
        lookback: int = 60,
    ):
        self.lookback = lookback
        self.samples = []

        # Group indices by ticker to build sequence lookbacks
        ticker_series = pd.Series(tickers)
        for ticker, idxs in ticker_series.groupby(ticker_series).groups.items():
            idx_list = idxs.values
            if len(idx_list) < lookback:
                continue
            for i in range(lookback - 1, len(idx_list)):
                target_idx = idx_list[i]
                window_idxs = idx_list[i - lookback + 1 : i + 1]
                self.samples.append((window_idxs, target_idx))

        self.features = features
        self.targets = targets
        self.dates = dates
        self.tickers = tickers

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_idxs, target_idx = self.samples[idx]
        x_seq = self.features[window_idxs]
        y_val = self.targets[target_idx]
        return torch.from_numpy(x_seq), torch.tensor(y_val, dtype=torch.float32)
