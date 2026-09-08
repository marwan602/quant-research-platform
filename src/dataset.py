from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    feature_cols = get_feature_columns(df)
    df = df.dropna(subset=["target"] + feature_cols).copy()
    df = df.sort_values(["Date", "Ticker"]).reset_index(drop=True)
    return df


def split_tabular_data(df: pd.DataFrame, config: dict | None = None):
    if config is None:
        config = load_config()

    split_cfg = config["split"]
    train_start = pd.to_datetime(split_cfg["train_start"])
    train_end = pd.to_datetime(split_cfg["train_end"])
    val_start = pd.to_datetime(split_cfg["val_start"])
    val_end = pd.to_datetime(split_cfg["val_end"])
    test_start = pd.to_datetime(split_cfg["test_start"])
    test_end = pd.to_datetime(split_cfg["test_end"])

    if not (train_start <= train_end < val_start <= val_end < test_start <= test_end):
        raise ValueError(
            "Split ranges must be strictly chronological and non-overlapping: "
            "train_end < val_start and val_end < test_start"
        )

    feature_cols = get_feature_columns(df)

    train_mask = (df["Date"] >= train_start) & (df["Date"] <= train_end)
    val_mask = (df["Date"] >= val_start) & (df["Date"] <= val_end)
    test_mask = (df["Date"] >= test_start) & (df["Date"] <= test_end)

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
        target_start_date: str | pd.Timestamp | None = None,
        target_end_date: str | pd.Timestamp | None = None,
    ):
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer >= 1")

        self.lookback = lookback
        self.samples = []
        self.features = features
        self.targets = targets
        self.dates = pd.to_datetime(dates)
        self.tickers = np.asarray(tickers)

        start_dt = pd.to_datetime(target_start_date) if target_start_date is not None else None
        end_dt = pd.to_datetime(target_end_date) if target_end_date is not None else None

        ticker_series = pd.Series(self.tickers)
        for ticker, idxs in ticker_series.groupby(ticker_series).groups.items():
            idx_list = idxs.values
            if len(idx_list) < lookback:
                continue
            for i in range(lookback - 1, len(idx_list)):
                target_idx = idx_list[i]
                target_dt = self.dates[target_idx]
                if start_dt is not None and target_dt < start_dt:
                    continue
                if end_dt is not None and target_dt > end_dt:
                    continue
                window_idxs = idx_list[i - lookback + 1 : i + 1]
                self.samples.append((window_idxs, target_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        window_idxs, target_idx = self.samples[idx]
        x_seq = self.features[window_idxs]
        y_val = self.targets[target_idx]
        return torch.from_numpy(x_seq), torch.tensor(y_val, dtype=torch.float32)

    def get_metadata(self) -> pd.DataFrame:
        if not self.samples:
            return pd.DataFrame(columns=["Date", "Ticker", "target"])
        target_indices = [target_idx for _, target_idx in self.samples]
        return pd.DataFrame({
            "Date": self.dates[target_indices].values,
            "Ticker": self.tickers[target_indices],
            "target": self.targets[target_indices],
        })


def prepare_sequence_data(
    df: pd.DataFrame,
    config: dict | None = None,
    lookback: int = 60,
    scaler: StandardScaler | None = None,
    clip_val: float = 5.0,
):
    if config is None:
        config = load_config()

    split_cfg = config["split"]
    train_start = pd.to_datetime(split_cfg["train_start"])
    train_end = pd.to_datetime(split_cfg["train_end"])
    val_start = pd.to_datetime(split_cfg["val_start"])
    val_end = pd.to_datetime(split_cfg["val_end"])
    test_start = pd.to_datetime(split_cfg["test_start"])
    test_end = pd.to_datetime(split_cfg["test_end"])

    if not (train_start <= train_end < val_start <= val_end < test_start <= test_end):
        raise ValueError(
            "Split ranges must be strictly chronological and non-overlapping: "
            "train_end < val_start and val_end < test_start"
        )

    feature_cols = get_feature_columns(df)
    sorted_df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    sorted_df["Date"] = pd.to_datetime(sorted_df["Date"])

    features_raw = sorted_df[feature_cols].values.astype(np.float32)
    targets = sorted_df["target"].values.astype(np.float32)
    dates = sorted_df["Date"].values
    tickers = sorted_df["Ticker"].values

    train_mask = (sorted_df["Date"] >= train_start) & (sorted_df["Date"] <= train_end)

    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(features_raw[train_mask.values])

    features_scaled = scaler.transform(features_raw).astype(np.float32)
    if clip_val is not None:
        features_scaled = np.clip(features_scaled, -clip_val, clip_val)

    train_ds = StockSequenceDataset(
        features_scaled,
        targets,
        dates,
        tickers,
        lookback=lookback,
        target_start_date=train_start,
        target_end_date=train_end,
    )
    val_ds = StockSequenceDataset(
        features_scaled,
        targets,
        dates,
        tickers,
        lookback=lookback,
        target_start_date=val_start,
        target_end_date=val_end,
    )
    test_ds = StockSequenceDataset(
        features_scaled,
        targets,
        dates,
        tickers,
        lookback=lookback,
        target_start_date=test_start,
        target_end_date=test_end,
    )

    return train_ds, val_ds, test_ds, scaler, feature_cols

