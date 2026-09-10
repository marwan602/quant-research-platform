import copy
import json
import math
import os
from pathlib import Path
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.dataset import load_config, load_processed_data, prepare_sequence_data
from src.evaluate import evaluate_predictions


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, dropout: float = 0.2, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TemporalAttention(nn.Module):
    def __init__(self, d_model: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1, bias=False),
        )

    def forward(self, H: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.projection(H)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(weights * H, dim=1)
        return context, weights.squeeze(-1)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int = 158,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_proj = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=100)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.attention = TemporalAttention(d_model=d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature_proj(x)
        h = self.pos_encoder(h)
        encoded = self.transformer_encoder(h)
        h_last = encoded[:, -1, :]
        context, _ = self.attention(encoded)
        combined = torch.cat([context, h_last], dim=-1)
        out = self.head(combined)
        return out.squeeze(-1)


def train_transformer(
    train_ds: Dataset,
    val_ds: Dataset,
    config: dict | None = None,
    device: str | None = None,
) -> tuple[TransformerModel, list[dict]]:
    if config is None:
        config = load_config()

    tf_cfg = config.get("models", {}).get("transformer", {})
    d_model = tf_cfg.get("d_model", 64)
    nhead = tf_cfg.get("nhead", 4)
    num_layers = tf_cfg.get("num_layers", 2)
    dim_feedforward = tf_cfg.get("dim_feedforward", 128)
    dropout = tf_cfg.get("dropout", 0.1)
    lr = tf_cfg.get("learning_rate", 0.0005)
    batch_size = tf_cfg.get("batch_size", 512)
    epochs = tf_cfg.get("epochs", 30)
    patience = tf_cfg.get("early_stopping_patience", 7)
    seed = config.get("models", {}).get("lightgbm", {}).get("random_state", 42)

    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    first_sample, _ = train_ds[0]
    input_size = first_sample.shape[-1]

    model = TransformerModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_samples = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_count = len(y_batch)
            train_loss_total += loss.item() * batch_count
            train_samples += batch_count

        train_loss = train_loss_total / max(train_samples, 1)

        model.eval()
        val_loss_total = 0.0
        val_samples = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_preds = model(x_val)
                loss = criterion(val_preds, y_val)
                batch_count = len(y_val)
                val_loss_total += loss.item() * batch_count
                val_samples += batch_count

        val_loss = val_loss_total / max(val_samples, 1)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        history.append(epoch_metrics)
        print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss:.6f}", flush=True)
                break

    model.load_state_dict(best_weights)
    return model, history


def predict_transformer(
    model: nn.Module,
    dataset_or_loader: Dataset | DataLoader,
    device: str | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(dataset_or_loader, DataLoader):
        loader = dataset_or_loader
    else:
        loader = DataLoader(
            dataset_or_loader,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                x_batch = batch[0]
            else:
                x_batch = batch
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            predictions.append(preds.cpu().numpy())

    if not predictions:
        return np.array([], dtype=np.float32)
    return np.concatenate(predictions, axis=0).astype(np.float32)


def save_model(model: nn.Module, save_path: str = "models/transformer_model.pt") -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(
    model_path: str = "models/transformer_model.pt",
    input_size: int = 158,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    dropout: float = 0.2,
    device: str = "cpu",
) -> TransformerModel:
    model = TransformerModel(
        input_size=input_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def save_scaler(scaler: object, save_path: str = "models/transformer_scaler.pkl") -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(scaler_path: str = "models/transformer_scaler.pkl") -> object:
    return joblib.load(scaler_path)


def run_pipeline(
    config_path: str = "configs/config.yaml",
    output_dir: str = "reports",
    model_dir: str = "models",
) -> dict:
    config = load_config(config_path)
    seed = config.get("models", {}).get("lightgbm", {}).get("random_state", 42)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_processed_data(config["data"]["processed_features_path"])
    lookback = config.get("features", {}).get("lookback_window", 60)

    train_ds, val_ds, test_ds, scaler, feat_cols = prepare_sequence_data(
        df, config=config, lookback=lookback
    )

    save_scaler(scaler, save_path=f"{model_dir}/transformer_scaler.pkl")

    model, history = train_transformer(train_ds, val_ds, config=config, device=device)
    save_model(model, save_path=f"{model_dir}/transformer_model.pt")

    tf_cfg = config.get("models", {}).get("transformer", {})
    batch_size = tf_cfg.get("batch_size", 512)
    pred_test = predict_transformer(model, test_ds, device=device, batch_size=batch_size)

    eval_df = test_ds.get_metadata()
    eval_df["pred"] = pred_test

    holding_period = config.get("target", {}).get("horizon", 5)
    metrics = evaluate_predictions(eval_df, holding_period=holding_period)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    metrics_path = out_p / "transformer_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    predictions_path = out_p / "transformer_test_predictions.parquet"
    eval_df.to_parquet(predictions_path, index=False)

    history_path = out_p / "transformer_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_pipeline()
