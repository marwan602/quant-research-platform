import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.dataset import StockSequenceDataset
from src.models.alstm_model import (
    ALSTMModel,
    TemporalAttention,
    load_model,
    load_scaler,
    predict_alstm,
    save_model,
    save_scaler,
    set_seed,
    train_alstm,
)


def test_temporal_attention_shapes_and_weights():
    batch_size = 8
    seq_len = 60
    hidden_size = 64

    attention = TemporalAttention(hidden_size=hidden_size)
    H = torch.randn(batch_size, seq_len, hidden_size)

    context, weights = attention(H)

    assert context.shape == (batch_size, hidden_size)
    assert weights.shape == (batch_size, seq_len)

    weight_sums = weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones(batch_size), atol=1e-5)


def test_alstm_forward_pass():
    batch_size = 4
    seq_len = 60
    input_size = 158
    hidden_size = 64

    model = ALSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=2)
    x = torch.randn(batch_size, seq_len, input_size)

    out = model(x)
    assert out.shape == (batch_size,)


def test_alstm_backward_pass():
    batch_size = 4
    seq_len = 60
    input_size = 158

    model = ALSTMModel(input_size=input_size, hidden_size=64, num_layers=2)
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size)

    criterion = torch.nn.MSELoss()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


def test_alstm_deterministic_reproducibility():
    set_seed(42)
    m1 = ALSTMModel(input_size=10, hidden_size=16, num_layers=2)
    x = torch.randn(2, 15, 10)
    m1.eval()
    with torch.no_grad():
        out1 = m1(x)

    set_seed(42)
    m2 = ALSTMModel(input_size=10, hidden_size=16, num_layers=2)
    m2.eval()
    with torch.no_grad():
        out2 = m2(x)

    assert torch.allclose(out1, out2, atol=1e-7)


def test_save_and_load_model(tmp_path):
    model = ALSTMModel(input_size=10, hidden_size=16, num_layers=2)
    model_path = tmp_path / "alstm_test.pt"

    save_model(model, str(model_path))
    assert model_path.exists()

    loaded = load_model(str(model_path), input_size=10, hidden_size=16, num_layers=2)
    x = torch.randn(2, 15, 10)

    model.eval()
    loaded.eval()
    with torch.no_grad():
        orig_out = model(x)
        loaded_out = loaded(x)

    assert torch.allclose(orig_out, loaded_out, atol=1e-7)


def test_save_and_load_scaler(tmp_path):
    data = np.random.randn(100, 10)
    scaler = StandardScaler()
    scaler.fit(data)

    scaler_path = tmp_path / "scaler_test.pkl"
    save_scaler(scaler, str(scaler_path))
    assert scaler_path.exists()

    loaded = load_scaler(str(scaler_path))
    transformed_orig = scaler.transform(data)
    transformed_loaded = loaded.transform(data)
    assert np.allclose(transformed_orig, transformed_loaded)


def test_train_and_predict_synthetic():
    n_samples = 40
    seq_len = 10
    n_features = 8

    x_data = torch.randn(n_samples, seq_len, n_features)
    y_data = torch.randn(n_samples)
    ds = TensorDataset(x_data, y_data)

    config = {
        "models": {
            "alstm": {
                "hidden_size": 16,
                "num_layers": 1,
                "dropout": 0.0,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 2,
                "early_stopping_patience": 2,
            },
            "lightgbm": {"random_state": 42},
        }
    }

    model, history = train_alstm(ds, ds, config=config, device="cpu")
    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]

    preds = predict_alstm(model, ds, device="cpu", batch_size=16)
    assert preds.shape == (n_samples,)
    assert not np.isnan(preds).any()
