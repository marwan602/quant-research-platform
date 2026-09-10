import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset

from src.models.transformer_model import (
    PositionalEncoding,
    TransformerModel,
    load_model,
    load_scaler,
    predict_transformer,
    save_model,
    save_scaler,
    set_seed,
    train_transformer,
)


def test_positional_encoding_shape_and_values():
    batch_size = 8
    seq_len = 60
    d_model = 64

    pe = PositionalEncoding(d_model=d_model, max_len=100)
    x = torch.zeros(batch_size, seq_len, d_model)
    out = pe(x)

    assert out.shape == (batch_size, seq_len, d_model)
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])
    assert torch.allclose(out[0], out[1])


def test_transformer_forward_pass():
    batch_size = 4
    seq_len = 60
    input_size = 158
    d_model = 64

    model = TransformerModel(
        input_size=input_size,
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    x = torch.randn(batch_size, seq_len, input_size)

    out = model(x)
    assert out.shape == (batch_size,)


def test_transformer_backward_pass():
    batch_size = 4
    seq_len = 60
    input_size = 158

    model = TransformerModel(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    x = torch.randn(batch_size, seq_len, input_size)
    y = torch.randn(batch_size)

    criterion = torch.nn.MSELoss()
    preds = model(x)
    loss = criterion(preds, y)
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient missing for {name}"


def test_transformer_deterministic_reproducibility():
    set_seed(42)
    m1 = TransformerModel(input_size=10, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    x = torch.randn(2, 15, 10)
    m1.eval()
    with torch.no_grad():
        out1 = m1(x)

    set_seed(42)
    m2 = TransformerModel(input_size=10, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    m2.eval()
    with torch.no_grad():
        out2 = m2(x)

    assert torch.allclose(out1, out2, atol=1e-7)


def test_save_and_load_model(tmp_path):
    model = TransformerModel(input_size=10, d_model=16, nhead=2, num_layers=1, dim_feedforward=32)
    model_path = tmp_path / "transformer_test.pt"

    save_model(model, str(model_path))
    assert model_path.exists()

    loaded = load_model(
        str(model_path),
        input_size=10,
        d_model=16,
        nhead=2,
        num_layers=1,
        dim_feedforward=32,
    )
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
            "transformer": {
                "d_model": 16,
                "nhead": 2,
                "num_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "learning_rate": 0.01,
                "batch_size": 16,
                "epochs": 2,
                "early_stopping_patience": 2,
            },
            "lightgbm": {"random_state": 42},
        }
    }

    model, history = train_transformer(ds, ds, config=config, device="cpu")
    assert len(history) == 2
    assert "train_loss" in history[0]
    assert "val_loss" in history[0]

    preds = predict_transformer(model, ds, device="cpu", batch_size=16)
    assert preds.shape == (n_samples,)
    assert not np.isnan(preds).any()
