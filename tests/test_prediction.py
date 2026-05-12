"""
Tests for prediction.py
使用合成資料測試 compute_analysis()、_make_sequences_multi()、StockLSTM 等。
不呼叫 predict_stock()（需要 yfinance），只測內部元件。
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ohlcv(n: int = 250, base_price: float = 150.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.randint(1_000_000, 50_000_000, n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)


class TestMakeSequencesMulti:
    def test_output_shape(self):
        from prediction import _make_sequences_multi
        matrix = np.random.rand(100, 8).astype(np.float32)
        X, y = _make_sequences_multi(matrix, look_back=30)
        assert X.shape == (70, 30, 8), f"X shape: {X.shape}"
        assert y.shape == (70,), f"y shape: {y.shape}"

    def test_target_is_first_column(self):
        from prediction import _make_sequences_multi
        matrix = np.random.rand(50, 8).astype(np.float32)
        X, y = _make_sequences_multi(matrix, look_back=10)
        for i in range(len(y)):
            assert abs(y[i] - matrix[i + 10, 0]) < 1e-6

    def test_sequence_content(self):
        from prediction import _make_sequences_multi
        matrix = np.arange(35 * 3).reshape(35, 3).astype(np.float32)
        X, y = _make_sequences_multi(matrix, look_back=5)
        np.testing.assert_array_equal(X[0], matrix[0:5])
        np.testing.assert_array_equal(X[1], matrix[1:6])


class TestStockLSTM:
    def test_univariate_forward_pass(self):
        from prediction import StockLSTM
        model = StockLSTM(input_size=1, hidden_size=64)
        x = torch.randn(4, 30, 1)
        out = model(x)
        assert out.shape == (4, 1)

    def test_multivariate_forward_pass(self):
        from prediction import StockLSTM
        model = StockLSTM(input_size=8, hidden_size=128, dropout=0.2)
        x = torch.randn(4, 30, 8)
        out = model(x)
        assert out.shape == (4, 1)

    def test_dropout_param(self):
        from prediction import StockLSTM
        model = StockLSTM(input_size=8, hidden_size=128, dropout=0.3)
        assert model.lstm.dropout == 0.3


class TestComputeAnalysis:
    def test_returns_expected_keys(self):
        from prediction import compute_analysis
        df = _make_ohlcv()
        result = compute_analysis(df, full=False)
        expected_keys = [
            "current", "change_pct", "ma5", "ma20", "ma60",
            "rsi", "macd", "macd_signal",
            "boll_upper", "boll_lower",
            "vol_today", "vol_avg20", "vol_ratio",
            "scores", "total_score", "verdict",
            "add_position", "add_position_msg",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_total_score_in_valid_range(self):
        from prediction import compute_analysis
        df = _make_ohlcv()
        result = compute_analysis(df, full=False)
        assert 0 <= result["total_score"] <= 115

    def test_scores_structure(self):
        from prediction import compute_analysis
        df = _make_ohlcv()
        result = compute_analysis(df, full=False)
        scores = result["scores"]
        for key in ["trend", "momentum", "macd", "bollinger", "volume", "reversal"]:
            assert key in scores
            assert scores[key]["score"] <= scores[key]["max"]

    def test_featured_df_internal_key(self):
        from prediction import compute_analysis
        df = _make_ohlcv()
        result = compute_analysis(df, full=True)
        assert "_featured_df" in result

    def test_short_data_fallback(self):
        from prediction import compute_analysis
        df = _make_ohlcv(n=10)
        result = compute_analysis(df, full=False)
        assert "total_score" in result


class TestNormalizeAndDenormalize:
    def test_round_trip(self):
        from prediction import _normalize, _denormalize
        data = np.array([100, 110, 120, 130, 140], dtype=np.float32)
        normed, vmin, vmax = _normalize(data)
        recovered = _denormalize(normed, vmin, vmax)
        np.testing.assert_array_almost_equal(data, recovered, decimal=4)

    def test_constant_series(self):
        from prediction import _normalize, _denormalize
        data = np.array([50, 50, 50], dtype=np.float32)
        normed, vmin, vmax = _normalize(data)
        np.testing.assert_array_equal(normed, data)


class TestReturnBasedLSTMEndToEnd:
    """端對端測試：合成資料 → return-based 特徵 → 多變量 LSTM → 預測真實價格。"""

    def test_train_and_predict_gives_real_prices(self):
        from prediction import StockLSTM, _make_sequences_multi, _denormalize, LOOK_BACK, PRED_DAYS
        from feature_engine import get_lstm_feature_matrix
        from prediction_config import LSTM_HIDDEN_SIZE, LSTM_DROPOUT

        df = _make_ohlcv(n=120, base_price=180.0)
        feat_matrix, scalers = get_lstm_feature_matrix(df)
        n_features = feat_matrix.shape[1]

        X, y = _make_sequences_multi(feat_matrix, LOOK_BACK)
        assert len(X) > 10

        device = torch.device("cpu")
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)

        model = StockLSTM(
            input_size=n_features, hidden_size=LSTM_HIDDEN_SIZE, dropout=LSTM_DROPOUT
        ).to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

        # 快速訓練 30 epochs
        model.train()
        for _ in range(30):
            pred = model(X_t)
            loss = criterion(pred, y_t)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # 產生 return-based 預測
        model.eval()
        ret_vmin, ret_vmax = scalers["close_return"]
        last_close = scalers["_last_close"]
        last_window = feat_matrix[-LOOK_BACK:].copy()
        predictions = []
        current_price = last_close

        with torch.no_grad():
            for _ in range(PRED_DAYS):
                inp = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
                out_norm = model(inp).item()
                pred_return = out_norm * (ret_vmax - ret_vmin) + ret_vmin
                current_price = current_price * (1.0 + pred_return)
                predictions.append(current_price)
                new_row = last_window[-1].copy()
                new_row[0] = out_norm
                last_window = np.vstack([last_window[1:], new_row])

        actual_close = float(df["Close"].iloc[-1])
        for i, p in enumerate(predictions):
            # 預測值應在合理價格範圍
            assert p > 50, f"Day {i+1} prediction {p:.2f} < 50"
            assert p < 500, f"Day {i+1} prediction {p:.2f} > 500"
            # Return-based 預測應非常接近 last_close（± 15%）
            ratio = p / actual_close
            assert 0.85 < ratio < 1.15, \
                f"Day {i+1}: pred={p:.2f}, actual={actual_close:.2f}, ratio={ratio:.2f}"

    def test_zero_return_prediction_equals_last_price(self):
        """如果模型預測的正規化 return 對應 0% → 價格不變。"""
        from feature_engine import get_lstm_feature_matrix
        df = _make_ohlcv(base_price=200.0)
        _, scalers = get_lstm_feature_matrix(df)

        ret_vmin, ret_vmax = scalers["close_return"]
        last_close = scalers["_last_close"]

        # 找到 return=0 對應的正規化值
        zero_norm = (0.0 - ret_vmin) / (ret_vmax - ret_vmin)
        pred_return = zero_norm * (ret_vmax - ret_vmin) + ret_vmin
        pred_price = last_close * (1.0 + pred_return)

        assert abs(pred_price - last_close) < 0.01, \
            f"Expected ≈ {last_close:.2f}, got {pred_price:.2f}"
