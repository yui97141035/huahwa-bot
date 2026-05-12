"""
Tests for feature_engine.py
使用合成的 OHLCV 資料，不依賴網路。
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

# 確保可以 import 專案模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_ohlcv(n: int = 250, base_price: float = 150.0, seed: int = 42) -> pd.DataFrame:
    """產生 n 天的合成 OHLCV 資料。"""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    returns = rng.normal(0.0005, 0.015, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    open_ = close * (1 + rng.uniform(-0.01, 0.01, n))
    volume = rng.randint(1_000_000, 50_000_000, n).astype(float)

    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)
    return df


class TestComputeFeatures:
    """Test compute_features() 回傳正確的欄位和值域。"""

    def test_returns_dataframe_with_indicator_columns(self):
        from feature_engine import compute_features
        df = _make_ohlcv()
        result = compute_features(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)

        # 必須有 LSTM 需要的特徵欄位
        required = ["close_return", "return_5d", "rsi_14", "macd_hist", "bb_pct",
                     "vol_ratio", "atr_14", "adx_14", "close_norm"]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_close_return_is_pct_change(self):
        from feature_engine import compute_features
        df = _make_ohlcv()
        result = compute_features(df)
        cr = result["close_return"].dropna()
        # 日報酬率通常在 -10% ~ +10%
        assert cr.min() > -0.3, f"close_return min={cr.min()}"
        assert cr.max() < 0.3, f"close_return max={cr.max()}"

    def test_rsi_in_0_100_range(self):
        from feature_engine import compute_features
        df = _make_ohlcv()
        result = compute_features(df)
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, f"RSI min={rsi.min()}"
        assert rsi.max() <= 100, f"RSI max={rsi.max()}"

    def test_short_data_does_not_crash(self):
        from feature_engine import compute_features
        df = _make_ohlcv(n=15)
        result = compute_features(df)
        assert len(result) == 15

    def test_fallback_has_return_columns(self):
        """Fallback 路徑應也有 close_return 和 return_5d。"""
        from feature_engine import _compute_features_fallback
        df = _make_ohlcv()
        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        volume = df["Volume"].squeeze()
        result = _compute_features_fallback(df.copy(), close, high, low, volume)

        required = ["close_return", "return_5d", "rsi_14", "MACDh_12_26_9",
                     "BBP_20_2.0", "vol_ratio", "atr_14", "adx_14"]
        for col in required:
            assert col in result.columns, f"Fallback missing: {col}"


class TestGetLstmFeatureMatrix:
    """Test get_lstm_feature_matrix() — return-based 特徵矩陣。"""

    def test_matrix_shape(self):
        from feature_engine import get_lstm_feature_matrix
        from prediction_config import LSTM_FEATURES
        df = _make_ohlcv()
        matrix, scalers = get_lstm_feature_matrix(df)
        assert matrix.shape == (250, len(LSTM_FEATURES)), f"Shape: {matrix.shape}"
        assert matrix.dtype == np.float32

    def test_matrix_values_in_0_1(self):
        from feature_engine import get_lstm_feature_matrix
        df = _make_ohlcv()
        matrix, _ = get_lstm_feature_matrix(df)
        assert matrix.min() >= -0.01, f"Min: {matrix.min()}"
        assert matrix.max() <= 1.01, f"Max: {matrix.max()}"

    def test_close_return_scaler_is_return_range(self):
        """close_return 的 scaler 應是日報酬率的 (min, max)，不是價格範圍。"""
        from feature_engine import get_lstm_feature_matrix
        df = _make_ohlcv(base_price=150.0)
        _, scalers = get_lstm_feature_matrix(df)

        vmin, vmax = scalers["close_return"]
        # 日報酬率通常在 -5% ~ +5%
        assert -0.3 < vmin < 0, f"close_return scaler vmin={vmin}"
        assert 0 < vmax < 0.3, f"close_return scaler vmax={vmax}"

    def test_last_close_in_scalers(self):
        """scalers 應包含 _last_close 供價格推算。"""
        from feature_engine import get_lstm_feature_matrix
        df = _make_ohlcv(base_price=150.0)
        _, scalers = get_lstm_feature_matrix(df)

        assert "_last_close" in scalers
        last_close = scalers["_last_close"]
        actual = float(df["Close"].iloc[-1])
        assert abs(last_close - actual) < 0.01

    def test_return_based_prediction_gives_real_prices(self):
        """模擬：LSTM 預測 return=0 → 價格不變（≈ last_close）。"""
        from feature_engine import get_lstm_feature_matrix
        from prediction import _denormalize
        df = _make_ohlcv(base_price=150.0)
        _, scalers = get_lstm_feature_matrix(df)

        ret_vmin, ret_vmax = scalers["close_return"]
        last_close = scalers["_last_close"]

        # 如果模型預測正規化空間的 0.5 → return 大約是 (vmin+vmax)/2 ≈ 0
        mid_norm = 0.5
        pred_return = mid_norm * (ret_vmax - ret_vmin) + ret_vmin
        pred_price = last_close * (1.0 + pred_return)

        # 預測應非常接近 last_close（因為 midpoint return ≈ 0）
        assert abs(pred_price - last_close) / last_close < 0.03, \
            f"pred={pred_price:.2f}, actual={last_close:.2f}"

    def test_reuse_featured_df(self):
        from feature_engine import compute_features, get_lstm_feature_matrix
        df = _make_ohlcv()
        featured = compute_features(df)

        m1, s1 = get_lstm_feature_matrix(df)
        m2, s2 = get_lstm_feature_matrix(df, featured=featured)

        np.testing.assert_array_almost_equal(m1, m2, decimal=5)
