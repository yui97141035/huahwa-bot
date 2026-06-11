"""
Tests for signal_gate.py — Quad Verification Gates
"""

import sys
import os
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
def _make_featured_df(n=200, rsi=50.0, adx=25.0, macd_hist=0.01, bb_pct=0.4,
                      vol_ratio=1.2, close_start=100.0):
    """Generate a synthetic featured DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = close_start + np.cumsum(np.random.randn(n) * 0.5)
    close = pd.Series(close, index=dates, name="Close")

    df = pd.DataFrame({
        "Open": close - 0.5,
        "High": close + 1.0,
        "Low": close - 1.0,
        "Close": close,
        "Volume": np.random.randint(100000, 500000, n),
        "rsi_14": rsi + np.random.randn(n) * 3,
        "adx_14": adx + np.random.randn(n) * 2,
        "macd_hist": macd_hist + np.random.randn(n) * 0.005,
        "bb_pct": np.clip(bb_pct + np.random.randn(n) * 0.1, 0, 1),
        "vol_ratio": np.clip(vol_ratio + np.random.randn(n) * 0.2, 0.1, 5.0),
        "atr_14": np.abs(np.random.randn(n) * 2 + 1),
        "close_return": np.random.randn(n) * 0.01,
        "return_5d": np.random.randn(n) * 0.03,
        "ma5": close.rolling(5).mean(),
        "ma20": close.rolling(20).mean(),
        "ma60": close.rolling(60).mean(),
        "MACD_12_26_9": macd_hist + np.random.randn(n) * 0.01,
        "MACDs_12_26_9": np.random.randn(n) * 0.005,
        "MACDh_12_26_9": macd_hist + np.random.randn(n) * 0.005,
        "BBU_20_2.0": close + 5,
        "BBL_20_2.0": close - 5,
        "BBM_20_2.0": close,
        "BBP_20_2.0": np.clip(bb_pct + np.random.randn(n) * 0.1, 0, 1),
        "mfi_14": 50 + np.random.randn(n) * 10,
        "willr_14": -50 + np.random.randn(n) * 15,
    }, index=dates)
    return df


def _make_analysis(score=65, rsi=50.0, featured_df=None):
    """Create a minimal analysis dict like compute_analysis() returns."""
    return {
        "total_score": score,
        "rsi": rsi,
        "current": 100.0,
        "change_pct": 0.5,
        "ma5": 99.0,
        "ma20": 98.0,
        "ma60": 97.0,
        "macd": 0.01,
        "macd_signal": 0.005,
        "boll_upper": 105.0,
        "boll_lower": 95.0,
        "vol_today": 200000,
        "vol_avg20": 180000,
        "vol_ratio": 1.1,
        "scores": {
            "trend": {"score": 18, "max": 25, "detail": ""},
            "momentum": {"score": 15, "max": 20, "detail": ""},
            "macd": {"score": 14, "max": 20, "detail": ""},
            "bollinger": {"score": 10, "max": 15, "detail": ""},
            "volume": {"score": 8, "max": 20, "detail": ""},
            "reversal": {"score": 0, "max": 15, "detail": ""},
        },
        "verdict": "🟢 適合進場佈局",
        "add_position": False,
        "add_position_msg": "",
        "_featured_df": featured_df,
    }


# ---------------------------------------------------------------------------
# Test: classify_ticker
# ---------------------------------------------------------------------------
class TestClassifyTicker:
    def test_etf_tw(self):
        from signal_gate import classify_ticker
        assert classify_ticker("0050.TW") == "etf_tw"
        assert classify_ticker("00878.TW") == "etf_tw"
        assert classify_ticker("006208.TW") == "etf_tw"

    def test_stock_tw(self):
        from signal_gate import classify_ticker
        assert classify_ticker("2330.TW") == "stock_tw"
        assert classify_ticker("2454.TW") == "stock_tw"

    def test_volatile_us(self):
        from signal_gate import classify_ticker
        assert classify_ticker("RGTI") == "volatile_us"
        assert classify_ticker("IONQ") == "volatile_us"

    def test_stock_us(self):
        from signal_gate import classify_ticker
        assert classify_ticker("AAPL") == "stock_us"
        assert classify_ticker("TSLA") == "stock_us"


# ---------------------------------------------------------------------------
# Test: Gate 1 — Technical
# ---------------------------------------------------------------------------
class TestGateTechnical:
    def test_pass_high_score(self):
        from signal_gate import gate_technical
        featured = _make_featured_df(rsi=50.0, adx=25.0)
        analysis = _make_analysis(score=70, rsi=50.0, featured_df=featured)
        result = gate_technical(analysis, "stock_tw")
        assert result.passed is True
        assert result.score == 70.0

    def test_fail_low_score(self):
        from signal_gate import gate_technical
        analysis = _make_analysis(score=40, rsi=50.0)
        result = gate_technical(analysis, "stock_tw")
        assert result.passed is False

    def test_fail_rsi_overbought(self):
        from signal_gate import gate_technical
        featured = _make_featured_df(rsi=80.0, adx=25.0)
        analysis = _make_analysis(score=70, rsi=80.0, featured_df=featured)
        result = gate_technical(analysis, "stock_tw")
        assert result.passed is False
        assert "RSI" in result.details

    def test_fail_low_adx(self):
        from signal_gate import gate_technical
        featured = _make_featured_df(rsi=50.0, adx=5.0)
        analysis = _make_analysis(score=70, rsi=50.0, featured_df=featured)
        result = gate_technical(analysis, "stock_tw")
        assert result.passed is False
        assert "ADX" in result.details

    def test_no_featured_df_adx_passes(self):
        from signal_gate import gate_technical
        analysis = _make_analysis(score=70, rsi=50.0, featured_df=None)
        result = gate_technical(analysis, "stock_tw")
        # ADX should auto-pass when no featured_df
        assert result.passed is True

    def test_threshold_boundary(self):
        from signal_gate import gate_technical
        # Score=46 should pass (threshold=45 after calibration)
        analysis = _make_analysis(score=46, rsi=50.0)
        result = gate_technical(analysis, "etf_tw")
        assert result.passed is True
        # Score=44 should fail (below threshold=45)
        analysis2 = _make_analysis(score=44, rsi=50.0)
        result2 = gate_technical(analysis2, "stock_tw")
        assert result2.passed is False


# ---------------------------------------------------------------------------
# Test: Gate 2 — ML Direction (fallback)
# ---------------------------------------------------------------------------
class TestGateMLDirection:
    def test_no_featured_df_autopass(self):
        from signal_gate import gate_ml_direction
        analysis = _make_analysis(featured_df=None)
        result = gate_ml_direction(analysis, "2330.TW")
        assert result.passed is True
        assert "auto-pass" in result.details

    def test_live_mode_autopass(self):
        """In live mode (default), Gate 2 should auto-pass."""
        from signal_gate import gate_ml_direction
        featured = _make_featured_df(n=200)
        analysis = _make_analysis(featured_df=featured)
        result = gate_ml_direction(analysis, "2330.TW")
        assert result.passed is True
        assert "auto-pass" in result.details

    def test_lgbm_fallback_when_not_installed(self):
        """When allow_training=True but lightgbm unavailable, Gate 2 should auto-pass."""
        import signal_gate
        original = signal_gate._HAS_LIGHTGBM
        try:
            signal_gate._HAS_LIGHTGBM = False
            featured = _make_featured_df(n=200)
            analysis = _make_analysis(featured_df=featured)
            result = signal_gate.gate_ml_direction(analysis, "2330.TW", allow_training=True)
            assert result.passed is True
            assert "fallback" in result.details
        finally:
            signal_gate._HAS_LIGHTGBM = original

    def test_insufficient_data_autopass(self):
        from signal_gate import gate_ml_direction
        # Only 50 rows, not enough for train_window=120
        featured = _make_featured_df(n=50)
        analysis = _make_analysis(featured_df=featured)
        result = gate_ml_direction(analysis, "AAPL", allow_training=True)
        assert result.passed is True
        assert "fallback" in result.details or "insufficient" in result.details


# ---------------------------------------------------------------------------
# Test: Gate 3 — Macro
# ---------------------------------------------------------------------------
class TestGateMacro:
    def test_low_vix_passes(self):
        from signal_gate import gate_macro
        result = gate_macro("2330.TW", vix=18.0, sentiment=None)
        assert result.passed is True

    def test_high_vix_fails(self):
        from signal_gate import gate_macro
        result = gate_macro("2330.TW", vix=35.0, sentiment=None)
        assert result.passed is False
        assert "VIX" in result.details

    def test_no_vix_passes(self):
        from signal_gate import gate_macro
        result = gate_macro("AAPL", vix=None, sentiment=None)
        assert result.passed is True

    def test_fear_greed_extreme_fear_fails(self):
        from signal_gate import gate_macro
        sentiment = {"fear_greed": {"score": 10}, "twii": None, "vix": None}
        result = gate_macro("AAPL", vix=20.0, sentiment=sentiment)
        assert result.passed is False
        assert "F&G" in result.details

    def test_fear_greed_neutral_passes(self):
        from signal_gate import gate_macro
        sentiment = {"fear_greed": {"score": 50}, "twii": None, "vix": None}
        result = gate_macro("AAPL", vix=20.0, sentiment=sentiment)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Test: evaluate_entry — integration
# ---------------------------------------------------------------------------
class TestEvaluateEntry:
    """Tests disable Gate 4 (agent consensus) to avoid 180s timeout."""

    @patch("prediction_config.ENABLE_AGENT_GATE", False)
    def test_all_gates_pass_high_confidence(self):
        from signal_gate import evaluate_entry
        analysis = _make_analysis(score=70, rsi=50.0)
        signal = evaluate_entry(analysis, "0050.TW", vix=18.0, sentiment=None)
        # Gate 1: score=70 >= 45(etf_tw), RSI=50 < 75, ADX=N/A(pass) → pass
        # Gate 2: no featured_df → auto-pass
        # Gate 3: VIX=18 < 30 → pass
        # Gate 4: disabled → auto-pass
        assert signal.confidence == "HIGH"
        assert signal.gates_passed == 4
        assert signal.should_alert is True

    @patch("prediction_config.ENABLE_AGENT_GATE", False)
    def test_low_score_not_high(self):
        from signal_gate import evaluate_entry
        analysis = _make_analysis(score=30, rsi=50.0)
        signal = evaluate_entry(analysis, "2330.TW", vix=18.0, sentiment=None)
        # Gate 1: score=30 < 60 → fail
        assert signal.confidence != "HIGH"
        assert signal.should_alert is False

    @patch("prediction_config.ENABLE_AGENT_GATE", False)
    def test_high_vix_reduces_confidence(self):
        from signal_gate import evaluate_entry
        analysis = _make_analysis(score=70, rsi=50.0)
        signal = evaluate_entry(analysis, "AAPL", vix=35.0, sentiment=None)
        # Gate 3 fails (VIX too high), G1+G2+G4 pass = 3
        assert signal.gates_passed == 3
        assert signal.confidence == "MEDIUM"
        assert signal.should_alert is False

    @patch("prediction_config.ENABLE_AGENT_GATE", False)
    def test_gates_passed_count(self):
        from signal_gate import evaluate_entry
        analysis = _make_analysis(score=40, rsi=80.0)
        signal = evaluate_entry(analysis, "2330.TW", vix=35.0, sentiment=None)
        # Gate 1: score=40 < 45 → fail, RSI=80 > 75 → fail
        # Gate 2: auto-pass
        # Gate 3: VIX=35 > 30 → fail
        # Gate 4: disabled → auto-pass
        assert signal.gates_passed == 2
        assert signal.confidence == "LOW"

    @patch("prediction_config.ENABLE_AGENT_GATE", False)
    def test_medium_confidence_three_gates(self):
        from signal_gate import evaluate_entry
        # Gate 1 passes (score=70, rsi=50), Gate 2 auto-pass, Gate 3 fails (VIX=35), Gate 4 disabled auto-pass
        analysis = _make_analysis(score=70, rsi=50.0)
        signal = evaluate_entry(analysis, "0050.TW", vix=35.0, sentiment=None)
        assert signal.gates_passed == 3
        assert signal.confidence == "MEDIUM"
        assert signal.should_alert is False


# ---------------------------------------------------------------------------
# Test: _prepare_ml_features
# ---------------------------------------------------------------------------
class TestPrepareMLFeatures:
    def test_all_features_present(self):
        from signal_gate import _prepare_ml_features, _GATE2_FEATURES
        featured = _make_featured_df(n=200)
        result = _prepare_ml_features(featured)
        for col in _GATE2_FEATURES:
            assert col in result.columns, f"Missing feature: {col}"

    def test_handles_missing_columns(self):
        from signal_gate import _prepare_ml_features, _GATE2_FEATURES
        # Minimal DataFrame without most features
        df = pd.DataFrame({
            "Close": [100.0] * 50,
            "Volume": [100000] * 50,
        })
        result = _prepare_ml_features(df)
        for col in _GATE2_FEATURES:
            assert col in result.columns
