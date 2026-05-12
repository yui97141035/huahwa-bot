"""
Tests for prediction_config.py
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPredictionConfig:
    def test_default_values(self):
        import prediction_config as cfg
        assert cfg.ENABLE_PANDAS_TA is True
        assert cfg.ENABLE_MULTIVARIATE_LSTM is True
        assert cfg.ENABLE_NHITS is True
        assert cfg.ENABLE_SENTIMENT is True
        assert cfg.ENABLE_QUANTSTATS is True
        assert cfg.LSTM_INPUT_SIZE == 8
        assert cfg.LSTM_HIDDEN_SIZE == 128
        assert cfg.LSTM_DROPOUT == pytest.approx(0.2)
        assert cfg.LSTM_GRAD_CLIP == pytest.approx(1.0)
        assert cfg.LSTM_WARMUP_EPOCHS == 30
        assert cfg.ENSEMBLE_LSTM_WEIGHT == pytest.approx(0.6)
        assert cfg.ENSEMBLE_NHITS_WEIGHT == pytest.approx(0.4)

    def test_lstm_features_list(self):
        import prediction_config as cfg
        assert len(cfg.LSTM_FEATURES) == 8
        assert cfg.LSTM_FEATURES[0] == "close_return"

    def test_env_override_bool(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_ENABLE_PANDAS_TA", "false")
        # 重新 import 才能套用 env
        import importlib
        import prediction_config as cfg
        importlib.reload(cfg)
        assert cfg.ENABLE_PANDAS_TA is False

        # 恢復
        monkeypatch.delenv("OPENCLAW_ENABLE_PANDAS_TA", raising=False)
        importlib.reload(cfg)

    def test_env_override_int(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_LSTM_HIDDEN_SIZE", "256")
        import importlib
        import prediction_config as cfg
        importlib.reload(cfg)
        assert cfg.LSTM_HIDDEN_SIZE == 256

        monkeypatch.delenv("OPENCLAW_LSTM_HIDDEN_SIZE", raising=False)
        importlib.reload(cfg)

    def test_env_override_float(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_LSTM_DROPOUT", "0.5")
        import importlib
        import prediction_config as cfg
        importlib.reload(cfg)
        assert cfg.LSTM_DROPOUT == pytest.approx(0.5)

        monkeypatch.delenv("OPENCLAW_LSTM_DROPOUT", raising=False)
        importlib.reload(cfg)
