"""
Tests for forecast_ensemble.py
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEnsemblePredictions:
    """Test ensemble_predictions() 加權平均邏輯。"""

    def test_weighted_average_60_40(self):
        from forecast_ensemble import ensemble_predictions
        lstm = [100.0, 110.0, 120.0]
        nhits = [105.0, 108.0, 115.0]
        result = ensemble_predictions(lstm, nhits)

        assert len(result) == 3
        # 100*0.6 + 105*0.4 = 60 + 42 = 102
        assert abs(result[0] - 102.0) < 0.01
        # 110*0.6 + 108*0.4 = 66 + 43.2 = 109.2
        assert abs(result[1] - 109.2) < 0.01
        # 120*0.6 + 115*0.4 = 72 + 46 = 118
        assert abs(result[2] - 118.0) < 0.01

    def test_lstm_longer_than_nhits(self):
        """LSTM 有 7 天但 N-HiTS 只有 5 天 → 後 2 天用純 LSTM。"""
        from forecast_ensemble import ensemble_predictions
        lstm = [100.0] * 7
        nhits = [110.0] * 5
        result = ensemble_predictions(lstm, nhits)

        assert len(result) == 7
        # 前 5 天: 100*0.6 + 110*0.4 = 104
        for i in range(5):
            assert abs(result[i] - 104.0) < 0.01
        # 後 2 天: 純 LSTM = 100
        assert result[5] == 100.0
        assert result[6] == 100.0

    def test_empty_nhits(self):
        from forecast_ensemble import ensemble_predictions
        lstm = [100.0, 110.0]
        nhits = []
        result = ensemble_predictions(lstm, nhits)
        # n=0, 所以全部是 lstm 的剩餘
        assert result == [100.0, 110.0]


class TestForecastNhits:
    """Test forecast_nhits() 的 graceful fallback。"""

    def test_returns_none_when_neuralforecast_not_installed(self):
        """如果 _HAS_NF=False，直接回 None。"""
        import forecast_ensemble as fe
        original = fe._HAS_NF
        try:
            fe._HAS_NF = False
            import pandas as pd
            import numpy as np
            dates = pd.bdate_range(end=pd.Timestamp.today(), periods=100)
            df = pd.DataFrame({"Close": np.random.randn(100) + 150}, index=dates)
            result = fe.forecast_nhits(df)
            assert result is None
        finally:
            fe._HAS_NF = original
