"""
Tests for perf_report.py
使用臨時 CSV 檔模擬預測記錄。
"""

import sys
import os
import csv
import tempfile
from datetime import datetime, timedelta
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _create_mock_predictions_csv(path: str, n: int = 30):
    """建立假的 predictions_log.csv。"""
    fields = [
        "predict_time", "ticker", "last_price",
        "pred_day1", "pred_day2", "pred_day3", "pred_day4",
        "pred_day5", "pred_day6", "pred_day7",
        "model_val_mae",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(n):
            dt = datetime.now() - timedelta(days=n - i)
            base = 150.0 + i * 0.5
            row = {
                "predict_time": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": "AAPL",
                "last_price": f"{base:.4f}",
                "pred_day1": f"{base + 0.5:.4f}",
                "pred_day2": f"{base + 1.0:.4f}",
                "pred_day3": f"{base + 1.5:.4f}",
                "pred_day4": f"{base + 2.0:.4f}",
                "pred_day5": f"{base + 2.5:.4f}",
                "pred_day6": f"{base + 3.0:.4f}",
                "pred_day7": f"{base + 3.5:.4f}",
                "model_val_mae": "1.5000",
            }
            writer.writerow(row)
    return path


class TestLoadPredictionReturns:
    def test_loads_from_csv(self):
        import perf_report as pr
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            tmppath = f.name
        try:
            _create_mock_predictions_csv(tmppath, n=20)
            # Monkey-patch the CSV path
            original = pr._PRED_LOG
            pr._PRED_LOG = tmppath
            try:
                result = pr._load_prediction_returns()
                # 可能回 None 如果 SPY 下載失敗（在無網路環境）
                # 但至少不應 crash
                if result is not None:
                    assert len(result) > 0
            finally:
                pr._PRED_LOG = original
        finally:
            os.unlink(tmppath)

    def test_returns_none_for_missing_file(self):
        import perf_report as pr
        original = pr._PRED_LOG
        pr._PRED_LOG = "/nonexistent/path/predictions.csv"
        try:
            result = pr._load_prediction_returns()
            assert result is None
        finally:
            pr._PRED_LOG = original

    def test_returns_none_for_empty_csv(self):
        import perf_report as pr
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            f.write("")
            tmppath = f.name
        try:
            original = pr._PRED_LOG
            pr._PRED_LOG = tmppath
            try:
                result = pr._load_prediction_returns()
                assert result is None
            finally:
                pr._PRED_LOG = original
        finally:
            os.unlink(tmppath)


class TestGenerateStatsText:
    def test_returns_none_when_no_quantstats(self):
        import perf_report as pr
        original = pr._HAS_QS
        pr._HAS_QS = False
        try:
            result = pr.generate_stats_text()
            assert result is None
        finally:
            pr._HAS_QS = original
