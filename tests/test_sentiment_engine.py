"""
Tests for sentiment_engine.py
不實際載入 FinBERT 模型，只測試邏輯和 fallback。
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScoreSentiment:
    def test_positive_sentiment(self):
        from sentiment_engine import _score_sentiment
        results = [
            {"label": "positive", "score": 0.85},
            {"label": "negative", "score": 0.05},
            {"label": "neutral", "score": 0.10},
        ]
        score, label = _score_sentiment(results)
        assert score == pytest.approx(0.80, abs=0.01)
        assert label == "正面"

    def test_negative_sentiment(self):
        from sentiment_engine import _score_sentiment
        results = [
            {"label": "positive", "score": 0.05},
            {"label": "negative", "score": 0.90},
            {"label": "neutral", "score": 0.05},
        ]
        score, label = _score_sentiment(results)
        assert score == pytest.approx(-0.85, abs=0.01)
        assert label == "負面"

    def test_neutral_sentiment(self):
        from sentiment_engine import _score_sentiment
        results = [
            {"label": "positive", "score": 0.35},
            {"label": "negative", "score": 0.25},
            {"label": "neutral", "score": 0.40},
        ]
        score, label = _score_sentiment(results)
        assert score == pytest.approx(0.10, abs=0.01)
        assert label == "中性"

    def test_score_range(self):
        """分數範圍應在 [-1, 1]。"""
        from sentiment_engine import _score_sentiment
        # 極端正面
        s1, _ = _score_sentiment([
            {"label": "positive", "score": 1.0},
            {"label": "negative", "score": 0.0},
            {"label": "neutral", "score": 0.0},
        ])
        assert s1 == 1.0

        # 極端負面
        s2, _ = _score_sentiment([
            {"label": "positive", "score": 0.0},
            {"label": "negative", "score": 1.0},
            {"label": "neutral", "score": 0.0},
        ])
        assert s2 == -1.0


class TestAnalyzeSentiment:
    def test_returns_unavailable_when_model_not_loaded(self):
        """FinBERT 沒載入時應回 available=False。"""
        import sentiment_engine as se
        # 強制重設
        original_pipeline = se._pipeline
        original_attempted = se._load_attempted
        try:
            se._pipeline = None
            se._load_attempted = True  # 假裝已嘗試過但失敗
            result = se.analyze_sentiment("AAPL")
            assert result["available"] is False
        finally:
            se._pipeline = original_pipeline
            se._load_attempted = original_attempted
