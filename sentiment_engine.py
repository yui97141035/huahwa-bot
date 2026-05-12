"""
OpenClaw Sentiment Engine — FinBERT 新聞情緒分析
使用 ProsusAI/finbert 模型分析 yfinance 新聞標題情緒。
模型 lazy-load（首次呼叫 ~5s 載入，之後 ~300ms/batch）。
import 或推理失敗 → 回傳 {"available": False}。
"""

import logging

_log = logging.getLogger("openclaw.sentiment")

# ---------------------------------------------------------------------------
# Lazy-loaded model globals
# ---------------------------------------------------------------------------
_tokenizer = None
_model = None
_pipeline = None
_load_attempted = False


def _ensure_model():
    """Lazy-load FinBERT model. Only attempts loading once."""
    global _tokenizer, _model, _pipeline, _load_attempted

    if _pipeline is not None:
        return True
    if _load_attempted:
        return False

    _load_attempted = True
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

        model_name = "ProsusAI/finbert"
        _log.info(f"sentiment_engine: loading {model_name}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_name)
        _model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _pipeline = pipeline(
            "sentiment-analysis",
            model=_model,
            tokenizer=_tokenizer,
            top_k=None,  # return all class probabilities
        )
        _log.info("sentiment_engine: FinBERT loaded successfully")
        return True
    except Exception as e:
        _log.warning(f"sentiment_engine: failed to load FinBERT ({e})")
        return False


def _get_news_headlines(ticker: str, max_headlines: int = 10) -> list[str]:
    """從 yfinance 取得個股新聞標題。"""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news
        if not news:
            return []

        headlines = []
        for item in news[:max_headlines]:
            title = item.get("title", "")
            if title:
                headlines.append(title)
        return headlines
    except Exception as e:
        _log.warning(f"sentiment_engine: failed to fetch news for {ticker} ({e})")
        return []


def _score_sentiment(results: list[dict]) -> tuple[float, str]:
    """
    從 FinBERT pipeline 輸出計算情緒分數。
    results: list of [{"label": "positive", "score": 0.9}, ...]
    回傳: (score: -1.0~+1.0, label: "正面"/"負面"/"中性")
    """
    pos_prob = 0.0
    neg_prob = 0.0
    neu_prob = 0.0
    for item in results:
        lbl = item["label"].lower()
        if lbl == "positive":
            pos_prob = item["score"]
        elif lbl == "negative":
            neg_prob = item["score"]
        elif lbl == "neutral":
            neu_prob = item["score"]

    score = pos_prob - neg_prob  # -1.0 ~ +1.0

    if score > 0.15:
        label = "正面"
    elif score < -0.15:
        label = "負面"
    else:
        label = "中性"

    return score, label


def analyze_sentiment(ticker: str) -> dict:
    """
    分析個股新聞情緒。
    回傳:
      {
        "available": True,
        "score": float (-1.0 ~ +1.0),
        "label": str ("正面"/"負面"/"中性"),
        "headline_count": int,
        "headlines": [
          {"title": str, "score": float, "label": str},
          ...
        ]
      }
    或 {"available": False} 如果失敗。
    """
    from prediction_config import SENTIMENT_MAX_HEADLINES, SENTIMENT_DISPLAY_TOP

    if not _ensure_model():
        return {"available": False}

    headlines = _get_news_headlines(ticker, max_headlines=SENTIMENT_MAX_HEADLINES)
    if not headlines:
        _log.info(f"sentiment_engine({ticker}): no news headlines found")
        return {"available": False}

    try:
        # FinBERT batch inference
        raw_results = _pipeline(headlines)

        scored_headlines = []
        total_score = 0.0

        for title, result in zip(headlines, raw_results):
            score, label = _score_sentiment(result)
            scored_headlines.append({
                "title": title,
                "score": round(score, 3),
                "label": label,
            })
            total_score += score

        avg_score = total_score / len(scored_headlines) if scored_headlines else 0.0

        if avg_score > 0.15:
            overall_label = "正面"
        elif avg_score < -0.15:
            overall_label = "負面"
        else:
            overall_label = "中性"

        # 按分數絕對值排序，取 top N 顯示
        sorted_headlines = sorted(scored_headlines, key=lambda x: abs(x["score"]), reverse=True)

        return {
            "available": True,
            "score": round(avg_score, 3),
            "label": overall_label,
            "headline_count": len(scored_headlines),
            "headlines": sorted_headlines[:SENTIMENT_DISPLAY_TOP],
        }

    except Exception as e:
        _log.warning(f"sentiment_engine({ticker}): inference failed ({e})")
        return {"available": False}
