"""
小龍蝦 OpenClaw — 市場情緒數據源
提供 CNN 恐懼貪婪指數、VIX 波動率、台股加權指數，內建 TTL 快取。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests
import yfinance as yf

_log = logging.getLogger("openclaw.market_data")

# ---------------------------------------------------------------------------
# TTL 快取
# ---------------------------------------------------------------------------
@dataclass
class _CacheEntry:
    data: Any
    ts: float  # time.monotonic()

_cache: dict[str, _CacheEntry] = {}

def _get_cached(key: str, ttl: float) -> Any | None:
    entry = _cache.get(key)
    if entry and (time.monotonic() - entry.ts) < ttl:
        return entry.data
    return None

def _set_cached(key: str, data: Any) -> Any:
    _cache[key] = _CacheEntry(data=data, ts=time.monotonic())
    return data

def clear_cache() -> None:
    _cache.clear()

# ---------------------------------------------------------------------------
# CNN 恐懼貪婪指數
# ---------------------------------------------------------------------------
_FG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
_FG_TTL = 1800  # 30 min

_FG_LABELS = {
    "extreme fear": ("😱", "極度恐懼"),
    "fear":         ("😰", "恐懼"),
    "neutral":      ("😐", "中性"),
    "greed":        ("😏", "貪婪"),
    "extreme greed":("🤑", "極度貪婪"),
}

_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

def fetch_fear_greed() -> dict | None:
    """取得 CNN Fear & Greed Index。回傳 dict 或 None (失敗時)。"""
    cached = _get_cached("fear_greed", _FG_TTL)
    if cached is not None:
        return cached

    try:
        headers = {"User-Agent": _BROWSER_UA, "Accept": "application/json"}
        resp = requests.get(_FG_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        fg = data.get("fear_and_greed", {})
        score = fg.get("score")
        rating = fg.get("rating", "").lower()
        if score is None:
            _log.warning("fetch_fear_greed: score missing from response")
            return None

        score = round(float(score), 1)
        emoji, label_zh = _FG_LABELS.get(rating, ("❓", rating))

        result = {
            "score": score,
            "rating": rating,
            "label": label_zh,
            "emoji": emoji,
            "text": f"{emoji} {score} ({label_zh})",
        }
        return _set_cached("fear_greed", result)

    except Exception as e:
        _log.warning(f"fetch_fear_greed failed: {e}")
        return None


# ---------------------------------------------------------------------------
# VIX 波動率
# ---------------------------------------------------------------------------
_VIX_TTL = 600  # 10 min

_VIX_LEVELS = [
    (35, "🔴", "extreme", "極高風險"),
    (25, "🟠", "high",    "高風險"),
    (18, "🟡", "moderate","中等風險"),
    (0,  "🟢", "low",     "低風險"),
]

def fetch_vix() -> dict | None:
    """取得 VIX 波動率指數。"""
    cached = _get_cached("vix", _VIX_TTL)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker("^VIX")
        hist = ticker.history(period="5d")
        if hist.empty:
            _log.warning("fetch_vix: no data")
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change_pct = (current - prev) / prev * 100 if prev != 0 else 0.0

        emoji, level, level_zh = "🟢", "low", "低風險"
        for threshold, e, lv, lv_zh in _VIX_LEVELS:
            if current >= threshold:
                emoji, level, level_zh = e, lv, lv_zh
                break

        result = {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2),
            "level": level,
            "level_zh": level_zh,
            "emoji": emoji,
            "text": f"{emoji} {current:.2f} ({change_pct:+.1f}%) [{level_zh}]",
        }
        return _set_cached("vix", result)

    except Exception as e:
        _log.warning(f"fetch_vix failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 台股加權指數
# ---------------------------------------------------------------------------
_TWII_TTL = 600  # 10 min

def fetch_twii() -> dict | None:
    """取得台股加權指數 (^TWII)。"""
    cached = _get_cached("twii", _TWII_TTL)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker("^TWII")
        hist = ticker.history(period="5d")
        if hist.empty:
            _log.warning("fetch_twii: no data")
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change_pct = (current - prev) / prev * 100 if prev != 0 else 0.0
        volume = int(hist["Volume"].iloc[-1])

        arrow = "🔺" if change_pct >= 0 else "🔻"

        result = {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2),
            "volume": volume,
            "arrow": arrow,
            "text": f"{arrow} {current:,.2f} ({change_pct:+.2f}%) 量 {volume:,}",
        }
        return _set_cached("twii", result)

    except Exception as e:
        _log.warning(f"fetch_twii failed: {e}")
        return None


# ---------------------------------------------------------------------------
# S&P 500 指數
# ---------------------------------------------------------------------------
_SP500_TTL = 600  # 10 min

def fetch_sp500() -> dict | None:
    """取得 S&P 500 指數 (^GSPC)。"""
    cached = _get_cached("sp500", _SP500_TTL)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker("^GSPC")
        hist = ticker.history(period="5d").dropna(subset=["Close"])
        if hist.empty:
            _log.warning("fetch_sp500: no data")
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change_pct = (current - prev) / prev * 100 if prev != 0 else 0.0
        arrow = "🔺" if change_pct >= 0 else "🔻"

        result = {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2),
            "arrow": arrow,
            "text": f"{arrow} {current:,.2f} ({change_pct:+.2f}%)",
        }
        return _set_cached("sp500", result)

    except Exception as e:
        _log.warning(f"fetch_sp500 failed: {e}")
        return None


# ---------------------------------------------------------------------------
# NASDAQ 綜合指數
# ---------------------------------------------------------------------------
_NASDAQ_TTL = 600  # 10 min

def fetch_nasdaq() -> dict | None:
    """取得 NASDAQ 綜合指數 (^IXIC)。"""
    cached = _get_cached("nasdaq", _NASDAQ_TTL)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker("^IXIC")
        hist = ticker.history(period="5d").dropna(subset=["Close"])
        if hist.empty:
            _log.warning("fetch_nasdaq: no data")
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change_pct = (current - prev) / prev * 100 if prev != 0 else 0.0
        arrow = "🔺" if change_pct >= 0 else "🔻"

        result = {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2),
            "arrow": arrow,
            "text": f"{arrow} {current:,.2f} ({change_pct:+.2f}%)",
        }
        return _set_cached("nasdaq", result)

    except Exception as e:
        _log.warning(f"fetch_nasdaq failed: {e}")
        return None


# ---------------------------------------------------------------------------
# 美股期貨 (ES=F / NQ=F)
# ---------------------------------------------------------------------------
_FUTURES_TTL = 600  # 10 min

def _fetch_single_future(symbol: str, cache_key: str) -> dict | None:
    """取得單一期貨商品。"""
    cached = _get_cached(cache_key, _FUTURES_TTL)
    if cached is not None:
        return cached

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d").dropna(subset=["Close"])
        if hist.empty:
            _log.warning(f"_fetch_single_future({symbol}): no data")
            return None

        current = float(hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else current
        change_pct = (current - prev) / prev * 100 if prev != 0 else 0.0
        arrow = "🔺" if change_pct >= 0 else "🔻"

        result = {
            "value": round(current, 2),
            "change_pct": round(change_pct, 2),
            "arrow": arrow,
            "text": f"{arrow} {current:,.2f} ({change_pct:+.2f}%)",
        }
        return _set_cached(cache_key, result)

    except Exception as e:
        _log.warning(f"_fetch_single_future({symbol}) failed: {e}")
        return None


def fetch_us_futures() -> dict:
    """取得 S&P 500 期貨 (ES=F) + NASDAQ 期貨 (NQ=F)。"""
    return {
        "sp500_futures": _fetch_single_future("ES=F", "es_futures"),
        "nasdaq_futures": _fetch_single_future("NQ=F", "nq_futures"),
    }


# ---------------------------------------------------------------------------
# 一次取得全部市場情緒指標
# ---------------------------------------------------------------------------
def fetch_market_sentiment() -> dict:
    """取得全部市場情緒指標。各欄位可能為 None（個別失敗時）。"""
    return {
        "fear_greed": fetch_fear_greed(),
        "vix": fetch_vix(),
        "twii": fetch_twii(),
        "sp500": fetch_sp500(),
        "nasdaq": fetch_nasdaq(),
        "us_futures": fetch_us_futures(),
    }


def format_sentiment_block(sentiment: dict) -> str:
    """將情緒指標格式化為 Discord embed 用的文字段落。"""
    lines = []

    fg = sentiment.get("fear_greed")
    if fg:
        lines.append(f"**恐懼貪婪指數** {fg['text']}")

    vix = sentiment.get("vix")
    if vix:
        lines.append(f"**VIX 波動率** {vix['text']}")

    twii = sentiment.get("twii")
    if twii:
        lines.append(f"**台股加權** {twii['text']}")

    sp500 = sentiment.get("sp500")
    if sp500:
        lines.append(f"**S&P 500** {sp500['text']}")

    nasdaq = sentiment.get("nasdaq")
    if nasdaq:
        lines.append(f"**NASDAQ** {nasdaq['text']}")

    # 美股期貨
    us_fut = sentiment.get("us_futures")
    if us_fut:
        sp_f = us_fut.get("sp500_futures")
        nq_f = us_fut.get("nasdaq_futures")
        if sp_f or nq_f:
            lines.append("── 美股期貨 ──")
            if sp_f:
                lines.append(f"**ES (S&P 500期)** {sp_f['text']}")
            if nq_f:
                lines.append(f"**NQ (NASDAQ期)** {nq_f['text']}")

    return "\n".join(lines) if lines else "⚠️ 市場數據暫時無法取得"
