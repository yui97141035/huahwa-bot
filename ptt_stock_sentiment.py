"""
PTT 股板情緒分析模組 — 爬取 PTT Stock 板並分析指定股票的社群情緒。

複用 wp_scraper.py 的 PTTScraper 爬取 PTT Stock 板，
用 FinBERT (sentiment_engine.py) 分析相關文章情緒。
結果快取 2 小時避免頻繁爬取。
"""

import re
import time as _time
import logging

_log = logging.getLogger("openclaw.ptt_sentiment")

# ---------------------------------------------------------------------------
# 快取
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 2 * 3600  # 2 hours


def _cache_get(ticker: str) -> dict | None:
    entry = _cache.get(ticker.upper())
    if entry and (_time.time() - entry[0]) < _CACHE_TTL:
        _log.info(f"ptt_sentiment({ticker}): cache hit")
        return entry[1]
    return None


def _cache_set(ticker: str, data: dict) -> None:
    _cache[ticker.upper()] = (_time.time(), data)


# ---------------------------------------------------------------------------
# 台股代碼 → 搜尋關鍵字
# ---------------------------------------------------------------------------
_TW_STOCK_NAMES: dict[str, list[str]] = {
    "2330": ["台積電", "台積", "TSMC", "GDR"],
    "2317": ["鴻海", "Foxconn"],
    "2454": ["聯發科", "MTK"],
    "2308": ["台達電", "台達"],
    "2881": ["富邦金"],
    "2882": ["國泰金"],
    "2886": ["兆豐金"],
    "2891": ["中信金"],
    "2884": ["玉山金"],
    "2303": ["聯電", "UMC"],
    "3711": ["日月光", "ASE"],
    "2412": ["中華電"],
    "2382": ["廣達"],
    "3231": ["緯創"],
    "2345": ["智邦"],
    "2357": ["華碩", "ASUS"],
    "6505": ["台塑化"],
    "1301": ["台塑"],
    "1303": ["南亞"],
    "2002": ["中鋼"],
    "3034": ["聯詠", "Novatek"],
    "2379": ["瑞昱", "Realtek"],
    "3443": ["創意"],
    "5274": ["信驊"],
    "3661": ["世芯"],
    "6669": ["緯穎"],
    "2603": ["長榮"],
    "2615": ["萬海"],
    "2609": ["陽明"],
}


def _get_search_keywords(ticker: str) -> list[str]:
    """從 ticker 產生 PTT 搜尋關鍵字列表。"""
    code = ticker.split(".")[0]
    keywords = [code]
    if code in _TW_STOCK_NAMES:
        keywords.extend(_TW_STOCK_NAMES[code])
    return keywords


# ---------------------------------------------------------------------------
# 核心函式
# ---------------------------------------------------------------------------
def fetch_ptt_stock_sentiment(ticker: str) -> dict:
    """爬取 PTT Stock 板並分析指定股票的情緒。

    Returns:
        {
            "available": bool,
            "score": float (-1.0 ~ +1.0),
            "posts_analyzed": int,
            "bullish_ratio": float (0.0 ~ 1.0),
            "headlines": [{"title": str, "label": str, "score": float}, ...]
        }
    或 {"available": False} 如果爬取失敗或無相關文章。
    """
    # 快取
    cached = _cache_get(ticker)
    if cached is not None:
        return cached

    result = _do_fetch(ticker)
    _cache_set(ticker, result)
    return result


def _do_fetch(ticker: str) -> dict:
    """實際執行 PTT 爬取 + 情緒分析。"""
    keywords = _get_search_keywords(ticker)
    _log.info(f"ptt_sentiment({ticker}): keywords={keywords}")

    # 爬取 PTT Stock 板
    try:
        from wp_scraper import PTTScraper
        scraper = PTTScraper(min_push=0)  # 不過濾推文數
        articles = scraper.scrape_board("Stock", pages=3)
    except Exception as e:
        _log.warning(f"ptt_sentiment({ticker}): scrape failed ({e})")
        return {"available": False}

    if not articles:
        _log.info(f"ptt_sentiment({ticker}): no articles from PTT Stock")
        return {"available": False}

    # 過濾與 ticker 相關的文章
    relevant = []
    for art in articles:
        title = art.get("title", "")
        if _matches_keywords(title, keywords):
            relevant.append(art)

    if not relevant:
        _log.info(f"ptt_sentiment({ticker}): no relevant articles found (total={len(articles)})")
        return {"available": False, "posts_analyzed": 0}

    _log.info(f"ptt_sentiment({ticker}): {len(relevant)} relevant articles out of {len(articles)}")

    # FinBERT 情緒分析
    try:
        import sentiment_engine as _se
    except ImportError:
        _log.warning("ptt_sentiment: sentiment_engine not available")
        return {"available": False}

    if not _se._ensure_model():
        _log.warning("ptt_sentiment: FinBERT model not loaded")
        return {"available": False}

    headlines_data = []
    total_score = 0.0

    for art in relevant[:20]:  # 最多分析 20 篇
        title = art["title"]
        try:
            raw = _se._pipeline([title])
            score, label = _se._score_sentiment(raw[0])
            headlines_data.append({
                "title": title,
                "score": round(score, 3),
                "label": label,
            })
            total_score += score
        except Exception as e:
            _log.debug(f"ptt_sentiment: FinBERT failed for '{title}': {e}")
            continue

    if not headlines_data:
        return {"available": False, "posts_analyzed": 0}

    avg_score = total_score / len(headlines_data)
    bullish_count = sum(1 for h in headlines_data if h["score"] > 0.15)
    bullish_ratio = bullish_count / len(headlines_data)

    # 按分數排序
    sorted_headlines = sorted(headlines_data, key=lambda x: abs(x["score"]), reverse=True)

    result = {
        "available": True,
        "score": round(avg_score, 3),
        "posts_analyzed": len(headlines_data),
        "bullish_ratio": round(bullish_ratio, 3),
        "headlines": sorted_headlines[:5],
    }
    _log.info(f"ptt_sentiment({ticker}): score={avg_score:.3f}, posts={len(headlines_data)}, bullish={bullish_ratio:.1%}")
    return result


def _matches_keywords(title: str, keywords: list[str]) -> bool:
    """檢查文章標題是否包含任一關鍵字。"""
    for kw in keywords:
        if kw in title:
            return True
    return False
