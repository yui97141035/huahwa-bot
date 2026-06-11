"""
Gate 4: 多代理共識門檻 — TradingAgents 牛熊辯論 + 風險評估

呼叫 TradingAgents 框架進行多代理辯論（牛熊研究員、風險辯論、投資組合經理），
解析最終 rating 產生 GateResult。

Rating mapping:
  Buy        -> passed=True,  score=1.0
  Overweight -> passed=True,  score=0.8
  Hold       -> passed=True,  score=0.5  (中性通過)
  Underweight -> passed=False, score=0.3
  Sell       -> passed=False, score=0.1

使用 ProcessPoolExecutor + timeout 避免阻塞。
結果快取 4 小時 TTL。
"""

import os
import re
import time as _time
import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone, timedelta

from signal_gate import GateResult

_log = logging.getLogger("openclaw.agent_gate")

# ---------------------------------------------------------------------------
# 快取
# ---------------------------------------------------------------------------
_agent_cache: dict[str, tuple[float, GateResult, str]] = {}
# key=ticker, value=(timestamp, gate_result, executive_summary)

_TW = timezone(timedelta(hours=8))


def _get_cache_ttl() -> int:
    from prediction_config import AGENT_CACHE_TTL
    return AGENT_CACHE_TTL


def _get_timeout() -> int:
    from prediction_config import AGENT_GATE_TIMEOUT
    return AGENT_GATE_TIMEOUT


def _cache_get(ticker: str) -> tuple[GateResult, str] | None:
    entry = _agent_cache.get(ticker.upper())
    if entry and (_time.time() - entry[0]) < _get_cache_ttl():
        _log.info(f"agent_gate({ticker}): cache hit")
        return entry[1], entry[2]
    return None


def _cache_set(ticker: str, result: GateResult, summary: str) -> None:
    _agent_cache[ticker.upper()] = (_time.time(), result, summary)


# ---------------------------------------------------------------------------
# Rating -> GateResult 映射
# ---------------------------------------------------------------------------
_RATING_MAP = {
    "buy":        (True,  1.0),
    "overweight": (True,  0.8),
    "hold":       (True,  0.5),
    "underweight":(False, 0.3),
    "sell":       (False, 0.1),
}


def _parse_rating(decision_text: str) -> str:
    """從 final_trade_decision 文字提取 rating。"""
    # 嘗試匹配 **Rating**: Buy 或 Rating: Overweight 等格式
    m = re.search(r"\*{0,2}Rating\*{0,2}\s*[:：]\s*\*{0,2}(\w+)\*{0,2}", decision_text, re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: 直接搜尋 rating 關鍵字
    text_lower = decision_text.lower()
    for rating in ["buy", "overweight", "hold", "underweight", "sell"]:
        if rating in text_lower:
            return rating.capitalize()

    return "Hold"  # 無法解析時預設 Hold


def _parse_summary(decision_text: str) -> str:
    """從 final_trade_decision 提取 executive_summary。"""
    m = re.search(
        r"\*{0,2}Executive Summary\*{0,2}\s*[:：]\s*(.+?)(?=\n\*{0,2}[A-Z]|\Z)",
        decision_text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        return m.group(1).strip()[:500]
    # Fallback: 取前 300 字
    return decision_text[:300].strip()


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------
def _run_agent_analysis(ticker: str, trade_date: str, config: dict) -> tuple[str, str]:
    """在子程序中執行 TradingAgents 分析。

    Returns: (final_trade_decision_text, signal_str)
    """
    # 確保子程序中也有 GOOGLE_API_KEY
    gemini_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        os.environ["GOOGLE_API_KEY"] = gemini_key

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    ta = TradingAgentsGraph(config=config)
    final_state, signal = ta.propagate(ticker, trade_date)

    decision_text = final_state.get("final_trade_decision", "")
    return decision_text, signal


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------
def gate_agent_consensus(analysis: dict, ticker: str,
                         trade_date: str | None = None) -> GateResult:
    """Gate 4: 多代理共識。

    呼叫 TradingAgents 進行牛熊辯論+風險評估，
    解析最終 rating 產生 GateResult。
    """
    from prediction_config import ENABLE_AGENT_GATE

    if not ENABLE_AGENT_GATE:
        return GateResult(passed=True, score=0.5, details="agent gate disabled (auto-pass)")

    # 快取
    cached = _cache_get(ticker)
    if cached is not None:
        return cached[0]

    if trade_date is None:
        trade_date = datetime.now(_TW).strftime("%Y-%m-%d")

    # TradingAgents config — 基於 DEFAULT_CONFIG 覆寫
    from prediction_config import AGENT_LLM_PROVIDER, AGENT_LLM_MODEL
    from tradingagents.default_config import DEFAULT_CONFIG
    config = dict(DEFAULT_CONFIG)
    config.update({
        "llm_provider": AGENT_LLM_PROVIDER,
        "deep_think_llm": AGENT_LLM_MODEL,
        "quick_think_llm": AGENT_LLM_MODEL,
        "output_language": "Chinese",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "global_news_queries": [
            "台灣央行利率 通膨 貨幣政策",
            "台股加權指數 外資動向 法人買賣超",
            "半導體產業 台積電 AI晶片",
            "兩岸關係 地緣政治風險",
            "美國聯準會 利率決策 美股走勢",
        ],
    })

    timeout = _get_timeout()
    _log.info(f"agent_gate({ticker}): starting TradingAgents analysis (timeout={timeout}s)")

    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_agent_analysis, ticker, trade_date, config)
            decision_text, signal = future.result(timeout=timeout)

        rating = _parse_rating(decision_text)
        summary = _parse_summary(decision_text)
        rating_lower = rating.lower()

        passed, score = _RATING_MAP.get(rating_lower, (True, 0.5))

        gate_result = GateResult(
            passed=passed,
            score=score,
            details=f"agent={rating}, signal={signal}",
        )

        _cache_set(ticker, gate_result, summary)
        _log.info(f"agent_gate({ticker}): rating={rating}, passed={passed}, score={score}")
        return gate_result

    except FuturesTimeoutError:
        _log.warning(f"agent_gate({ticker}): timeout after {timeout}s (auto-pass)")
        result = GateResult(passed=True, score=0.5, details=f"agent timeout {timeout}s (auto-pass)")
        _cache_set(ticker, result, "")
        return result

    except Exception as e:
        _log.error(f"agent_gate({ticker}): error ({e}) (auto-pass)")
        result = GateResult(passed=True, score=0.5, details=f"agent error: {e} (auto-pass)")
        _cache_set(ticker, result, "")
        return result


def get_agent_summary(ticker: str) -> str:
    """取得快取中的 agent executive summary。"""
    entry = _agent_cache.get(ticker.upper())
    if entry:
        return entry[2]
    return ""


def get_agent_rating(ticker: str) -> str | None:
    """取得快取中的 agent rating。"""
    entry = _agent_cache.get(ticker.upper())
    if entry:
        gr = entry[1]
        # 從 details 提取 rating
        m = re.search(r"agent=(\w+)", gr.details)
        if m:
            return m.group(1)
    return None
