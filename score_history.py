"""
小龍蝦 OpenClaw — Score 歷史持久化
每日存檔各股 score，提供 delta 與連續趨勢計算。
"""

import json
import logging
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

_log = logging.getLogger("openclaw.score_history")
_TW = timezone(timedelta(hours=8))
_HISTORY_PATH = Path("score_history.json")
_lock = threading.Lock()
_MAX_DAYS = 30


def _load_all() -> dict:
    with _lock:
        if _HISTORY_PATH.exists():
            try:
                return json.loads(_HISTORY_PATH.read_text("utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                _log.warning(f"score_history load failed: {e}")
        return {}


def _save_all(data: dict) -> None:
    with _lock:
        _HISTORY_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), "utf-8"
        )


def _cleanup_old(data: dict, max_days: int = _MAX_DAYS) -> dict:
    """移除超過 max_days 天前的資料。"""
    cutoff = (datetime.now(_TW) - timedelta(days=max_days)).strftime("%Y-%m-%d")
    keys_to_remove = [k for k in data if k < cutoff]
    for k in keys_to_remove:
        del data[k]
    return data


def save_score_snapshot(wl: list[dict], batch: list[dict], tag: str = "") -> None:
    """
    存今日 scores。wl 與 batch 對齊。
    tag 可區分同日多次存檔（如 "premarket"/"postclose"），空字串則直接覆寫。
    """
    today = datetime.now(_TW).strftime("%Y-%m-%d")
    key = f"{today}_{tag}" if tag else today

    snapshot = {}
    for item, br in zip(wl, batch):
        analysis = br.get("analysis")
        if analysis is None:
            continue
        ticker = item["ticker"]
        snapshot[ticker] = {
            "score": analysis["total_score"],
            "rsi": round(analysis.get("rsi", 0), 1),
            "change_pct": round(analysis.get("change_pct", 0), 2),
        }

    data = _load_all()
    data[key] = snapshot
    data = _cleanup_old(data)
    _save_all(data)
    _log.info(f"score_history: saved {len(snapshot)} tickers for {key}")


def load_score_history(days: int = 7) -> dict:
    """
    載入最近 N 天的 score 歷史。
    回傳原始 dict: {"2026-06-11": {"0050.TW": {...}, ...}, ...}
    包含帶 tag 和不帶 tag 的 key。
    """
    data = _load_all()
    cutoff = (datetime.now(_TW) - timedelta(days=days)).strftime("%Y-%m-%d")
    return {k: v for k, v in data.items() if k >= cutoff}


def get_score_delta(ticker: str, today_score: int, history: dict) -> tuple[str, int | None]:
    """
    計算 score delta 與連續趨勢。
    回傳 (display_text, delta_value)。
    display_text 例: "📈+7 (連升3日)" / "📉-12 (連跌2日)" / "➡️ 0" / "🆕"
    delta_value: int 或 None（無歷史時）
    """
    # 找到最近一天（不含今天）的 score
    today_str = datetime.now(_TW).strftime("%Y-%m-%d")

    # 收集所有日期 key（不含今天的任何 tag），按日期降序
    prev_keys = sorted(
        [k for k in history if not k.startswith(today_str)],
        reverse=True,
    )

    if not prev_keys:
        return "🆕", None

    # 取最近一天的 score
    prev_day_key = prev_keys[0]
    prev_entry = history[prev_day_key].get(ticker)
    if prev_entry is None:
        return "🆕", None

    prev_score = prev_entry["score"]
    delta = today_score - prev_score

    if delta == 0:
        return "➡️0", 0

    # 計算連續趨勢（往前看最多 7 天）
    streak = 1
    direction = 1 if delta > 0 else -1
    last_score = prev_score

    for key in prev_keys[1:7]:
        entry = history[key].get(ticker)
        if entry is None:
            break
        s = entry["score"]
        if direction > 0 and s < last_score:
            streak += 1
        elif direction < 0 and s > last_score:
            streak += 1
        else:
            break
        last_score = s

    if delta > 0:
        arrow = "📈"
        streak_text = f" (連升{streak}日)" if streak >= 2 else ""
    else:
        arrow = "📉"
        streak_text = f" (連跌{streak}日)" if streak >= 2 else ""

    return f"{arrow}{delta:+d}{streak_text}", delta
