"""
花城 — 股票監控清單 + 自動進場提醒
"""

import json
import logging
import threading
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import discord

log = logging.getLogger("huacheng.watchlist")

# ---------------------------------------------------------------------------
# 初始監控清單（僅首次啟動時使用，之後全部存在 watchlist.json）
# ---------------------------------------------------------------------------
_INITIAL_WATCHLIST: list[dict] = [
    {"input": "0050",   "ticker": "0050.TW",  "name": "元大台灣50"},
    {"input": "0056",   "ticker": "0056.TW",  "name": "元大高股息"},
    {"input": "00878",  "ticker": "00878.TW", "name": "國泰永續高股息"},
    {"input": "00662",  "ticker": "00662.TW", "name": "富邦NASDAQ"},
    {"input": "006208", "ticker": "006208.TW","name": "富邦台50"},
    {"input": "00929",  "ticker": "00929.TW", "name": "復華台灣科技優息"},
    {"input": "RGTI",   "ticker": "RGTI",     "name": "Rigetti Computing"},
    {"input": "IONQ",   "ticker": "IONQ",     "name": "IonQ"},
    {"input": "2467",   "ticker": "2467.TW",  "name": "志聖工業"},
    {"input": "5443",   "ticker": "5443.TWO", "name": "均豪精密"},
    {"input": "6640",   "ticker": "6640.TWO", "name": "均華精密"},
]

_STATE_PATH = Path("watchlist.json")
_COOLDOWN = timedelta(hours=24)
_TW = timezone(timedelta(hours=8))
_US_ET = ZoneInfo("America/New_York")
_state_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 持久化狀態（thread-safe）
# ---------------------------------------------------------------------------
def _load_state() -> dict:
    with _state_lock:
        if _STATE_PATH.exists():
            try:
                state = json.loads(_STATE_PATH.read_text("utf-8"))
            except json.JSONDecodeError:
                log.warning("watchlist.json JSON 格式損壞，使用預設值")
                state = None
            except OSError as e:
                # EMFILE 等系統錯誤：不覆蓋檔案，直接 raise 讓呼叫者重試
                log.error(f"watchlist.json 無法讀取 (系統錯誤): {e}")
                raise
        else:
            state = None

        if state is None:
            state = {"enabled": False, "channel_id": None, "chat_channel_id": None, "cooldowns": {},
                     "watchlist": list(_INITIAL_WATCHLIST)}

        # 自動補齊：確保 _INITIAL_WATCHLIST 裡的股票都在 watchlist 中
        if "watchlist" in state:
            existing_tickers = {w["ticker"] for w in state["watchlist"]}
            added = []
            for item in _INITIAL_WATCHLIST:
                if item["ticker"] not in existing_tickers:
                    state["watchlist"].append(dict(item))
                    existing_tickers.add(item["ticker"])
                    added.append(item["ticker"])
            if added:
                try:
                    _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
                    log.info(f"watchlist 自動補齊 {len(added)} 檔: {', '.join(added)}")
                except OSError as e:
                    log.error(f"watchlist 自動補齊寫入失敗: {e}")

        # 遷移：將舊的 DEFAULT + custom_stocks 合併為統一 watchlist
        if "watchlist" not in state:
            existing_tickers = set()
            merged = []
            for item in _INITIAL_WATCHLIST:
                if item["ticker"] not in existing_tickers:
                    merged.append(item)
                    existing_tickers.add(item["ticker"])
            for item in state.pop("custom_stocks", []):
                if item["ticker"] not in existing_tickers:
                    merged.append(item)
                    existing_tickers.add(item["ticker"])
            state["watchlist"] = merged
            try:
                _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")
                log.info(f"watchlist 已遷移：{len(merged)} 檔統一清單")
            except OSError as e:
                log.error(f"watchlist 遷移寫入失敗: {e}")

        return state


def _save_state(state: dict) -> None:
    with _state_lock:
        _STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------------------
# 啟用 / 停用
# ---------------------------------------------------------------------------
def set_monitor(enabled: bool, channel_id: int | None = None) -> dict:
    state = _load_state()
    state["enabled"] = enabled
    if channel_id is not None:
        state["channel_id"] = channel_id
    _save_state(state)
    return state


def get_state() -> dict:
    return _load_state()


def set_chat_channel(channel_id: int) -> None:
    state = _load_state()
    state["chat_channel_id"] = channel_id
    _save_state(state)


def get_chat_channel() -> int | None:
    return _load_state().get("chat_channel_id")


# ---------------------------------------------------------------------------
# 持久化設定：AI Prompt / 問安模板 / 問安 Prompts
# ---------------------------------------------------------------------------
def save_prompt(prompt: str) -> None:
    """將 AI prompt 存入 watchlist.json。"""
    state = _load_state()
    state["ai_prompt"] = prompt
    _save_state(state)
    log.info("save_prompt: AI prompt 已儲存")


def load_prompt() -> str | None:
    """從 watchlist.json 讀取已儲存的 AI prompt，無則回傳 None。"""
    return _load_state().get("ai_prompt")


def save_greeting_template(template: str) -> None:
    """將問安模板存入 watchlist.json。"""
    state = _load_state()
    state["greeting_template"] = template
    _save_state(state)
    log.info("save_greeting_template: 問安模板已儲存")


def load_greeting_template() -> str | None:
    """從 watchlist.json 讀取已儲存的問安模板，無則回傳 None。"""
    return _load_state().get("greeting_template")


def save_greeting_prompts(prompts: dict) -> None:
    """將各時段問安 prompt 存入 watchlist.json。key 為小時 (int)，value 為 prompt 文字。"""
    state = _load_state()
    # JSON key 必須是 str
    state["greeting_prompts"] = {str(k): v for k, v in prompts.items()}
    _save_state(state)
    log.info(f"save_greeting_prompts: 已儲存 {list(prompts.keys())}")


def load_greeting_prompts() -> dict | None:
    """從 watchlist.json 讀取已儲存的問安 prompts，回傳 {int_hour: str} 或 None。"""
    raw = _load_state().get("greeting_prompts")
    if not raw:
        return None
    return {int(k): v for k, v in raw.items()}


# ---------------------------------------------------------------------------
# 動態監控清單
# ---------------------------------------------------------------------------
def add_stock(input_code: str, ticker: str, name: str) -> bool:
    """將股票加入監控清單。回傳 True 表示新增成功，False 表示已存在。"""
    state = _load_state()
    watchlist = state.get("watchlist", [])
    for item in watchlist:
        if item["ticker"] == ticker:
            return False
    watchlist.append({"input": input_code, "ticker": ticker, "name": name})
    state["watchlist"] = watchlist
    _save_state(state)
    log.info(f"add_stock: 新增 {ticker} ({name})")
    return True


def remove_stock(ticker: str) -> bool:
    """從監控清單移除股票。回傳 True 表示移除成功，False 表示找不到。"""
    state = _load_state()
    watchlist = state.get("watchlist", [])
    new_watchlist = [item for item in watchlist if item["ticker"] != ticker]
    if len(new_watchlist) == len(watchlist):
        return False
    state["watchlist"] = new_watchlist
    _save_state(state)
    log.info(f"remove_stock: 移除 {ticker}")
    return True


def get_watchlist() -> list[dict]:
    """回傳監控清單。"""
    return _load_state().get("watchlist", [])


# ---------------------------------------------------------------------------
# 冷卻判斷
# ---------------------------------------------------------------------------
def should_alert(ticker: str, score: int, *, entry_signal=None) -> bool:
    """進場提醒判斷 + 24 小時冷卻。

    entry_signal: signal_gate.EntrySignal（三驗證模式）
    當 ENABLE_TRIPLE_GATE=True 且提供 entry_signal 時，只有 HIGH 信心度才發送。
    否則回退到舊邏輯 score >= 60。
    """
    from prediction_config import ENABLE_TRIPLE_GATE

    # 三驗證模式
    if ENABLE_TRIPLE_GATE and entry_signal is not None:
        if not entry_signal.should_alert:
            return False
    else:
        # 舊邏輯：score >= 60
        if score < 60:
            return False

    state = _load_state()
    cooldowns = state.get("cooldowns", {})
    record = cooldowns.get(ticker)
    if record is None:
        return True
    last_time = datetime.fromisoformat(record["time"])
    last_level = record.get("level", "BUY")
    now = datetime.now(timezone.utc)
    # 升級到 STRONG BUY：忽略冷卻
    if score >= 75 and last_level != "STRONG_BUY":
        return True
    # 24 小時內已提醒
    if now - last_time < _COOLDOWN:
        return False
    return True


def record_alert(ticker: str, score: int) -> None:
    state = _load_state()
    cooldowns = state.setdefault("cooldowns", {})
    level = "STRONG_BUY" if score >= 75 else "BUY"
    cooldowns[ticker] = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "score": score,
    }
    _save_state(state)


# ---------------------------------------------------------------------------
# 115 年 (2026) 台灣證券交易所休市日
# 來源: https://www.twse.com.tw/zh/trading/holiday.html
# ---------------------------------------------------------------------------
TWSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),       # 開國紀念日
    date(2026, 2, 12),      # 春節前結算（無交易）
    date(2026, 2, 13),      # 春節前結算（無交易）
    date(2026, 2, 16),      # 春節
    date(2026, 2, 17),      # 春節
    date(2026, 2, 18),      # 春節
    date(2026, 2, 19),      # 春節
    date(2026, 2, 20),      # 春節補假
    date(2026, 2, 27),      # 和平紀念日補假
    date(2026, 4, 3),       # 兒童節／清明節連假
    date(2026, 4, 6),       # 清明節補假
    date(2026, 5, 1),       # 勞動節
    date(2026, 6, 19),      # 端午節
    date(2026, 9, 25),      # 中秋節
    date(2026, 9, 28),      # 教師節
    date(2026, 10, 9),      # 國慶日補假
    date(2026, 12, 25),     # 行憲紀念日
}


# ---------------------------------------------------------------------------
# 2026 年 NYSE / NASDAQ 休市日
# 來源: https://www.nyse.com/markets/hours-calendars
# ---------------------------------------------------------------------------
NYSE_HOLIDAYS_2026 = {
    date(2026, 1, 1),       # New Year's Day
    date(2026, 1, 19),      # Martin Luther King Jr. Day
    date(2026, 2, 16),      # Presidents' Day
    date(2026, 4, 3),       # Good Friday
    date(2026, 5, 25),      # Memorial Day
    date(2026, 6, 19),      # Juneteenth
    date(2026, 7, 3),       # Independence Day (observed)
    date(2026, 9, 7),       # Labor Day
    date(2026, 11, 26),     # Thanksgiving
    date(2026, 12, 25),     # Christmas
}


# ---------------------------------------------------------------------------
# 市場時段（台股 / 美股分開判斷）
# ---------------------------------------------------------------------------
def is_tw_market_active() -> bool:
    """台股開市：週一~週五、非 TWSE 休市日、09:00~13:30 台灣時間。"""
    now_tw = datetime.now(_TW)
    if now_tw.weekday() >= 5:
        return False
    if now_tw.date() in TWSE_HOLIDAYS_2026:
        return False
    return 9 <= now_tw.hour < 14


def is_us_market_active() -> bool:
    """美股開市：週一~週五、非 NYSE 休市日、09:00~16:00 美東時間（含盤前 09:00 掃描）。"""
    now_et = datetime.now(_US_ET)
    if now_et.weekday() >= 5:
        return False
    if now_et.date() in NYSE_HOLIDAYS_2026:
        return False
    return 9 <= now_et.hour < 16


def is_stock_market_active(ticker: str) -> bool:
    """根據股票代碼判斷對應市場是否開市。.TW/.TWO 結尾 → 台股，其餘 → 美股。"""
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return is_tw_market_active()
    return is_us_market_active()


def is_market_active() -> bool:
    """任一市場活躍時回傳 True（用於全域排程判斷）。"""
    return is_tw_market_active() or is_us_market_active()


# ---------------------------------------------------------------------------
# Discord Embed 建構
# ---------------------------------------------------------------------------
def build_alert_embed(ticker: str, name: str, analysis: dict,
                      entry_signal=None) -> discord.Embed:
    score = analysis["total_score"]
    verdict = analysis["verdict"]

    if entry_signal and entry_signal.confidence == "HIGH":
        title = f"🔥 [高信心度] 進場提醒 — {ticker}"
        color = 0xFF4500
    elif score >= 75:
        title = f"🔥 強烈買入提醒 — {ticker}"
        color = 0xFF4500
    else:
        title = f"📢 進場提醒 — {ticker}"
        color = 0x2ECC71

    embed = discord.Embed(
        title=title,
        description=(
            f"**{name}**\n"
            f"## 評分: {score}/100  {verdict}"
        ),
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    embed.add_field(
        name="📊 關鍵數據",
        value=(
            f"收盤 **${analysis['current']:,.2f}** ({analysis['change_pct']:+.2f}%)\n"
            f"RSI {analysis['rsi']:.1f} | MACD {analysis['macd']:.3f} | 量比 {analysis['vol_ratio']:.1f}x"
        ),
        inline=False,
    )

    scores = analysis["scores"]
    lines = []
    for key, label in [("trend", "趨勢"), ("momentum", "動能"), ("macd", "MACD"),
                        ("bollinger", "布林"), ("volume", "量能"), ("reversal", "反轉")]:
        s = scores[key]
        lines.append(f"**{label}** {s['score']}/{s['max']}　{s['detail']}")
    embed.add_field(name="📋 六維評分", value="\n".join(lines), inline=False)

    embed.add_field(
        name="💰 加碼判斷",
        value=analysis["add_position_msg"],
        inline=False,
    )

    # 進階技術分析摘要（若有）
    ta_overlay = analysis.get("ta_overlay", {})
    ta_conf = analysis.get("ta_confidence", 0)
    ta_conf_max = analysis.get("ta_confidence_max", 40)
    if ta_overlay and ta_conf > 0:
        ta_parts = []
        kline = ta_overlay.get("kline", {})
        if kline.get("strength", 0) > 0:
            ta_parts.append(f"K線: {kline.get('detail', '')}")
        ma_cross = ta_overlay.get("ma_cross", {})
        if ma_cross.get("strength", 0) > 0:
            ta_parts.append(f"均線: {ma_cross.get('detail', '')}")
        if ta_parts:
            embed.add_field(
                name=f"🔬 TA 信心 {ta_conf}/{ta_conf_max}",
                value="\n".join(ta_parts),
                inline=False,
            )

    # 四驗證 Gate 狀態（若有）
    if entry_signal:
        gate_lines = []
        for gate_key, gate_label in [
            ("gate1", "技術面"),
            ("gate2", "ML分類"),
            ("gate3", "總體環境"),
            ("gate4", "AI代理共識"),
        ]:
            gr = entry_signal.gate_results.get(gate_key)
            if gr:
                icon = "✅" if gr.passed else "❌"
                gate_lines.append(f"{icon} **{gate_label}** — {gr.details}")
        g2 = entry_signal.gate_results.get("gate2")
        if g2 and g2.score != 0.5:
            gate_lines.append(f"ML 預測機率: **{g2.score:.1%}**")
        embed.add_field(
            name=f"🔒 四驗證 ({entry_signal.gates_passed}/4)",
            value="\n".join(gate_lines) if gate_lines else "N/A",
            inline=False,
        )

    embed.set_footer(text="🦋 花城 自動監控 | 僅供參考，不構成投資建議")
    return embed


# ---------------------------------------------------------------------------
# 自適應監控模式 (normal / alert)
# ---------------------------------------------------------------------------
_NORMAL_INTERVAL = 60   # 正常模式: 60 分鐘掃一次
_ALERT_INTERVAL = 15    # 警戒模式: 15 分鐘掃一次
_ALERT_VIX_THRESHOLD = 25
_ALERT_SWING_PCT = 3.0
_SPIKE_PCT = 5.0
_VOL_SPIKE_RATIO = 2.0

_monitor_mode: str = "normal"       # "normal" | "alert"
_last_scan_ts: float = 0.0          # monotonic timestamp of last actual scan
_spike_cooldowns: dict[str, str] = {}   # ticker -> date string (per-day cooldown)
_vol_cooldowns: dict[str, str] = {}     # ticker -> date string (per-day cooldown)

import time as _time_mod


def get_monitor_mode() -> str:
    return _monitor_mode


def should_scan_now() -> bool:
    """根據 monitor mode 判斷本次 tick 是否該掃描。
    容許 30 秒誤差，避免計時器精度問題導致跳過 tick。"""
    global _last_scan_ts
    now = _time_mod.monotonic()
    interval = _ALERT_INTERVAL if _monitor_mode == "alert" else _NORMAL_INTERVAL
    if now - _last_scan_ts >= interval * 60 - 30:
        _last_scan_ts = now
        return True
    return False


def update_monitor_mode(vix_value: float | None, swing_pcts: list[float]) -> str:
    """根據 VIX + 個股波動更新監控模式。回傳新模式。"""
    global _monitor_mode
    old = _monitor_mode

    triggered = False
    if vix_value is not None and vix_value >= _ALERT_VIX_THRESHOLD:
        triggered = True
    if any(abs(p) >= _ALERT_SWING_PCT for p in swing_pcts):
        triggered = True

    _monitor_mode = "alert" if triggered else "normal"
    if old != _monitor_mode:
        log.info(f"monitor_mode: {old} → {_monitor_mode}")
    return _monitor_mode


def _market_date_key(ticker: str) -> str:
    """回傳該股票所屬市場的當前交易日字串（用於每日冷卻）。"""
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        return datetime.now(_TW).strftime("%Y-%m-%d")
    return datetime.now(_US_ET).strftime("%Y-%m-%d")


def check_spike_alert(ticker: str, change_pct: float) -> str | None:
    """檢查暴漲暴跌（±5%）。同一交易日同一檔只發一次。"""
    if abs(change_pct) < _SPIKE_PCT:
        return None
    today = _market_date_key(ticker)
    if _spike_cooldowns.get(ticker) == today:
        return None
    _spike_cooldowns[ticker] = today
    return "spike_up" if change_pct > 0 else "spike_down"


def check_volume_alert(ticker: str, vol_ratio: float) -> bool:
    """檢查量能異常（量比 ≥ 2.0）。同一交易日同一檔只發一次。"""
    if vol_ratio < _VOL_SPIKE_RATIO:
        return False
    today = _market_date_key(ticker)
    if _vol_cooldowns.get(ticker) == today:
        return False
    _vol_cooldowns[ticker] = today
    return True


def build_spike_embed(ticker: str, name: str, analysis: dict, direction: str) -> discord.Embed:
    """暴漲暴跌警報 Embed。"""
    change_pct = analysis["change_pct"]
    current = analysis["current"]

    if direction == "spike_up":
        title = f"🟢 暴漲警報 — {ticker}"
        color = 0x00FF00
        desc = f"**{name}** 漲幅達 **{change_pct:+.2f}%**"
    else:
        title = f"🔴 暴跌警報 — {ticker}"
        color = 0xFF0000
        desc = f"**{name}** 跌幅達 **{change_pct:+.2f}%**"

    embed = discord.Embed(title=title, description=desc, color=color,
                          timestamp=datetime.now(timezone.utc))
    embed.add_field(
        name="📊 數據",
        value=(f"現價 **${current:,.2f}** | RSI {analysis['rsi']:.1f} "
               f"| 量比 {analysis['vol_ratio']:.1f}x | 評分 {analysis['total_score']}/100"),
        inline=False,
    )
    embed.set_footer(text="🦋 花城 即時警報 | 僅供參考")
    return embed


def build_volume_embed(ticker: str, name: str, analysis: dict) -> discord.Embed:
    """量能異常警報 Embed。"""
    vol_ratio = analysis["vol_ratio"]
    current = analysis["current"]
    change_pct = analysis["change_pct"]

    embed = discord.Embed(
        title=f"🟠 量能異常 — {ticker}",
        description=f"**{name}** 成交量為 20 日均量的 **{vol_ratio:.1f} 倍**",
        color=0xFF8C00,
        timestamp=datetime.now(timezone.utc),
    )
    embed.add_field(
        name="📊 數據",
        value=(f"現價 **${current:,.2f}** ({change_pct:+.2f}%) | RSI {analysis['rsi']:.1f} "
               f"| 評分 {analysis['total_score']}/100"),
        inline=False,
    )
    embed.set_footer(text="🦋 花城 量能警報 | 僅供參考")
    return embed
