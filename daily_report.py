"""
花城 — 每日開盤前預報 + 收盤後報告
提供 Discord Embed 建構 + Gemini prompt 生成。
"""

from datetime import datetime, timezone, timedelta

import discord

_TW = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# Embed 顏色
# ---------------------------------------------------------------------------
_COLOR_PREMARKET = 0xF39C12   # amber
_COLOR_POSTCLOSE = 0x3498DB   # blue
_COLOR_MIDDAY    = 0x2ECC71   # green

# ---------------------------------------------------------------------------
# 分類 watchlist
# ---------------------------------------------------------------------------

def _classify_watchlist(wl: list[dict]) -> tuple[list[dict], list[dict]]:
    """將 watchlist 分為台股和美股兩組。"""
    tw, us = [], []
    for item in wl:
        ticker = item.get("ticker", "")
        if ticker.endswith(".TW") or ticker.endswith(".TWO"):
            tw.append(item)
        else:
            us.append(item)
    return tw, us


def _score_indicator(score: int) -> str:
    if score >= 75:
        return "🟢"
    elif score >= 60:
        return "🔵"
    elif score >= 45:
        return "🟡"
    elif score >= 30:
        return "🟠"
    else:
        return "🔴"


# ---------------------------------------------------------------------------
# 排序格式化
# ---------------------------------------------------------------------------

def _build_ranked_lines(
    group: list[dict],
    batch: list[dict],
    wl: list[dict],
    max_items: int = 15,
    score_history: dict | None = None,
) -> str:
    """依 score 降序排列，回傳格式化文字。batch 與 wl 對齊。"""
    from score_history import get_score_delta

    # 建立 ticker -> batch result 的對照表
    ticker_to_result = {}
    for item, br in zip(wl, batch):
        ticker_to_result[item["ticker"]] = (item, br)

    scored = []
    for item in group:
        pair = ticker_to_result.get(item["ticker"])
        if not pair:
            continue
        _, br = pair
        analysis = br.get("analysis")
        if analysis is None:
            continue
        scored.append((item, analysis))

    scored.sort(key=lambda x: x[1]["total_score"], reverse=True)

    lines = []
    for i, (item, analysis) in enumerate(scored[:max_items], 1):
        score = analysis["total_score"]
        indicator = _score_indicator(score)
        rsi = analysis.get("rsi", 0)
        change = analysis.get("change_pct", 0)

        # Score delta
        delta_text = ""
        if score_history:
            delta_text, _ = get_score_delta(item["ticker"], score, score_history)

        lines.append(
            f"`{i:>2}.` {indicator} **{item['name']}** `{item['input']}` "
            f"— {score}分{delta_text} RSI:{rsi:.0f} ({change:+.1f}%)"
        )

    return "\n".join(lines) if lines else "_暫無資料_"


# ---------------------------------------------------------------------------
# 進場候選 + RSI 警報
# ---------------------------------------------------------------------------

def _extract_signals(wl: list[dict], batch: list[dict]) -> tuple[str, str]:
    """
    回傳 (candidates_text, rsi_alerts_text)。
    進場候選: score >= 45 且 RSI < 70
    RSI 警報: RSI > 70 (超買) 或 RSI < 30 (超賣)
    """
    candidates = []
    overbought = []
    oversold = []

    for item, br in zip(wl, batch):
        analysis = br.get("analysis")
        if analysis is None:
            continue
        score = analysis["total_score"]
        rsi = analysis.get("rsi", 50)

        if score >= 45 and rsi < 70:
            candidates.append((item, analysis))

        if rsi > 70:
            overbought.append((item, rsi))
        elif rsi < 30:
            oversold.append((item, rsi))

    # 進場候選（依 score 降序）
    candidates.sort(key=lambda x: x[1]["total_score"], reverse=True)
    if candidates:
        cand_lines = []
        for item, analysis in candidates[:8]:
            score = analysis["total_score"]
            rsi = analysis.get("rsi", 0)
            cand_lines.append(f"• **{item['name']}** `{item['input']}` — {score}分 RSI:{rsi:.0f}")
        cand_text = "\n".join(cand_lines)
    else:
        cand_text = "_目前無符合條件的候選_"

    # RSI 警報
    alert_lines = []
    for item, rsi in overbought:
        alert_lines.append(f"🔴 **{item['name']}** `{item['input']}` RSI {rsi:.0f} (超買)")
    for item, rsi in oversold:
        alert_lines.append(f"🟢 **{item['name']}** `{item['input']}` RSI {rsi:.0f} (超賣)")
    rsi_text = "\n".join(alert_lines) if alert_lines else "_無 RSI 異常_"

    return cand_text, rsi_text


# ---------------------------------------------------------------------------
# 今日最大波動 top 3
# ---------------------------------------------------------------------------

def _extract_biggest_movers(wl: list[dict], batch: list[dict]) -> str:
    """回傳今日波動最大的前 3 檔。"""
    movers = []
    for item, br in zip(wl, batch):
        analysis = br.get("analysis")
        if analysis is None:
            continue
        change = analysis.get("change_pct", 0)
        movers.append((item, analysis, abs(change), change))

    movers.sort(key=lambda x: x[2], reverse=True)

    lines = []
    for item, analysis, abs_chg, change in movers[:3]:
        arrow = "📈" if change >= 0 else "📉"
        lines.append(
            f"{arrow} **{item['name']}** `{item['input']}` — "
            f"{change:+.2f}% (${analysis['current']:,.2f})"
        )

    return "\n".join(lines) if lines else "_暫無資料_"


# ---------------------------------------------------------------------------
# 開盤前預報 Embed
# ---------------------------------------------------------------------------

def build_premarket_embed(
    wl: list[dict],
    batch: list[dict],
    sentiment: dict,
    us_futures: dict | None = None,
    score_history: dict | None = None,
) -> discord.Embed:
    """建構開盤前預報 Embed (08:30 TWN)。"""
    from market_data import format_sentiment_block

    now_tw = datetime.now(_TW)
    date_str = now_tw.strftime("%Y/%m/%d")

    tw_group, us_group = _classify_watchlist(wl)

    # Description: 市場情緒
    desc_lines = [format_sentiment_block(sentiment)]

    # 美股期貨段落（獨立顯示，若 sentiment 裡沒有）
    if us_futures:
        sp_f = us_futures.get("sp500_futures")
        nq_f = us_futures.get("nasdaq_futures")
        if sp_f or nq_f:
            fut_lines = ["── 美股期貨 ──"]
            if sp_f:
                fut_lines.append(f"**ES (S&P 500期)** {sp_f['text']}")
            if nq_f:
                fut_lines.append(f"**NQ (NASDAQ期)** {nq_f['text']}")
            desc_lines.append("\n".join(fut_lines))

    embed = discord.Embed(
        title=f"\U0001f305 開盤前預報 ({date_str})",
        description="\n".join(desc_lines),
        color=_COLOR_PREMARKET,
        timestamp=now_tw,
    )

    # 台股技術評分
    tw_text = _build_ranked_lines(tw_group, batch, wl, max_items=15, score_history=score_history)
    embed.add_field(name="\U0001f1f9\U0001f1fc 台股技術評分", value=tw_text, inline=False)

    # 美股技術評分
    us_text = _build_ranked_lines(us_group, batch, wl, max_items=10, score_history=score_history)
    embed.add_field(name="\U0001f1fa\U0001f1f8 美股技術評分", value=us_text, inline=False)

    # 進場候選 + RSI 警報
    cand_text, rsi_text = _extract_signals(wl, batch)
    embed.add_field(name="\U0001f3af 進場候選 (score\u226545, RSI<70)", value=cand_text, inline=False)
    embed.add_field(name="\u26a0\ufe0f RSI 警報", value=rsi_text, inline=False)

    embed.set_footer(text="花城 Daily Report")
    return embed


# ---------------------------------------------------------------------------
# 收盤後報告 Embed
# ---------------------------------------------------------------------------

def build_postclose_embed(
    wl: list[dict],
    batch: list[dict],
    sentiment: dict,
    score_history: dict | None = None,
) -> discord.Embed:
    """建構收盤後報告 Embed (14:00 TWN)。"""
    from market_data import format_sentiment_block

    now_tw = datetime.now(_TW)
    date_str = now_tw.strftime("%Y/%m/%d")

    tw_group, us_group = _classify_watchlist(wl)

    desc_lines = [format_sentiment_block(sentiment)]

    embed = discord.Embed(
        title=f"\U0001f4ca 收盤報告 ({date_str})",
        description="\n".join(desc_lines),
        color=_COLOR_POSTCLOSE,
        timestamp=now_tw,
    )

    # 台股排名
    tw_text = _build_ranked_lines(tw_group, batch, wl, max_items=15, score_history=score_history)
    embed.add_field(name="\U0001f1f9\U0001f1fc 台股排名", value=tw_text, inline=False)

    # 今日最大波動 top 3
    movers_text = _extract_biggest_movers(wl, batch)
    embed.add_field(name="\U0001f4c8\U0001f4c9 今日最大波動", value=movers_text, inline=False)

    # 美股排名
    us_text = _build_ranked_lines(us_group, batch, wl, max_items=10, score_history=score_history)
    embed.add_field(name="\U0001f1fa\U0001f1f8 美股排名", value=us_text, inline=False)

    # 進場候選 + RSI 警報
    cand_text, rsi_text = _extract_signals(wl, batch)
    embed.add_field(name="\U0001f3af 進場候選 (score\u226545, RSI<70)", value=cand_text, inline=False)
    embed.add_field(name="\u26a0\ufe0f RSI 警報", value=rsi_text, inline=False)

    embed.set_footer(text="花城 Daily Report")
    return embed


# ---------------------------------------------------------------------------
# 盤中快閃 Embed (10:30)
# ---------------------------------------------------------------------------

def build_midday_embed(
    wl: list[dict],
    batch: list[dict],
    score_history: dict | None = None,
) -> discord.Embed:
    """建構盤中快閃 Embed (10:30 TWN)。wl 與 batch 須對齊（通常只傳台股）。"""
    from score_history import get_score_delta

    now_tw = datetime.now(_TW)
    date_str = now_tw.strftime("%Y/%m/%d %H:%M")

    scored = []
    for item, br in zip(wl, batch):
        analysis = br.get("analysis")
        if analysis is None:
            continue
        scored.append((item, analysis))

    scored.sort(key=lambda x: x[1]["total_score"], reverse=True)

    lines = []
    vol_alerts = []
    for i, (item, analysis) in enumerate(scored[:15], 1):
        score = analysis["total_score"]
        indicator = _score_indicator(score)
        change = analysis.get("change_pct", 0)
        vol_ratio = analysis.get("vol_ratio", 1.0)

        # Score delta vs 08:30
        delta_text = ""
        if score_history:
            delta_text, _ = get_score_delta(item["ticker"], score, score_history)

        vol_flag = " **[量!]**" if vol_ratio > 2.0 else ""
        lines.append(
            f"`{i:>2}.` {indicator} **{item['name']}** `{item['input']}` "
            f"— {score}分{delta_text} ({change:+.1f}%){vol_flag}"
        )

        if vol_ratio > 2.0:
            vol_alerts.append(f"  {item['name']} `{item['input']}` 量比 {vol_ratio:.1f}x")

    embed = discord.Embed(
        title=f"\u26a1 盤中快閃 ({date_str})",
        color=_COLOR_MIDDAY,
        timestamp=now_tw,
    )

    tw_text = "\n".join(lines) if lines else "_暫無資料_"
    embed.add_field(name="\U0001f1f9\U0001f1fc 台股即時", value=tw_text, inline=False)

    if vol_alerts:
        embed.add_field(
            name="\U0001f4a5 異常量能",
            value="\n".join(vol_alerts),
            inline=False,
        )

    embed.set_footer(text="花城 Midday Flash")
    return embed


# ---------------------------------------------------------------------------
# Gemini Prompt 生成
# ---------------------------------------------------------------------------

def build_gemini_prompt_premarket(sentiment: dict, us_futures: dict | None = None) -> str:
    """生成盤前 AI 觀察的 Gemini prompt。"""
    now_tw = datetime.now(_TW)
    date_str = now_tw.strftime("%Y年%m月%d日")

    parts = [f"今天是 {date_str}。請搜尋並分析以下盤前資訊："]
    parts.append("1. 隔夜美股（S&P 500、NASDAQ、道瓊）收盤表現")
    parts.append("2. 亞洲盤前期貨動向")
    parts.append("3. 影響今日台股的重大新聞（半導體、AI、量子計算）")
    parts.append("4. 台股今日可能走勢的簡要觀察")

    # 附加即時數據
    data_parts = []
    fg = sentiment.get("fear_greed")
    if fg:
        data_parts.append(f"CNN 恐懼貪婪指數: {fg['score']} ({fg['label']})")
    vix = sentiment.get("vix")
    if vix:
        data_parts.append(f"VIX: {vix['value']} ({vix['change_pct']:+.1f}%)")
    sp500 = sentiment.get("sp500")
    if sp500:
        data_parts.append(f"S&P 500: {sp500['value']:,.2f} ({sp500['change_pct']:+.2f}%)")
    nasdaq = sentiment.get("nasdaq")
    if nasdaq:
        data_parts.append(f"NASDAQ: {nasdaq['value']:,.2f} ({nasdaq['change_pct']:+.2f}%)")

    # 美股期貨
    if us_futures:
        sp_f = us_futures.get("sp500_futures")
        if sp_f:
            data_parts.append(f"S&P 500 期貨 (ES): {sp_f['value']:,.2f} ({sp_f['change_pct']:+.2f}%)")
        nq_f = us_futures.get("nasdaq_futures")
        if nq_f:
            data_parts.append(f"NASDAQ 期貨 (NQ): {nq_f['value']:,.2f} ({nq_f['change_pct']:+.2f}%)")

    if data_parts:
        parts.append("\n【即時市場數據】\n" + "\n".join(data_parts))

    parts.append("\n請用花城的語氣，簡潔地（300字內）給出盤前觀察。")
    return "\n".join(parts)


def build_gemini_prompt_postclose(sentiment: dict) -> str:
    """生成收盤 AI 觀察的 Gemini prompt。"""
    now_tw = datetime.now(_TW)
    date_str = now_tw.strftime("%Y年%m月%d日")

    parts = [f"今天是 {date_str}。請搜尋並分析今日收盤資訊："]
    parts.append("1. 台股今日表現總結（加權指數、成交量、主要類股）")
    parts.append("2. 影響盤勢的重大事件或新聞")
    parts.append("3. 美股盤前動態（若已開盤則看即時表現）")
    parts.append("4. 明日展望與需注意事項")

    data_parts = []
    twii = sentiment.get("twii")
    if twii:
        data_parts.append(f"台股加權: {twii['value']:,.2f} ({twii['change_pct']:+.2f}%)")
    fg = sentiment.get("fear_greed")
    if fg:
        data_parts.append(f"CNN 恐懼貪婪指數: {fg['score']} ({fg['label']})")
    vix = sentiment.get("vix")
    if vix:
        data_parts.append(f"VIX: {vix['value']} ({vix['change_pct']:+.1f}%)")
    sp500 = sentiment.get("sp500")
    if sp500:
        data_parts.append(f"S&P 500: {sp500['value']:,.2f} ({sp500['change_pct']:+.2f}%)")

    if data_parts:
        parts.append("\n【即時市場數據】\n" + "\n".join(data_parts))

    parts.append("\n請用花城的語氣，簡潔地（300字內）給出收盤觀察與明日展望。")
    return "\n".join(parts)
