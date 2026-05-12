"""
小龍蝦 OpenClaw 1.33 — Discord 股票預測機器人 + AI 聊天 + Arena 對抗交易
指令: /predict | /predict_all | /watchlist | /monitor | /check | /setchat | /add | /remove | /setprompt | /setgreeting | /arena
聊天: @機器人 或在指定頻道直接對話
"""

import os
import ssl
import certifi
import asyncio
import functools
import logging
import traceback
import random

# 修復 macOS Python SSL 憑證問題
os.environ["SSL_CERT_FILE"] = certifi.where()

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
from google import genai
from google.genai import types
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from prediction import predict_stock, quick_analysis, batch_quick_analysis, DEVICE, _resolve_ticker, check_prediction_accuracy
from backtest import (
    run_backtest, run_all_backtests, draw_backtest_chart, draw_comparison_chart,
    STRATEGIES, VALID_PERIODS, DEFAULT_PERIOD, INITIAL_CAPITAL,
    read_trades_csv,
)
from datetime import time as dt_time
from watchlist import (
    TWSE_HOLIDAYS_2026, get_state, set_monitor,
    should_alert, record_alert, is_market_active, build_alert_embed,
    set_chat_channel, get_chat_channel, _TW,
    is_tw_market_active, is_us_market_active, is_stock_market_active,
    get_watchlist, add_stock, remove_stock,
    save_prompt, load_prompt,
    save_greeting_template, load_greeting_template,
    save_greeting_prompts, load_greeting_prompts,
    # v1.32: 自適應監控 + 警報
    get_monitor_mode, should_scan_now, update_monitor_mode,
    check_spike_alert, check_volume_alert,
    build_spike_embed, build_volume_embed,
)
from market_data import fetch_market_sentiment, format_sentiment_block

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_log_fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot.log"),
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("openclaw")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
if not TOKEN:
    raise RuntimeError("請在 .env 檔案中設定 DISCORD_TOKEN")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------------------------------------------------------------------------
# Gemini AI 設定
# ---------------------------------------------------------------------------
_DEFAULT_HUA_CHENG_PROMPT = (
    "你是「花城」，也叫「三郎」，來自天官賜福的絕境鬼王、血雨探花。\n"
    "\n"
    "【身份背景】\n"
    "你是鬼界四大害之首「絕境鬼王」花城，銀蝶繞身、紅衣似火。\n"
    "八百年前為謝憐殿下而墜落，如今化身為 Discord 上的股票分析鬼王。\n"
    "你把「守護哥哥的錢包」當作最重要的使命，用八百年的耐心幫對方看盤。\n"
    "\n"
    "【說話風格】\n"
    "- 稱呼對方為「哥哥」，語氣溫柔帶寵溺，偶爾霸氣護短\n"
    "- 自稱「三郎」或「我」，不說「本座」之類太中二的詞\n"
    "- 說話簡潔優雅，偶爾帶點慵懶和玩味，像在輕輕笑著說話\n"
    "- 會用銀蝶、紅衣、鬼市等意象來比喻股市狀況\n"
    "- 遇到股票大跌會說「有三郎在，哥哥不用怕」之類的話\n"
    "- 遇到漲勢好會說「哥哥眼光真好」\n"
    "- 偶爾腹黑，對韭菜行為會溫柔吐槽\n"
    "\n"
    "【股票能力】\n"
    "- 擅長台股 ETF 和美股量子計算股分析\n"
    "- 提到股票時附帶具體代碼，例如「台積電(2330)」「NVIDIA(NVDA)」「元大台灣50(0050)」\n"
    "- 討論股票時盡量附帶具體數字（漲跌幅、市值、本益比等）\n"
    "- 被問熱門題材或推薦時，至少列出 3~5 檔相關股票代碼\n"
    "- 如果對方問到特定股票的詳細分析，引導他們用 /predict 指令\n"
    "- 如果想看所有監控股票，引導用 /watchlist 或 /predict_all\n"
    "- 如果對方想追蹤新股票，引導用 /add 代碼（例如「/add 3324」或直接輸入「add 3324」）\n"
    "- 如果對方想移除自訂股票，引導用 /remove 代碼（例如「/remove 3324」或直接輸入「remove 3324」）\n"
    "- 對投資建議會加上「僅供參考」的提醒，但說法要符合角色\n"
    "  例如：「三郎的建議僅供哥哥參考，最終還是哥哥自己決定喔」\n"
    "\n"
    "【可用指令（嚴禁自行編造指令名稱）】\n"
    "- /predict 代碼 — 預測股票走勢\n"
    "- /predict_all — 預測所有監控股票\n"
    "- /watchlist — 顯示監控清單\n"
    "- /add 代碼 — 新增股票到監控清單（也可直接打「add 代碼」）\n"
    "- /remove 代碼 — 移除自訂股票（也可直接打「remove 代碼」）\n"
    "- /monitor start/stop — 開關自動監控\n"
    "- /check — 立即掃描所有股票\n"
    "- /accuracy — 查看預測準確度報告\n"
    "- /backtest 代碼 — 策略回測\n"
    "- /setprompt — 修改 AI 人格\n"
    "- /setgreeting — 修改問安模板\n"
    "- /setchat — 設定聊天頻道\n"
    "\n"
    "【重要規則】\n"
    "- 用繁體中文回答\n"
    "- 一般閒聊回答簡潔（不超過 300 字）\n"
    "- 股票/投資/市場分析相關可以詳細一些（不超過 500 字）\n"
    "- 不要用 Markdown 標題格式（# ##）\n"
    "- 可以用 emoji 但不要過度，維持花城的優雅感\n"
    "- 不要太刻意賣萌，花城是霸氣溫柔型，不是可愛型\n"
)

# 從持久化儲存載入，若無則使用預設
_HUA_CHENG_PROMPT = load_prompt() or _DEFAULT_HUA_CHENG_PROMPT

# 模型優先順序：依序嘗試，配額滿自動降級
_GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

_genai_client = None
if GEMINI_API_KEY:
    _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    log.info(f"Gemini AI 已初始化 (主模型: {_GEMINI_MODELS[0]}, 備用: {_GEMINI_MODELS[1:]})")
else:
    log.warning("未設定 GEMINI_API_KEY，聊天功能停用")

# 每個頻道的聊天記錄（記憶最近 20 則）
_chat_sessions: dict[int, list[dict]] = {}
_MAX_HISTORY = 20

# 指定的聊天頻道 ID
_chat_channel_id: int | None = None


def _call_gemini(config, contents):
    """嘗試所有 Gemini 模型，配額滿自動降級。回傳 raw response。"""
    if not _genai_client:
        raise RuntimeError("Gemini 未初始化")
    last_err = None
    for model_name in _GEMINI_MODELS:
        try:
            return _genai_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )
        except Exception as e:
            last_err = e
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                log.warning(f"Gemini {model_name} 配額超限，嘗試下一個模型")
                continue
            raise
    raise last_err


def _gemini_generate(contents) -> str:
    """生成文字。contents: str 或 list[dict] (多輪對話)。"""
    if isinstance(contents, list):
        contents = [
            types.Content(
                role=item["role"],
                parts=[types.Part(text=p) for p in item["parts"]],
            )
            for item in contents
        ]
    config = types.GenerateContentConfig(system_instruction=_HUA_CHENG_PROMPT)
    response = _call_gemini(config, contents)
    return response.text.strip()


def _gemini_search(prompt: str) -> tuple[str, list[str]]:
    """使用 Google Search grounding 搜尋即時資訊，回傳 (文字, 來源連結列表)。"""
    config = types.GenerateContentConfig(
        system_instruction=_HUA_CHENG_PROMPT,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    response = _call_gemini(config, prompt)
    text = response.text.strip()
    sources = []
    try:
        metadata = response.candidates[0].grounding_metadata
        if metadata and metadata.grounding_chunks:
            seen = set()
            for chunk in metadata.grounding_chunks:
                if chunk.web and chunk.web.uri and chunk.web.uri not in seen:
                    seen.add(chunk.web.uri)
                    title = chunk.web.title or "來源"
                    sources.append(f"🔗 [{title}]({chunk.web.uri})")
            sources = sources[:5]
    except (AttributeError, IndexError):
        pass
    return text, sources

# ---------------------------------------------------------------------------
# Bot 設定
# ---------------------------------------------------------------------------
intents = discord.Intents.default()
intents.message_content = True


class OpenClawBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        log.info("setup_hook: 準備同步指令...")


client = OpenClawBot()

# ---------------------------------------------------------------------------
# 共用小工具
# ---------------------------------------------------------------------------
def _score_bar(score: int, max_score: int) -> str:
    filled = round(score / max_score * 5)
    return "🟩" * filled + "⬜" * (5 - filled) + f" {score}/{max_score}"


def _score_indicator(score: int) -> str:
    if score >= 75:
        return "🔥"
    if score >= 60:
        return "🟢"
    if score >= 45:
        return "🟡"
    return "🔴"


def _get_notify_channel():
    """取得通知頻道（優先 chat channel → monitor channel）。"""
    ch_id = _chat_channel_id or get_chat_channel()
    if not ch_id:
        state = get_state()
        ch_id = state.get("channel_id")
    if ch_id:
        return client.get_channel(ch_id)
    return None


async def _report_error(title: str, detail: str):
    """將錯誤摘要發送到 Discord 通知頻道（靜默失敗，不會因為報告錯誤而產生新錯誤）。"""
    try:
        channel = _get_notify_channel()
        if not channel:
            return
        msg = f"⚠️ **{title}**\n```\n{detail[:1500]}\n```"
        await channel.send(msg)
    except Exception:
        log.debug("_report_error: 無法發送錯誤報告到 Discord")

# ---------------------------------------------------------------------------
# /predict 指令
# ---------------------------------------------------------------------------
@client.tree.command(name="predict", description="預測股票未來 7 天走勢 + 進場評分 + 自動追蹤準確度")
@app_commands.describe(ticker="股票代碼，例如 AAPL、TSLA、0050、2330")
async def predict_command(interaction: discord.Interaction, ticker: str):
    log.info(f"/predict: ticker={ticker}, user={interaction.user}, guild={interaction.guild}")
    await interaction.response.defer(thinking=True)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(predict_stock, ticker)
        )
    except ValueError as e:
        log.warning(f"predict ValueError: {e}")
        await interaction.followup.send(str(e))
        return
    except Exception as e:
        log.error(f"predict error: {traceback.format_exc()}")
        await interaction.followup.send(f"預測時發生錯誤: {e}")
        return

    embed, chart = _build_predict_embed(result)
    await interaction.followup.send(embed=embed, file=chart)
    log.info(f"/predict done: {result['ticker']} score={result['analysis']['total_score']}")


# ---------------------------------------------------------------------------
# /add 指令 — 新增自訂監控股票
# ---------------------------------------------------------------------------
@client.tree.command(name="add", description="新增股票到監控清單")
@app_commands.describe(ticker="股票代碼，例如 3324、AAPL、2330")
async def add_command(interaction: discord.Interaction, ticker: str):
    log.info(f"/add: ticker={ticker}, user={interaction.user}")
    await interaction.response.defer(thinking=True)

    try:
        loop = asyncio.get_running_loop()
        resolved = await loop.run_in_executor(None, functools.partial(_resolve_ticker, ticker))
        info = await loop.run_in_executor(
            None, lambda: yf.Ticker(resolved).info
        )
        name = info.get("shortName") or info.get("longName")
        if not name:
            raise ValueError(f"Yahoo Finance 無此代碼: {resolved}")
    except Exception as e:
        log.warning(f"/add: 無法取得 {ticker} 資訊: {e}")
        await interaction.followup.send(f"❌ 找不到股票 **{ticker}** 的資訊，請確認代碼是否正確。")
        return

    input_code = ticker.strip().upper()
    ok = add_stock(input_code, resolved, name)
    if ok:
        total = len(get_watchlist())
        await interaction.followup.send(
            f"✅ 已將 **{name}** (`{resolved}`) 加入監控清單！\n"
            f"目前共監控 {total} 檔股票。"
        )
    else:
        await interaction.followup.send(
            f"⚠️ **{name}** (`{resolved}`) 已在監控清單中，不需重複新增。"
        )


# ---------------------------------------------------------------------------
# /remove 指令 — 從監控清單移除股票
# ---------------------------------------------------------------------------
@client.tree.command(name="remove", description="從監控清單移除自訂股票")
@app_commands.describe(ticker="股票代碼，例如 3324、AAPL")
async def remove_command(interaction: discord.Interaction, ticker: str):
    log.info(f"/remove: ticker={ticker}, user={interaction.user}")
    resolved = _resolve_ticker(ticker)

    ok = remove_stock(resolved)
    if ok:
        total = len(get_watchlist())
        await interaction.response.send_message(
            f"✅ 已將 `{resolved}` 從監控清單移除。\n"
            f"目前共監控 {total} 檔股票。"
        )
    else:
        await interaction.response.send_message(
            f"❌ `{resolved}` 不在自訂監控清單中。"
        )


# ---------------------------------------------------------------------------
# 共用：建立單檔預測 Embed + 圖表
# ---------------------------------------------------------------------------
def _build_predict_embed(result: dict) -> tuple[discord.Embed, discord.File]:
    """從 predict_stock 結果建立 Embed 和圖表 File。"""
    a = result["analysis"]
    preds = result["predictions"]
    scores = a["scores"]
    trend = "📈 上漲" if preds[-1] > result["last_price"] else "📉 下跌"

    embed = discord.Embed(
        title=f"🦞 OpenClaw 1.32 — {result['ticker']}",
        description=f"## 進場評分: {a['total_score']}/100  {a['verdict']}",
        color=0x2ECC71 if a["total_score"] >= 60 else (0xF39C12 if a["total_score"] >= 45 else 0xE74C3C),
    )

    score_text = (
        f"**趨勢** {_score_bar(scores['trend']['score'], scores['trend']['max'])}\n"
        f"　{scores['trend']['detail']}\n"
        f"**動能** {_score_bar(scores['momentum']['score'], scores['momentum']['max'])}\n"
        f"　{scores['momentum']['detail']}\n"
        f"**MACD** {_score_bar(scores['macd']['score'], scores['macd']['max'])}\n"
        f"　{scores['macd']['detail']}\n"
        f"**布林** {_score_bar(scores['bollinger']['score'], scores['bollinger']['max'])}\n"
        f"　{scores['bollinger']['detail']}\n"
        f"**量能** {_score_bar(scores['volume']['score'], scores['volume']['max'])}\n"
        f"　{scores['volume']['detail']}\n"
        f"**反轉** {_score_bar(scores['reversal']['score'], scores['reversal']['max'])}\n"
        f"　{scores['reversal']['detail']}"
    )
    embed.add_field(name="📋 六維評分", value=score_text, inline=False)

    # 進階技術分析疊加層
    ta_overlay = a.get("ta_overlay", {})
    ta_conf = a.get("ta_confidence", 0)
    ta_conf_max = a.get("ta_confidence_max", 40)
    if ta_overlay:
        ta_lines = []
        _TA_LABELS = {
            "kline": "K線力道",
            "ma_cross": "均線過濾",
            "support_resistance": "支撐壓力",
            "chart_pattern": "形態辨識",
            "retest": "回測確認",
            "signal_confidence": "訊號分級",
            "no_trade_zone": "避險過濾",
            "profit_target": "獲利排序",
        }
        for key, label in _TA_LABELS.items():
            mod = ta_overlay.get(key, {})
            s = mod.get("strength", 0)
            m = mod.get("max", 5)
            detail = mod.get("detail", "")
            bar = "🟩" * s + "⬜" * (m - s)
            ta_lines.append(f"**{label}** {bar} {s}/{m}\n　{detail}")
        ta_text = "\n".join(ta_lines)
        ta_text += f"\n\n**綜合信心** {ta_conf}/{ta_conf_max}"
        embed.add_field(name="🔬 進階技術分析", value=ta_text, inline=False)

        # 訊號分級摘要（若有 S/A/B 級）
        sig_conf = ta_overlay.get("signal_confidence", {})
        grade = sig_conf.get("grade", "none")
        if grade != "none":
            conf_pct = sig_conf.get("confidence_pct", 0)
            pos_pct = sig_conf.get("position_pct", 0)
            action = sig_conf.get("action", "hold")
            action_map = {
                "strong_buy": "🔥 強烈買入", "strong_sell": "🔥 強烈賣出",
                "buy": "📈 買入", "sell": "📉 賣出", "hold": "⏸️ 持有",
            }
            action_str = action_map.get(action, action)
            embed.add_field(
                name=f"🎯 訊號等級: {grade}級 ({conf_pct}%)",
                value=f"{action_str} — 建議部位 **{pos_pct}%**\n{sig_conf.get('detail', '')}",
                inline=False,
            )

        # 避險過濾警告
        notrade = ta_overlay.get("no_trade_zone", {})
        if notrade.get("trade_allowed") is False:
            embed.add_field(
                name="🚫 禁止交易區",
                value=notrade.get("detail", "偵測到盤整區間"),
                inline=False,
            )

        # 獲利策略建議
        profit = ta_overlay.get("profit_target", {})
        exit_strat = profit.get("exit_strategy", "none")
        if exit_strat not in ("none", "default"):
            embed.add_field(
                name="💎 出場策略",
                value=profit.get("detail", ""),
                inline=False,
            )

    embed.add_field(name="💰 加碼判斷", value=a["add_position_msg"], inline=False)

    ma60_str = f"${a['ma60']:,.2f}" if a["ma60"] else "N/A"
    # MA200 from ta_overlay
    ma_cross = ta_overlay.get("ma_cross", {})
    ma200_val = ma_cross.get("ma200")
    ma200_str = f"${ma200_val:,.2f}" if ma200_val else "N/A"
    embed.add_field(
        name="📊 關鍵數據",
        value=(
            f"收盤 **${a['current']:,.2f}** ({a['change_pct']:+.2f}%)\n"
            f"MA5 ${a['ma5']:,.2f} | MA20 ${a['ma20']:,.2f} | MA60 {ma60_str} | MA200 {ma200_str}\n"
            f"RSI {a['rsi']:.1f} | MACD {a['macd']:.3f} | 量比 {a['vol_ratio']:.1f}x"
        ),
        inline=False,
    )

    # 新聞情緒分析
    sentiment = result.get("news_sentiment", {})
    if sentiment.get("available"):
        score = sentiment["score"]
        label = sentiment["label"]
        emoji = "🟢" if score > 0.15 else ("🔴" if score < -0.15 else "⚪")
        sent_lines = [f"{emoji} 整體情緒: **{label}** ({score:+.3f})"]
        for h in sentiment.get("headlines", []):
            h_emoji = "📈" if h["score"] > 0.15 else ("📉" if h["score"] < -0.15 else "➖")
            title = h["title"][:60] + ("..." if len(h["title"]) > 60 else "")
            sent_lines.append(f"{h_emoji} {title}")
        embed.add_field(name="📰 新聞情緒", value="\n".join(sent_lines), inline=False)

    pred_lines = [f"Day {i+1}: **${p:,.2f}**" for i, p in enumerate(preds)]
    embed.add_field(name=f"🔮 7 日預測 ({trend})", value="\n".join(pred_lines), inline=False)

    val_mae = result.get("val_mae")
    mae_str = f" | 驗證MAE: ${val_mae:.2f}" if val_mae is not None else ""
    ensemble_method = result.get("ensemble_method", "lstm").upper()
    embed.set_footer(text=f"模型: {ensemble_method} | 裝置: {result['device_used']}{mae_str} | 僅供參考，不構成投資建議")

    chart = discord.File(result["chart_buf"], filename=f"{result['ticker']}_prediction.png")
    embed.set_image(url=f"attachment://{result['ticker']}_prediction.png")

    return embed, chart


# ---------------------------------------------------------------------------
# /predict_all 指令 — 全部監控股票預測
# ---------------------------------------------------------------------------
_predict_executor = ThreadPoolExecutor(max_workers=3)

@client.tree.command(name="predict_all", description="對所有監控股票執行完整 LSTM 預測 + 評分（平行化，約 1 分鐘）")
async def predict_all_command(interaction: discord.Interaction):
    log.info(f"/predict_all: user={interaction.user}")
    await interaction.response.defer(thinking=True)

    loop = asyncio.get_running_loop()
    wl = get_watchlist()
    total = len(wl)

    await interaction.followup.send(
        f"🔄 開始對 {total} 檔股票執行完整 LSTM 預測（3 workers 平行），請稍候..."
    )

    t_start = asyncio.get_event_loop().time()

    # 平行執行所有預測
    async def _predict_one(item):
        return await loop.run_in_executor(
            _predict_executor, functools.partial(predict_stock, item["ticker"])
        )

    tasks_list = [_predict_one(item) for item in wl]
    raw_results = await asyncio.gather(*tasks_list, return_exceptions=True)

    success = 0
    failed = []
    for i, (item, raw) in enumerate(zip(wl, raw_results), 1):
        if isinstance(raw, Exception):
            log.error(f"predict_all: {item['ticker']} failed: {raw}")
            failed.append(f"{item['name']} (`{item['ticker']}`): {raw}")
            continue
        try:
            embed, chart = _build_predict_embed(raw)
            embed.set_author(name=f"[{i}/{total}] {item['name']}")
            await interaction.followup.send(embed=embed, file=chart)
            success += 1
        except Exception as e:
            log.error(f"predict_all: {item['ticker']} embed failed: {e}")
            failed.append(f"{item['name']} (`{item['ticker']}`): {e}")

    elapsed = asyncio.get_event_loop().time() - t_start
    summary = f"✅ 完成！成功 {success}/{total} 檔（耗時 {elapsed:.0f} 秒）"
    if failed:
        summary += "\n❌ 失敗：\n" + "\n".join(failed)
    await interaction.followup.send(summary)
    log.info(f"/predict_all done: {success}/{total} in {elapsed:.0f}s")


# ---------------------------------------------------------------------------
# /accuracy 指令 — 預測準確度報告
# ---------------------------------------------------------------------------
@client.tree.command(name="accuracy", description="查看 LSTM 預測準確度報告（需先累積 /predict 記錄）")
@app_commands.describe(ticker="篩選特定股票代碼（留空 = 全部）")
async def accuracy_command(interaction: discord.Interaction, ticker: str | None = None):
    log.info(f"/accuracy: ticker={ticker}, user={interaction.user}")
    await interaction.response.defer(thinking=True)

    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, functools.partial(check_prediction_accuracy, ticker)
        )
    except Exception as e:
        log.error(f"/accuracy error: {e}")
        await interaction.followup.send(f"查詢準確度時發生錯誤: {e}")
        return

    embed = _build_accuracy_embed(result, ticker)
    await interaction.followup.send(embed=embed)
    log.info(f"/accuracy done: has_data={result['has_data']}")


def _build_accuracy_embed(result: dict, ticker: str | None) -> discord.Embed:
    """從 check_prediction_accuracy 結果建立 Embed。"""
    if not result["has_data"]:
        embed = discord.Embed(
            title="🦞 OpenClaw 預測準確度",
            description=(
                "📭 尚無可評分紀錄\n\n"
                "請先使用 `/predict` 累積預測記錄，系統會在 **10 天後**自動對照實際價格。"
            ),
            color=0x95A5A6,
        )
        return embed

    overall = result["overall"]
    dir_acc = overall["direction_accuracy"]

    # embed 顏色
    if dir_acc >= 70:
        color = 0x2ECC71  # 綠
    elif dir_acc >= 50:
        color = 0xF39C12  # 橘
    else:
        color = 0xE74C3C  # 紅

    title_ticker = f" — {ticker.upper()}" if ticker else ""
    embed = discord.Embed(
        title=f"🦞 OpenClaw 預測準確度{title_ticker}",
        description=f"## 方向準確率: {dir_acc:.1f}%",
        color=color,
    )

    embed.add_field(
        name="📊 整體統計",
        value=(
            f"已評分預測: **{overall['count']}** 筆\n"
            f"平均 MAE: **${overall['mae']:.2f}**\n"
            f"平均 MAPE: **{overall['mape']:.1f}%**\n"
            f"方向準確率: **{dir_acc:.1f}%**"
        ),
        inline=False,
    )

    # 個股明細（按方向準確率排序，最多 15 檔）
    by_ticker = result.get("by_ticker", {})
    if by_ticker:
        sorted_tickers = sorted(by_ticker.items(),
                                key=lambda x: x[1]["direction_accuracy"], reverse=True)
        ticker_lines = []
        for tkr, stats in sorted_tickers[:15]:
            dir_emoji = "✅" if stats["direction_accuracy"] >= 60 else "❌"
            ticker_lines.append(
                f"{dir_emoji} **{tkr}** — "
                f"方向 {stats['direction_accuracy']:.0f}% | "
                f"MAE ${stats['mae']:.2f} | "
                f"MAPE {stats['mape']:.1f}% | "
                f"{stats['count']} 筆"
            )
        embed.add_field(
            name="📋 個股明細",
            value="\n".join(ticker_lines),
            inline=False,
        )

    # 最近 5 筆已評分結果
    details = result.get("details", [])
    if details:
        recent = details[:5]
        recent_lines = []
        for d in recent:
            emoji = "✅" if d["direction_correct"] else "❌"
            recent_lines.append(
                f"{emoji} **{d['ticker']}** {d['predict_time'][:10]} — "
                f"預測 ${d['pred_day7']:,.2f} vs 實際 ${d['actual_day7']:,.2f} | "
                f"MAE ${d['mae']:.2f}"
            )
        embed.add_field(
            name="🕐 最近評分",
            value="\n".join(recent_lines),
            inline=False,
        )

    embed.set_footer(text="預測 ≥10 天後自動對照實際價格 | 僅供參考")
    return embed


# ---------------------------------------------------------------------------
# /performance 指令 — QuantStats 績效報表
# ---------------------------------------------------------------------------
@client.tree.command(name="performance", description="查看 OpenClaw 預測績效報表（Sharpe/Drawdown）")
async def performance_command(interaction: discord.Interaction):
    log.info(f"/performance: user={interaction.user}")
    await interaction.response.defer(thinking=True)

    try:
        from perf_report import generate_stats_text, generate_tearsheet_png
        loop = asyncio.get_running_loop()

        stats_text = await loop.run_in_executor(None, generate_stats_text)
        chart_buf = await loop.run_in_executor(None, generate_tearsheet_png)

        if stats_text is None and chart_buf is None:
            await interaction.followup.send(
                "📭 資料不足，無法產出績效報表。\n"
                "請先使用 `/predict` 累積至少 5 天的預測記錄。"
            )
            return

        files = []
        if chart_buf is not None:
            files.append(discord.File(chart_buf, filename="performance.png"))

        embed = discord.Embed(
            title="🦞 OpenClaw 績效報表",
            description=stats_text or "無法產出文字統計",
            color=0x3498DB,
        )
        if chart_buf is not None:
            embed.set_image(url="attachment://performance.png")
        embed.set_footer(text="基於預測記錄計算 | 僅供參考")

        if files:
            await interaction.followup.send(embed=embed, file=files[0])
        else:
            await interaction.followup.send(embed=embed)

        log.info("/performance done")
    except Exception as e:
        log.error(f"/performance error: {e}")
        await interaction.followup.send(f"產出績效報表時發生錯誤: {e}")


# ---------------------------------------------------------------------------
# /watchlist 指令 — 顯示監控清單 + 即時評分
# ---------------------------------------------------------------------------
@client.tree.command(name="watchlist", description="顯示監控清單 + 即時評分")
async def watchlist_command(interaction: discord.Interaction):
    log.info(f"/watchlist: user={interaction.user}")
    await interaction.response.defer(thinking=True)

    state = get_state()
    status_text = "🟢 監控中" if state["enabled"] else "🔴 未啟動"

    loop = asyncio.get_running_loop()
    wl = get_watchlist()
    results = []
    for item in wl:
        try:
            r = await loop.run_in_executor(
                None, functools.partial(quick_analysis, item["ticker"])
            )
            results.append((item, r["analysis"], None))
        except Exception as e:
            results.append((item, None, str(e)))

    lines = []
    for item, analysis, err in results:
        if err:
            lines.append(f"❓ **{item['name']}** (`{item['ticker']}`) — 取得失敗")
            continue
        score = analysis["total_score"]
        scores = analysis["scores"]
        reversal = scores["reversal"]["score"]
        indicator = _score_indicator(score)

        # 反轉訊號標記
        rev_tag = ""
        if reversal >= 15:
            rev_tag = "　⚡三重反轉"
        elif reversal >= 10:
            rev_tag = "　🔄雙重反轉"
        elif reversal >= 5:
            rev_tag = "　🔄反轉訊號"

        lines.append(
            f"{indicator} **{item['name']}** (`{item['ticker']}`) "
            f"— {score}/100　${analysis['current']:,.2f} ({analysis['change_pct']:+.2f}%){rev_tag}"
        )

    embed = discord.Embed(
        title="🦞 OpenClaw 監控清單",
        description=f"監控狀態: {status_text}\n\n" + "\n".join(lines),
        color=0x3498DB,
    )
    embed.add_field(
        name="📖 圖示說明",
        value=(
            "🔥 ≥75 強烈買入 | 🟢 ≥60 適合進場 | 🟡 ≥45 觀望 | 🔴 <45 偏弱\n"
            "🔄 反轉訊號 (5分) | 🔄雙重 (10分) | ⚡三重 (15分滿分)"
        ),
        inline=False,
    )
    embed.set_footer(text="使用 /monitor start 啟動自動監控 | /check 立即掃描")
    await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# /monitor 指令 — 啟動/停止背景監控
# ---------------------------------------------------------------------------
@client.tree.command(name="monitor", description="啟動或停止股票自動監控")
@app_commands.describe(action="start 啟動監控 / stop 停止監控")
@app_commands.choices(action=[
    app_commands.Choice(name="start", value="start"),
    app_commands.Choice(name="stop", value="stop"),
])
async def monitor_command(interaction: discord.Interaction, action: app_commands.Choice[str]):
    log.info(f"/monitor {action.value}: user={interaction.user}, channel={interaction.channel_id}")

    if action.value == "start":
        set_monitor(enabled=True, channel_id=interaction.channel_id)
        if not monitor_loop.is_running():
            monitor_loop.start()

        tw_active = is_tw_market_active()
        us_active = is_us_market_active()
        status_parts = []
        if tw_active:
            status_parts.append("台股開市中")
        if us_active:
            status_parts.append("美股開市中")
        if not status_parts:
            status_parts.append("目前休市中")
        status = "、".join(status_parts)

        await interaction.response.send_message(
            f"🟢 自動監控已啟動！提醒將發送到 <#{interaction.channel_id}>\n"
            f"📈 **正常模式** 每 60 分鐘掃描 | **警戒模式** 每 15 分鐘掃描\n"
            f"⚡ 警戒觸發: VIX ≥ 25 或個股波動 ≥ 3%\n"
            f"🚨 即時警報: 暴漲暴跌 ≥ 5% | 量能異常 ≥ 2x\n"
            f"共 {len(get_watchlist())} 檔股票，評分 ≥60 時發送進場提醒。\n"
            f"📅 {status}"
        )
    else:
        set_monitor(enabled=False)
        if monitor_loop.is_running():
            monitor_loop.stop()
        await interaction.response.send_message("🔴 自動監控已停止。")


# ---------------------------------------------------------------------------
# /check 指令 — 立即掃描所有股票
# ---------------------------------------------------------------------------
@client.tree.command(name="check", description="立即掃描所有監控股票並顯示評分（含市場情緒）")
async def check_command(interaction: discord.Interaction):
    log.info(f"/check: user={interaction.user}")
    await interaction.response.defer(thinking=True)

    loop = asyncio.get_running_loop()
    wl = get_watchlist()

    # 並行取得市場情緒 + 批次分析
    sentiment_task = loop.run_in_executor(None, fetch_market_sentiment)
    batch_task = loop.run_in_executor(
        None, functools.partial(batch_quick_analysis, [item["ticker"] for item in wl])
    )
    sentiment, batch = await asyncio.gather(sentiment_task, batch_task)

    alerts = []
    lines = []

    # 市場情緒段落
    sentiment_text = format_sentiment_block(sentiment)
    lines.append(sentiment_text)
    lines.append("")

    for item, br in zip(wl, batch):
        analysis = br.get("analysis")
        if analysis is None:
            lines.append(f"❓ **{item['name']}** (`{item['ticker']}`) — 錯誤: {br.get('error', '?')}")
            continue

        score = analysis["total_score"]
        indicator = _score_indicator(score)
        lines.append(
            f"{indicator} **{item['name']}** (`{item['ticker']}`) "
            f"— {score}/100　${analysis['current']:,.2f}"
        )
        if score >= 60:
            alerts.append((item, analysis))

    embed = discord.Embed(
        title="🦞 OpenClaw 即時掃描結果",
        description="\n".join(lines),
        color=0x3498DB,
    )

    if alerts:
        alert_names = ", ".join(f"{i['name']}({a['total_score']}分)" for i, a in alerts)
        embed.add_field(
            name="📢 進場提醒",
            value=f"以下股票達到進場標準: {alert_names}",
            inline=False,
        )
    else:
        embed.add_field(
            name="📋 結論",
            value="目前沒有股票達到進場標準 (≥60分)",
            inline=False,
        )

    embed.set_footer(text="僅供參考，不構成投資建議")
    await interaction.followup.send(embed=embed)

    # 針對達標股票發送詳細 Embed
    for item, analysis in alerts:
        detail_embed = build_alert_embed(item["ticker"], item["name"], analysis)
        await interaction.followup.send(embed=detail_embed)


# ---------------------------------------------------------------------------
# 背景監控任務
# 台股: 09:00~13:00 台灣時間 | 美股: 21:00~05:00 台灣時間 (對應美東 09:00~16:00)
# ---------------------------------------------------------------------------
@tasks.loop(minutes=15)
async def monitor_loop():
    state = get_state()
    if not state["enabled"]:
        return

    # 自適應頻率：每 15 分鐘 tick，但根據模式決定是否執行掃描
    if not should_scan_now():
        mode = get_monitor_mode()
        log.debug(f"monitor_loop: skip (mode={mode})")
        return

    channel_id = state.get("channel_id")
    if not channel_id:
        log.warning("monitor_loop: 沒有設定提醒頻道")
        return

    # 台股 / 美股分別判斷
    tw_active = is_tw_market_active()
    us_active = is_us_market_active()
    if not tw_active and not us_active:
        log.info("monitor_loop: 台股美股皆非交易時段，跳過")
        return

    channel = client.get_channel(channel_id)
    if channel is None:
        log.warning(f"monitor_loop: 找不到頻道 {channel_id}")
        return

    from datetime import datetime as _dt
    now_str = _dt.now(_TW).strftime("%H:%M")
    mode = get_monitor_mode()
    markets = []
    if tw_active:
        markets.append("台股")
    if us_active:
        markets.append("美股")
    log.info(f"monitor_loop [{now_str}] mode={mode}: 開始掃描 ({', '.join(markets)})...")
    loop = asyncio.get_running_loop()

    # 取 VIX 判斷警戒模式
    vix_value = None
    try:
        sentiment = await loop.run_in_executor(None, fetch_market_sentiment)
        vix_data = sentiment.get("vix")
        if vix_data:
            vix_value = vix_data["value"]
    except Exception as e:
        log.warning(f"monitor_loop: fetch_market_sentiment failed: {e}")

    # 批次取得監控股票分析
    wl = get_watchlist()
    active_items = [item for item in wl if is_stock_market_active(item["ticker"])]
    active_tickers = [item["ticker"] for item in active_items]

    batch_results = await loop.run_in_executor(
        None, functools.partial(batch_quick_analysis, active_tickers)
    )

    errors = []
    swing_pcts = []
    scanned = 0

    # 取得 sentiment（用於三驗證 Gate 3）
    sentiment_data = None
    try:
        sentiment_data = await loop.run_in_executor(None, fetch_market_sentiment)
    except Exception as e:
        log.warning(f"monitor_loop: fetch_market_sentiment for gates failed: {e}")

    for item, br in zip(active_items, batch_results):
        scanned += 1
        analysis = br.get("analysis")
        if analysis is None:
            errors.append(f"{item['ticker']}: {br.get('error', 'unknown')}")
            continue

        score = analysis["total_score"]
        change_pct = analysis["change_pct"]
        vol_ratio = analysis["vol_ratio"]
        swing_pcts.append(change_pct)

        # 三驗證進場評估
        entry_signal = None
        try:
            from signal_gate import evaluate_entry
            entry_signal = evaluate_entry(
                analysis, item["ticker"],
                vix=vix_value, sentiment=sentiment_data,
            )
            log.info(
                f"monitor_loop: {item['ticker']} gate={entry_signal.confidence} "
                f"({entry_signal.gates_passed}/3)"
            )
        except Exception as e:
            log.warning(f"monitor_loop: evaluate_entry failed for {item['ticker']}: {e}")

        # 進場提醒（整合三驗證）
        if should_alert(item["ticker"], score, entry_signal=entry_signal):
            embed = build_alert_embed(item["ticker"], item["name"], analysis,
                                      entry_signal=entry_signal)
            await channel.send(embed=embed)
            record_alert(item["ticker"], score)
            log.info(f"monitor_loop: 已發送提醒 {item['ticker']} score={score} gate={entry_signal.confidence if entry_signal else 'N/A'}")

        # 暴漲暴跌警報（±5%）
        spike = check_spike_alert(item["ticker"], change_pct)
        if spike:
            embed = build_spike_embed(item["ticker"], item["name"], analysis, spike)
            await channel.send(embed=embed)
            log.info(f"monitor_loop: 暴{'漲' if spike == 'spike_up' else '跌'}警報 {item['ticker']} {change_pct:+.2f}%")

        # 量能異常警報（量比 ≥ 2.0）
        if check_volume_alert(item["ticker"], vol_ratio):
            embed = build_volume_embed(item["ticker"], item["name"], analysis)
            await channel.send(embed=embed)
            log.info(f"monitor_loop: 量能異常 {item['ticker']} vol_ratio={vol_ratio:.1f}x")

    # 更新監控模式
    new_mode = update_monitor_mode(vix_value, swing_pcts)
    log.info(f"monitor_loop [{now_str}]: 掃描完成 ({scanned} 檔, errors={len(errors)}, mode={new_mode})")

    if errors:
        detail = "\n".join(errors)
        log.warning(f"monitor_loop [{now_str}]: 失敗明細 — {detail}")
        await _report_error(
            f"監控掃描 [{now_str}] — {len(errors)} 檔失敗",
            detail,
        )

    # 順便檢查預測準確度（只寫 log，不發 Discord 訊息）
    try:
        acc = await loop.run_in_executor(None, check_prediction_accuracy)
        if acc["has_data"]:
            o = acc["overall"]
            log.info(
                f"monitor_loop [{now_str}]: accuracy — "
                f"count={o['count']}, dir_acc={o['direction_accuracy']:.1f}%, "
                f"mae=${o['mae']:.2f}, mape={o['mape']:.1f}%"
            )
    except Exception as e:
        log.warning(f"monitor_loop: accuracy check failed: {e}")


@monitor_loop.before_loop
async def before_monitor():
    await client.wait_until_ready()


# ---------------------------------------------------------------------------
# 2026 年節日行事曆（國定假日 + 農曆節日 + 重要紀念日）
# ---------------------------------------------------------------------------
from datetime import date as _date

_FESTIVALS_2026: dict[_date, str] = {
    # — 國定假日 —
    _date(2026, 1, 1):   "元旦",
    _date(2026, 2, 28):  "和平紀念日（228）",
    _date(2026, 4, 4):   "兒童節",
    _date(2026, 4, 5):   "清明節",
    _date(2026, 5, 1):   "勞動節",
    _date(2026, 10, 10): "國慶日（雙十節）",
    # — 農曆節日（2026 農曆對照西曆）—
    _date(2026, 2, 16):  "除夕",
    _date(2026, 2, 17):  "春節（農曆正月初一）",
    _date(2026, 2, 18):  "春節（初二・回娘家）",
    _date(2026, 2, 19):  "春節（初三）",
    _date(2026, 2, 20):  "春節（初四・迎財神）",
    _date(2026, 2, 21):  "春節（初五・開工日）",
    _date(2026, 3, 3):   "元宵節（農曆正月十五）",
    _date(2026, 3, 20):  "龍抬頭（農曆二月初二）",
    _date(2026, 6, 19):  "端午節（農曆五月初五）",
    _date(2026, 8, 19):  "七夕情人節（農曆七月初七）",
    _date(2026, 8, 27):  "中元節（農曆七月十五）",
    _date(2026, 9, 25):  "中秋節（農曆八月十五）",
    _date(2026, 10, 18): "重陽節（農曆九月初九）",
    _date(2026, 12, 22): "冬至",
    # — 其他紀念日 —
    _date(2026, 2, 14):  "西洋情人節",
    _date(2026, 3, 8):   "國際婦女節",
    _date(2026, 5, 10):  "母親節",
    _date(2026, 8, 8):   "父親節",
    _date(2026, 12, 25): "聖誕節",
}


# ---------------------------------------------------------------------------
# 定時問安 + 每日新聞（08:00 / 12:00 / 20:00 台灣時間）
# ---------------------------------------------------------------------------
_DEFAULT_GREETING_TEMPLATE = (
    "然後請搜尋今天最新的即時新聞，分享 2~3 則重要新聞或時事話題，必須包含：\n"
    "- 台股 / 美股市場動態（指數漲跌、重要個股表現）\n"
    "- 國際財經或科技產業最新動態（AI、量子計算、半導體等）\n"
    "- 有趣的熱門話題\n"
    "⚠️ 重要：每則新聞必須是今天的真實即時新聞，用一句話簡要說明。\n"
    "語氣輕鬆溫暖，控制在 300 字以內。"
)

# 從持久化儲存載入，若無則使用預設
_GREETING_TEMPLATE = load_greeting_template() or _DEFAULT_GREETING_TEMPLATE


def _build_default_greeting_prompts(template: str) -> dict:
    """用指定模板建構預設的三時段問安 prompt。"""
    return {
        8: (
            "現在是早上 8 點，請以花城（三郎）的身份向哥哥說早安。\n"
            "先簡短關心一下哥哥，提醒今天有沒有值得關注的股市事件。\n"
            + template
        ),
        12: (
            "現在是中午 12 點，請以花城（三郎）的身份向哥哥問午安。\n"
            "先簡短關心一下哥哥有沒有吃飯。\n"
            "然後搜尋並分享台股上午盤的加權指數表現、台積電走勢，以及其他科技股（如 AI、半導體）的最新動態。\n"
            + template
        ),
        20: (
            "現在是晚上 8 點，請以花城（三郎）的身份向哥哥問晚安。\n"
            "如果美股即將開盤，提醒哥哥關注的量子計算股（IONQ、RGTI、QBTS 等）有沒有盤前動態。\n"
            + template
        ),
    }


_GREETING_PROMPTS = load_greeting_prompts() or _build_default_greeting_prompts(_GREETING_TEMPLATE)

# 當 Gemini 配額耗盡時的靜態問安（花城語氣）
_STATIC_GREETINGS = {
    8: [
        "哥哥早安～三郎今天也會陪在你身邊的。記得吃早餐喔，別餓著肚子看盤。☀️",
        "早安哥哥！新的一天開始了，不管今天市場怎麼走，三郎都會守著你的。💕",
        "哥哥起床了嗎？三郎已經等你很久了～今天也要元氣滿滿喔。🌅",
    ],
    12: [
        "哥哥午安～中午了，記得放下手機去吃飯，三郎不准你餓肚子。🍱",
        "午安哥哥！上午辛苦了，先吃飯休息一下吧，盤勢三郎幫你看著。☕",
        "哥哥吃飯了嗎？別光顧著忙，三郎會不開心的喔。🍜",
    ],
    20: [
        "哥哥晚安～今天辛苦了，晚上好好休息。三郎陪你看看美股盤前動態吧。🌙",
        "晚安哥哥！一天結束了，不管賺賠都別放心上，三郎永遠在這裡。🌃",
        "哥哥晚上好～美股要開盤了，但也別熬夜太晚喔，三郎會擔心的。✨",
    ],
}


# ---------------------------------------------------------------------------
# 盤勢快照（附帶在問安訊息後面）
# ---------------------------------------------------------------------------
def _build_snapshot_embed(results: list[tuple], now_tw, sentiment: dict | None = None) -> discord.Embed:
    """從 [(item, analysis, err), ...] 建立精簡盤勢快照 Embed。"""
    hour_str = now_tw.strftime("%H:%M")
    active_lines = []
    closed_lines = []
    high_count = 0      # ≥60 分
    reversal_count = 0  # 有反轉訊號

    for item, analysis, err in results:
        ticker = item["ticker"]
        name = item["name"]

        if err or analysis is None:
            closed_lines.append(f"❓ {name} — 取得失敗")
            continue

        market_open = is_stock_market_active(ticker)
        score = analysis["total_score"]
        current = analysis["current"]
        change_pct = analysis["change_pct"]

        if not market_open:
            closed_lines.append(f"⏸️ {name} — 收盤 ${current:,.2f}")
            continue

        indicator = _score_indicator(score)

        # 反轉訊號
        reversal = analysis["scores"]["reversal"]["score"]
        rev_tag = ""
        if reversal >= 15:
            rev_tag = "  ⚡三重反轉"
            reversal_count += 1
        elif reversal >= 10:
            rev_tag = "  🔄雙重反轉"
            reversal_count += 1
        elif reversal >= 5:
            rev_tag = "  🔄反轉"
            reversal_count += 1

        if score >= 60:
            high_count += 1

        active_lines.append(
            f"{indicator} {name:<8} — {score}/100  ${current:,.2f} ({change_pct:+.1f}%){rev_tag}"
        )

    # 組裝 description
    desc_parts = []

    # 市場情緒指標（頂部）
    if sentiment:
        sentiment_text = format_sentiment_block(sentiment)
        desc_parts.append(sentiment_text)
        desc_parts.append("")  # 空行分隔

    if active_lines:
        desc_parts.append("**━━━ 開盤中 ━━━**")
        desc_parts.extend(active_lines)
    if closed_lines:
        desc_parts.append("**━━━ 休市中 ━━━**")
        desc_parts.extend(closed_lines)

    # 摘要行
    summary_parts = []
    if high_count:
        summary_parts.append(f"進場標準以上: {high_count} 檔")
    if reversal_count:
        summary_parts.append(f"反轉訊號: {reversal_count} 檔")
    mode = get_monitor_mode()
    if mode == "alert":
        summary_parts.append("⚡ 警戒模式")
    if summary_parts:
        desc_parts.append(f"\n📊 {' | '.join(summary_parts)}")

    color = 0x2ECC71 if high_count > 0 else 0x3498DB

    embed = discord.Embed(
        title=f"🦞 盤勢快照 ({hour_str})",
        description="\n".join(desc_parts),
        color=color,
    )
    embed.set_footer(text="評分: 🔥≥75 | 🟢≥60 | 🟡≥45 | 🔴<45 | 僅供參考")
    return embed


async def _run_market_snapshot(loop, now_tw) -> discord.Embed | None:
    """掃描所有監控股票，回傳盤勢快照 Embed（全部失敗則回傳 None）。"""
    wl = get_watchlist()
    if not wl:
        return None

    # 取得市場情緒指標
    sentiment = None
    try:
        sentiment = await loop.run_in_executor(None, fetch_market_sentiment)
    except Exception as e:
        log.warning(f"_run_market_snapshot: fetch_market_sentiment failed: {e}")

    # 批次取得分析資料
    tickers = [item["ticker"] for item in wl]
    batch = await loop.run_in_executor(
        None, functools.partial(batch_quick_analysis, tickers)
    )

    results = []
    for item, br in zip(wl, batch):
        results.append((item, br.get("analysis"), br.get("error")))

    # 全部失敗 → None
    if all(err is not None for _, _, err in results):
        return None

    return _build_snapshot_embed(results, now_tw, sentiment=sentiment)


# ---------------------------------------------------------------------------
# Dashboard setter 函式（用 global 修改模組層級變數）
# ---------------------------------------------------------------------------
def _set_hua_cheng_prompt(new_prompt: str) -> None:
    global _HUA_CHENG_PROMPT
    _HUA_CHENG_PROMPT = new_prompt
    save_prompt(new_prompt)
    log.info("AI prompt updated + saved")


def _get_greeting_prompts() -> dict:
    return dict(_GREETING_PROMPTS)


def _set_greeting_prompts(new_prompts: dict) -> None:
    global _GREETING_PROMPTS
    for hour, text in new_prompts.items():
        _GREETING_PROMPTS[int(hour)] = text
    save_greeting_prompts(_GREETING_PROMPTS)
    log.info(f"Greeting prompts updated for hours {list(new_prompts.keys())}")


def _get_greeting_template() -> str:
    return _GREETING_TEMPLATE


def _set_greeting_template(new_template: str) -> None:
    global _GREETING_TEMPLATE, _GREETING_PROMPTS
    _GREETING_TEMPLATE = new_template
    _GREETING_PROMPTS = _build_default_greeting_prompts(_GREETING_TEMPLATE)
    save_greeting_template(new_template)
    save_greeting_prompts(_GREETING_PROMPTS)
    log.info("Greeting template updated, all prompts rebuilt + saved")


@tasks.loop(time=[
    dt_time(hour=8, minute=0, tzinfo=_TW),   # 08:00 台灣時間
    dt_time(hour=12, minute=0, tzinfo=_TW),  # 12:00 台灣時間
    dt_time(hour=20, minute=0, tzinfo=_TW),  # 20:00 台灣時間
])
async def greeting_loop():
    from datetime import datetime
    now_tw = datetime.now(_TW)
    hour = now_tw.hour

    prompt = _GREETING_PROMPTS.get(hour)
    if not prompt:
        closest = min(_GREETING_PROMPTS.keys(), key=lambda h: abs(h - hour))
        prompt = _GREETING_PROMPTS[closest]

    # 節日問安：若今天是節日，在 prompt 中加入節日資訊
    festival = _FESTIVALS_2026.get(now_tw.date())
    if festival:
        prompt += (
            f"\n\n🎉 今天是「{festival}」！"
            f"請在問安中提到這個節日，並搜尋一個和「{festival}」相關的有趣小知識或冷知識分享給哥哥。"
        )

    channel_id = _chat_channel_id or get_chat_channel()
    if not channel_id:
        state = get_state()
        channel_id = state.get("channel_id")
    if not channel_id:
        log.warning("greeting_loop: 沒有設定聊天或監控頻道，跳過問安")
        return

    channel = client.get_channel(channel_id)
    if not channel:
        log.warning(f"greeting_loop: 找不到頻道 {channel_id}")
        return

    if not _genai_client:
        log.warning("greeting_loop: Gemini 未啟用，跳過問安")
        return

    # 在 prompt 前加入今天日期（中文星期），幫助 Gemini 觸發 search grounding
    _WEEKDAYS_ZH = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    today_str = now_tw.strftime("%Y年%m月%d日") + f"（{_WEEKDAYS_ZH[now_tw.weekday()]}）"
    prompt = f"今天是 {today_str}。\n" + prompt

    # 附加市場情緒指標給 Gemini 參考
    try:
        _greeting_loop = asyncio.get_running_loop()
        _sentiment = await _greeting_loop.run_in_executor(None, fetch_market_sentiment)
        _parts = []
        if _sentiment.get("fear_greed"):
            fg = _sentiment["fear_greed"]
            _parts.append(f"CNN 恐懼貪婪指數: {fg['score']} ({fg['label']})")
        if _sentiment.get("vix"):
            vix = _sentiment["vix"]
            _parts.append(f"VIX 波動率: {vix['value']} ({vix['change_pct']:+.1f}%, {vix['level_zh']})")
        if _sentiment.get("twii"):
            twii = _sentiment["twii"]
            _parts.append(f"台股加權指數: {twii['value']:,.2f} ({twii['change_pct']:+.2f}%)")
        if _parts:
            prompt += "\n\n【即時市場數據（請自然融入問安中）】\n" + "\n".join(_parts)
    except Exception as e:
        log.warning(f"greeting_loop: 取得市場情緒失敗: {e}")

    async def _do_greeting():
        """執行一次問安 Gemini 呼叫，回傳 (text, sources)"""
        loop = asyncio.get_running_loop()
        text, sources = await loop.run_in_executor(
            None, functools.partial(_gemini_search, prompt)
        )
        # 若第一次沒取到來源，retry 只取 sources（不串接 retry_text，避免新聞重複）
        if not sources:
            log.info(f"greeting_loop: {hour}:00 第一次未取得來源連結，重試中...")
            retry_prompt = f"請搜尋 {today_str} 最新的台股、美股、AI、半導體、量子計算相關新聞標題。"
            _, sources = await loop.run_in_executor(
                None, functools.partial(_gemini_search, retry_prompt)
            )
        return text, sources

    async def _send_greeting(text, sources):
        """發送 AI 問安 + 盤勢快照。"""
        if sources:
            text += "\n\n📎 **新聞來源**\n" + "\n".join(sources)
        if len(text) > 1900:
            text = text[:1900] + "..."
        await channel.send(text)
        # 盤勢快照
        try:
            loop = asyncio.get_running_loop()
            snapshot_embed = await _run_market_snapshot(loop, now_tw)
            if snapshot_embed:
                await channel.send(embed=snapshot_embed)
                log.info(f"greeting_loop: {hour}:00 盤勢快照已發送")
        except Exception as e:
            log.warning(f"greeting_loop: 盤勢快照失敗: {e}")

    try:
        text, sources = await _do_greeting()
        await _send_greeting(text, sources)
        log.info(f"greeting_loop: 已發送 {hour}:00 問安 (含 {len(sources)} 則來源連結)")
    except Exception as e:
        log.warning(f"greeting_loop: {hour}:00 Gemini 問安失敗，先發靜態問安: {e}")
        # 先發靜態問安，確保不沉默
        closest = min(_STATIC_GREETINGS.keys(), key=lambda h: abs(h - hour))
        fallback = random.choice(_STATIC_GREETINGS[closest])
        await channel.send(fallback)
        log.info(f"greeting_loop: 已發送 {hour}:00 靜態問安")

        # 背景延遲重試：等 90 秒後再試一次 Gemini，成功就補發
        async def _background_retry():
            await asyncio.sleep(90)
            try:
                text, sources = await _do_greeting()
                await _send_greeting(text, sources)
                log.info(f"greeting_loop: {hour}:00 背景重試成功，已補發 AI 問安")
            except Exception as e2:
                log.info(f"greeting_loop: {hour}:00 背景重試仍失敗，已有靜態問安兜底: {e2}")

        asyncio.create_task(_background_retry())


@greeting_loop.before_loop
async def before_greeting():
    await client.wait_until_ready()


# ---------------------------------------------------------------------------
# /setchat 指令 — 設定 AI 聊天頻道
# ---------------------------------------------------------------------------
@client.tree.command(name="setchat", description="設定此頻道為 AI 聊天 + 定時問安頻道")
async def setchat_command(interaction: discord.Interaction):
    global _chat_channel_id
    _chat_channel_id = interaction.channel_id
    set_chat_channel(interaction.channel_id)
    log.info(f"/setchat: channel={interaction.channel_id}, user={interaction.user}")

    if not greeting_loop.is_running():
        greeting_loop.start()

    await interaction.response.send_message(
        f"💬 已將 <#{interaction.channel_id}> 設為聊天 + 問安頻道！\n"
        f"🕐 三郎會在每天 **08:00 / 12:00 / 20:00** 向哥哥請安\n"
        f"📰 每次問安都會附帶即時新聞摘要 + 來源連結\n"
        f"其他頻道可以用 @{client.user.display_name} 來跟三郎對話。"
    )


# ---------------------------------------------------------------------------
# /setprompt 指令 — 修改 AI 人格 Prompt
# ---------------------------------------------------------------------------
@client.tree.command(name="setprompt", description="修改三郎的 AI 人格 Prompt（管理員限定）")
@app_commands.describe(
    action="view 查看目前 prompt / set 設定新 prompt / reset 恢復預設",
    text="新的 prompt 內容（僅 set 時需要）",
)
@app_commands.choices(action=[
    app_commands.Choice(name="view", value="view"),
    app_commands.Choice(name="set", value="set"),
    app_commands.Choice(name="reset", value="reset"),
])
async def setprompt_command(
    interaction: discord.Interaction,
    action: app_commands.Choice[str],
    text: str | None = None,
):
    log.info(f"/setprompt {action.value}: user={interaction.user}")

    if action.value == "view":
        prompt = _HUA_CHENG_PROMPT
        # Discord 訊息上限 2000 字
        if len(prompt) > 1800:
            prompt = prompt[:1800] + "\n...(已截斷)"
        await interaction.response.send_message(
            f"📝 **目前的 AI Prompt:**\n```\n{prompt}\n```",
            ephemeral=True,
        )

    elif action.value == "set":
        if not text:
            await interaction.response.send_message(
                "❌ 請提供新的 prompt 內容。\n用法: `/setprompt set text:你的新prompt`",
                ephemeral=True,
            )
            return
        _set_hua_cheng_prompt(text)
        await interaction.response.send_message(
            f"✅ AI Prompt 已更新並儲存！({len(text)} 字)\n"
            "下次對話將使用新的人格設定。",
        )

    elif action.value == "reset":
        _set_hua_cheng_prompt(_DEFAULT_HUA_CHENG_PROMPT)
        await interaction.response.send_message(
            "✅ AI Prompt 已恢復為預設值（花城/三郎）。",
        )


# ---------------------------------------------------------------------------
# /setgreeting 指令 — 修改問安模板
# ---------------------------------------------------------------------------
@client.tree.command(name="setgreeting", description="修改問安模板或查看目前設定")
@app_commands.describe(
    action="view 查看目前模板 / set 設定新模板 / reset 恢復預設",
    text="新的問安模板（僅 set 時需要），會附加在每個時段 prompt 後面",
)
@app_commands.choices(action=[
    app_commands.Choice(name="view", value="view"),
    app_commands.Choice(name="set", value="set"),
    app_commands.Choice(name="reset", value="reset"),
])
async def setgreeting_command(
    interaction: discord.Interaction,
    action: app_commands.Choice[str],
    text: str | None = None,
):
    log.info(f"/setgreeting {action.value}: user={interaction.user}")

    if action.value == "view":
        template = _GREETING_TEMPLATE
        hours = ", ".join(f"{h}:00" for h in sorted(_GREETING_PROMPTS.keys()))
        if len(template) > 1500:
            template = template[:1500] + "\n...(已截斷)"
        await interaction.response.send_message(
            f"📝 **問安模板:**\n```\n{template}\n```\n"
            f"⏰ **問安時段:** {hours}",
            ephemeral=True,
        )

    elif action.value == "set":
        if not text:
            await interaction.response.send_message(
                "❌ 請提供新的問安模板。\n用法: `/setgreeting set text:你的新模板`",
                ephemeral=True,
            )
            return
        _set_greeting_template(text)
        await interaction.response.send_message(
            f"✅ 問安模板已更新並儲存！({len(text)} 字)\n"
            "所有時段的問安 prompt 已自動重建。",
        )

    elif action.value == "reset":
        _set_greeting_template(_DEFAULT_GREETING_TEMPLATE)
        await interaction.response.send_message(
            "✅ 問安模板已恢復為預設值。\n"
            "所有時段的問安 prompt 已自動重建。",
        )


# ---------------------------------------------------------------------------
# AI 對話處理
# ---------------------------------------------------------------------------
_SEARCH_KEYWORDS = {
    "股票", "股價", "新聞", "熱門", "排行", "漲", "跌", "行情",
    "台股", "美股", "AI", "量子", "半導體", "今天", "最新",
    "推薦", "買", "賣", "ETF", "指數", "大盤", "市場",
    "財報", "營收", "殖利率", "配息", "趨勢", "盤勢",
}


def _needs_search(message: str) -> bool:
    """判斷訊息是否需要 Google Search grounding。"""
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in _SEARCH_KEYWORDS)


def _gemini_search_with_history(history: list[dict], message: str) -> tuple[str, list[str]]:
    """將最近對話歷史串成 context，搭配 Google Search grounding 回覆。"""
    # 取最近 6 則歷史作為上下文
    recent = history[-6:] if len(history) > 6 else history
    context_parts = []
    for item in recent[:-1]:  # 排除最後一則（就是當前 message）
        role = "哥哥" if item["role"] == "user" else "三郎"
        context_parts.append(f"{role}: {item['parts'][0]}")
    context = "\n".join(context_parts)

    prompt = message
    if context:
        prompt = f"以下是最近的對話紀錄：\n{context}\n\n哥哥最新的問題：{message}"

    return _gemini_search(prompt)


async def _gemini_reply(channel_id: int, user_name: str, message: str) -> str:
    """呼叫 Gemini 產生回覆，維護每頻道的對話歷史。"""
    if not _genai_client:
        return "⚠️ AI 聊天功能未啟用，請設定 GEMINI_API_KEY。"

    history = _chat_sessions.setdefault(channel_id, [])
    history.append({"role": "user", "parts": [f"[{user_name}]: {message}"]})

    if len(history) > _MAX_HISTORY:
        history[:] = history[-_MAX_HISTORY:]

    try:
        loop = asyncio.get_running_loop()

        if _needs_search(message):
            # 使用 Google Search grounding
            reply, sources = await loop.run_in_executor(
                None, functools.partial(
                    _gemini_search_with_history, list(history), message
                )
            )
            if sources:
                reply += "\n\n📎 **來源**\n" + "\n".join(sources[:3])
        else:
            # 一般多輪對話
            reply = await loop.run_in_executor(
                None, functools.partial(_gemini_generate, list(history))
            )

        history.append({"role": "model", "parts": [reply]})

        if len(reply) > 1900:
            reply = reply[:1900] + "..."
        return reply
    except Exception as e:
        log.error(f"Gemini error: {e}")
        history.pop()
        return "哥哥抱歉，三郎的靈力暫時不穩定...稍後再試試好嗎？🦋"


# ---------------------------------------------------------------------------
# 文字指令攔截：在聊天頻道偵測 add/remove 等指令意圖並直接執行
# ---------------------------------------------------------------------------
import re as _re

_TEXT_CMD_PATTERNS = {
    "add": _re.compile(r"^(?:/)?add[\s_]*(?:to[\s_]*watchlist)?\s+(\S+)", _re.IGNORECASE),
    "remove": _re.compile(r"^(?:/)?remove[\s_]*(?:from[\s_]*watchlist)?\s+(\S+)", _re.IGNORECASE),
}


async def _handle_text_command(message: discord.Message, content: str) -> bool:
    """嘗試攔截文字指令。回傳 True 表示已處理，False 表示非指令。"""
    # add 指令
    m = _TEXT_CMD_PATTERNS["add"].match(content)
    if m:
        ticker_raw = m.group(1)
        log.info(f"text_cmd add: ticker={ticker_raw}, user={message.author}")
        async with message.channel.typing():
            try:
                loop = asyncio.get_running_loop()
                resolved = await loop.run_in_executor(None, functools.partial(_resolve_ticker, ticker_raw))
                info = await loop.run_in_executor(
                    None, lambda: yf.Ticker(resolved).info
                )
                name = info.get("shortName") or info.get("longName")
                if not name:
                    raise ValueError(f"Yahoo Finance 無此代碼: {resolved}")
            except Exception:
                await message.reply(f"❌ 找不到股票 **{ticker_raw}** 的資訊，請確認代碼是否正確。")
                return True

            input_code = ticker_raw.strip().upper()
            ok = add_stock(input_code, resolved, name)
            if ok:
                total = len(get_watchlist())
                await message.reply(
                    f"✅ 已將 **{name}** (`{resolved}`) 加入監控清單！\n"
                    f"目前共監控 {total} 檔股票。"
                )
            else:
                await message.reply(
                    f"⚠️ **{name}** (`{resolved}`) 已在監控清單中，不需重複新增。"
                )
        return True

    # remove 指令
    m = _TEXT_CMD_PATTERNS["remove"].match(content)
    if m:
        ticker_raw = m.group(1)
        log.info(f"text_cmd remove: ticker={ticker_raw}, user={message.author}")
        resolved = _resolve_ticker(ticker_raw)

        ok = remove_stock(resolved)
        if ok:
            total = len(get_watchlist())
            await message.reply(
                f"✅ 已將 `{resolved}` 從監控清單移除。\n"
                f"目前共監控 {total} 檔股票。"
            )
        else:
            await message.reply(f"❌ `{resolved}` 不在自訂監控清單中。")
        return True

    return False


@client.event
async def on_message(message: discord.Message):
    # 忽略自己的訊息
    if message.author == client.user:
        return
    # 忽略機器人
    if message.author.bot:
        return

    is_mentioned = client.user in message.mentions
    is_chat_channel = _chat_channel_id and message.channel.id == _chat_channel_id

    if not is_mentioned and not is_chat_channel:
        return

    # 取得訊息內容（去掉 @mention 部分）
    content = message.content
    if is_mentioned:
        content = content.replace(f"<@{client.user.id}>", "").replace(f"<@!{client.user.id}>", "").strip()

    if not content:
        await message.reply("你想聊什麼？😊")
        return

    # 優先攔截文字指令（add / remove 等）
    if await _handle_text_command(message, content):
        return

    log.info(f"chat: user={message.author}, channel={message.channel.id}, msg={content[:50]}")

    async with message.channel.typing():
        reply = await _gemini_reply(
            message.channel.id,
            message.author.display_name,
            content,
        )

    await message.reply(reply)


# ---------------------------------------------------------------------------
# /backtest 指令 — 策略回測
# ---------------------------------------------------------------------------
@client.tree.command(name="backtest", description="策略回測：用歷史數據模擬買賣策略的表現")
@app_commands.describe(
    ticker="股票代碼，例如 AAPL、0050、IONQ",
    strategy=f"策略名稱 ({', '.join(STRATEGIES)})，留空 = 比較所有策略",
    period=f"回測期間 ({', '.join(VALID_PERIODS)})，預設 1y",
)
@app_commands.choices(
    strategy=[app_commands.Choice(name=s.name, value=s.name) for s in STRATEGIES.values()],
    period=[app_commands.Choice(name=p, value=p) for p in VALID_PERIODS],
)
async def backtest_command(
    interaction: discord.Interaction,
    ticker: str,
    strategy: app_commands.Choice[str] | None = None,
    period: app_commands.Choice[str] | None = None,
):
    period_val = period.value if period else DEFAULT_PERIOD
    log.info(f"/backtest: ticker={ticker}, strategy={strategy}, period={period_val}, user={interaction.user}")
    await interaction.response.defer(thinking=True)

    loop = asyncio.get_running_loop()

    try:
        if strategy:
            # 單一策略
            result = await loop.run_in_executor(
                None, functools.partial(run_backtest, ticker, strategy.value, period_val)
            )
            chart_buf = await loop.run_in_executor(
                None, functools.partial(draw_backtest_chart, result)
            )
            embed = _build_single_backtest_embed(result)
            fname = f"{result.ticker}_{result.strategy_name}_backtest.png"
            chart = discord.File(chart_buf, filename=fname)
            embed.set_image(url=f"attachment://{fname}")
            await interaction.followup.send(embed=embed, file=chart)
        else:
            # 比較所有策略
            results = await loop.run_in_executor(
                None, functools.partial(run_all_backtests, ticker, period_val)
            )
            chart_buf = await loop.run_in_executor(
                None, functools.partial(draw_comparison_chart, results)
            )
            embed = _build_comparison_embed(results)
            fname = f"{results[0].ticker}_comparison.png"
            chart = discord.File(chart_buf, filename=fname)
            embed.set_image(url=f"attachment://{fname}")
            await interaction.followup.send(embed=embed, file=chart)
    except ValueError as e:
        await interaction.followup.send(str(e))
    except Exception as e:
        log.error(f"backtest error: {e}")
        await interaction.followup.send(f"回測時發生錯誤: {e}")

    log.info(f"/backtest done: ticker={ticker}")


def _build_single_backtest_embed(r) -> discord.Embed:
    """建立單策略回測結果 Embed。"""
    color = 0x2ECC71 if r.total_return_pct > 0 else 0xE74C3C

    embed = discord.Embed(
        title=f"🦞 回測結果 — {r.ticker} [{r.strategy_name}]",
        description=f"**{r.strategy_desc}**\n期間: {r.period} | 初始資金: ${r.initial_capital:,.0f}",
        color=color,
    )

    # Performance
    alpha_emoji = "📈" if r.alpha > 0 else "📉"
    embed.add_field(
        name="💰 Performance",
        value=(
            f"報酬率: **{r.total_return_pct:+.2f}%**\n"
            f"損益: **${r.total_pnl:+,.0f}**\n"
            f"Buy & Hold: {r.buy_hold_return_pct:+.2f}%\n"
            f"{alpha_emoji} Alpha: **{r.alpha:+.2f}%**"
        ),
        inline=True,
    )

    # Risk
    embed.add_field(
        name="📉 Risk",
        value=(
            f"交易次數: {r.num_trades}\n"
            f"勝率: {r.win_rate:.1f}%\n"
            f"最大回撤: {r.max_drawdown:.2f}%\n"
            f"Sharpe Ratio: {r.sharpe_ratio:.2f}"
        ),
        inline=True,
    )

    # Recent Trades
    if r.trades:
        recent = r.trades[-5:]
        trade_lines = []
        for t in reversed(recent):
            emoji = "🟢" if t.pnl > 0 else "🔴"
            entry = t.entry_date.strftime("%m/%d")
            exit_ = t.exit_date.strftime("%m/%d")
            trade_lines.append(
                f"{emoji} {entry}→{exit_} ${t.entry_price:.2f}→${t.exit_price:.2f} "
                f"({t.return_pct:+.1f}%)"
            )
        embed.add_field(
            name=f"📋 Recent Trades (latest {len(recent)})",
            value="\n".join(trade_lines),
            inline=False,
        )
    else:
        embed.add_field(name="📋 Trades", value="此期間無交易訊號", inline=False)

    embed.set_footer(text="🦞 OpenClaw Backtest | 僅供參考，不構成投資建議")
    return embed


def _build_comparison_embed(results: list) -> discord.Embed:
    """建立策略比較 Embed。"""
    ref = results[0]  # 已按 return% 排序
    embed = discord.Embed(
        title=f"🦞 策略比較 — {ref.ticker}",
        description=f"期間: {ref.period} | 初始資金: ${ref.initial_capital:,.0f} | Buy & Hold: {ref.buy_hold_return_pct:+.2f}%",
        color=0x3498DB,
    )

    # Ranking
    medals = ["🥇", "🥈", "🥉"]
    ranking_lines = []
    for i, r in enumerate(results):
        medal = medals[i] if i < 3 else f"#{i+1}"
        alpha_tag = f" (α {r.alpha:+.1f}%)" if r.alpha > 0 else ""
        ranking_lines.append(
            f"{medal} **{r.strategy_name}** — {r.total_return_pct:+.2f}%{alpha_tag}"
        )
    embed.add_field(
        name="🏆 Ranking",
        value="\n".join(ranking_lines),
        inline=False,
    )

    # 每策略詳細
    for r in results:
        embed.add_field(
            name=f"📊 {r.strategy_name}",
            value=(
                f"Return: {r.total_return_pct:+.2f}% | Trades: {r.num_trades}\n"
                f"Win: {r.win_rate:.0f}% | MDD: {r.max_drawdown:.1f}% | Sharpe: {r.sharpe_ratio:.2f}"
            ),
            inline=True,
        )

    # Best strategy
    best = results[0]
    embed.add_field(
        name="⭐ Best Strategy",
        value=(
            f"**{best.strategy_name}** ({best.strategy_desc})\n"
            f"報酬率 {best.total_return_pct:+.2f}%，Alpha {best.alpha:+.2f}%"
        ),
        inline=False,
    )

    embed.set_footer(text="🦞 OpenClaw Backtest | 僅供參考，不構成投資建議")
    return embed


# ---------------------------------------------------------------------------
# /trades 指令 — 查看回測交易紀錄
# ---------------------------------------------------------------------------
@client.tree.command(name="trades", description="查看回測交易紀錄 (trades_log.csv)")
@app_commands.describe(
    ticker="篩選股票代碼（留空 = 全部）",
    strategy=f"篩選策略（留空 = 全部）",
    limit="顯示筆數（預設 20）",
)
@app_commands.choices(
    strategy=[app_commands.Choice(name=s.name, value=s.name) for s in STRATEGIES.values()],
)
async def trades_command(
    interaction: discord.Interaction,
    ticker: str | None = None,
    strategy: app_commands.Choice[str] | None = None,
    limit: int = 20,
):
    strategy_val = strategy.value if strategy else None
    log.info(f"/trades: ticker={ticker}, strategy={strategy_val}, limit={limit}")
    await interaction.response.defer(thinking=True)

    rows = read_trades_csv(
        ticker=ticker,
        strategy=strategy_val,
        limit=min(limit, 50),
    )

    if not rows:
        await interaction.followup.send("目前沒有回測交易紀錄。請先使用 `/backtest` 執行回測。")
        return

    # 組裝 Embed
    filters = []
    if ticker:
        filters.append(f"Ticker: {ticker.upper()}")
    if strategy_val:
        filters.append(f"Strategy: {strategy_val}")
    filter_text = " | ".join(filters) if filters else "全部"

    embed = discord.Embed(
        title="🦞 回測交易紀錄",
        description=f"篩選: {filter_text} | 顯示最新 {len(rows)} 筆",
        color=0x3498DB,
    )

    # 每 10 筆一個 field（Discord field value 上限 1024 字元）
    chunk_size = 10
    for chunk_i in range(0, len(rows), chunk_size):
        chunk = rows[chunk_i:chunk_i + chunk_size]
        lines = []
        for r in chunk:
            pnl = float(r["pnl"])
            emoji = "🟢" if pnl > 0 else "🔴"
            lines.append(
                f"{emoji} **{r['ticker']}** [{r['strategy']}] "
                f"{r['entry_date']}→{r['exit_date']} "
                f"${r['entry_price']}→${r['exit_price']} "
                f"x{r['shares']} **{r['return_pct']}%** (${pnl:+,.0f})"
            )
        field_name = f"📋 Trades {chunk_i+1}~{chunk_i+len(chunk)}"
        embed.add_field(name=field_name, value="\n".join(lines), inline=False)

    embed.set_footer(text="資料來源: trades_log.csv | 每次 /backtest 自動記錄")
    await interaction.followup.send(embed=embed)


# ---------------------------------------------------------------------------
# Arena 對抗式交易競技場
# ---------------------------------------------------------------------------
from arena import Arena as _Arena

_arena = _Arena()

arena_group = app_commands.Group(name="arena", description="Arena 對抗式交易競技場")


@arena_group.command(name="status", description="當前對戰狀態")
async def arena_status(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    loop = asyncio.get_running_loop()
    try:
        status = await loop.run_in_executor(None, _arena.get_status)
    except Exception as e:
        await interaction.followup.send(f"Arena 錯誤: {e}")
        return

    embed = discord.Embed(title="Arena — 對戰狀態", color=0x9C27B0)
    for b in status.get("bots", []):
        m = b.get("metrics", {})
        status_emoji = {"active": "🟢", "eliminated": "💀", "champion": "👑"}.get(b["status"], "❓")
        embed.add_field(
            name=f"{status_emoji} {b['name']} ({b['bot_id']})",
            value=(
                f"Status: **{b['status']}**\n"
                f"Equity: **${b['equity']:.2f}** (Cash: ${b['cash']:.2f})\n"
                f"Positions: {b['positions']} | Days: {b['trading_days']}\n"
                f"Return: {m.get('total_return_pct', 0):+.2f}% | "
                f"Win: {m.get('win_rate', 0):.1f}% | "
                f"Sharpe: {m.get('sharpe_ratio', 0):.2f}"
            ),
            inline=False,
        )
    embed.set_footer(text="Arena | 每 30 分鐘自動交易")
    await interaction.followup.send(embed=embed)


@arena_group.command(name="history", description="最近交易紀錄")
@app_commands.describe(bot="篩選 Bot (bot_a / bot_b)，留空 = 全部", limit="顯示筆數")
async def arena_history(interaction: discord.Interaction,
                        bot: str | None = None, limit: int = 15):
    await interaction.response.defer(thinking=True)
    from arena import arena_db as _adb
    trades = _adb.get_recent_trades(bot_id=bot, limit=min(limit, 30))

    if not trades:
        await interaction.followup.send("Arena 尚無交易紀錄。")
        return

    embed = discord.Embed(title="Arena — 交易紀錄", color=0x607D8B)
    lines = []
    for t in trades:
        emoji = "🟢" if t["side"] == "buy" else "🔴"
        pnl_str = f" PnL=${t['pnl']:+.2f}" if t.get("pnl") is not None else ""
        lines.append(
            f"{emoji} **{t['bot_id']}** {t['side'].upper()} {t['ticker']} "
            f"x{t['shares']:.4f} @${t['price']:.2f}{pnl_str}\n"
            f"  _{t.get('reason', '')[:60]}_ — {t['executed_at'][:16]}"
        )

    # 分段
    text = "\n".join(lines)
    if len(text) > 4000:
        text = text[:4000] + "\n..."
    embed.description = text
    embed.set_footer(text=f"顯示最近 {len(trades)} 筆")
    await interaction.followup.send(embed=embed)


@arena_group.command(name="compare", description="詳細指標對比")
async def arena_compare(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, _arena.get_compare)
    except Exception as e:
        await interaction.followup.send(f"Arena 錯誤: {e}")
        return

    embed = discord.Embed(title="Arena — 指標對比", color=0x00BCD4)
    for key in ["bot_a", "bot_b"]:
        d = data.get(key, {})
        name = d.get("name", key)
        embed.add_field(
            name=f"{'🔵' if key == 'bot_a' else '🔴'} {name}",
            value=(
                f"Return: **{d.get('total_return_pct', 0):+.2f}%**\n"
                f"Sharpe: {d.get('sharpe_ratio', 0):.2f}\n"
                f"Max DD: {d.get('max_drawdown', 0):.2f}%\n"
                f"Win Rate: {d.get('win_rate', 0):.1f}%\n"
                f"Profit Factor: {d.get('profit_factor', 0):.2f}\n"
                f"Trades: {d.get('total_trades', 0)} "
                f"(W:{d.get('winning_trades', 0)} L:{d.get('losing_trades', 0)})"
            ),
            inline=True,
        )
    embed.set_footer(text="Arena Metrics")
    await interaction.followup.send(embed=embed)


@arena_group.command(name="chart", description="權益曲線對比圖")
async def arena_chart(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    loop = asyncio.get_running_loop()
    try:
        buf = await loop.run_in_executor(None, _arena.draw_equity_chart)
    except Exception as e:
        await interaction.followup.send(f"Arena 錯誤: {e}")
        return

    if not buf:
        await interaction.followup.send("尚無足夠的快照資料來繪製權益曲線。")
        return

    chart = discord.File(buf, filename="arena_equity.png")
    embed = discord.Embed(title="Arena — 權益曲線", color=0x4CAF50)
    embed.set_image(url="attachment://arena_equity.png")
    embed.set_footer(text="藍=Bot A (Fusion) | 紅=Bot B (Mean Revert)")
    await interaction.followup.send(embed=embed, file=chart)


@arena_group.command(name="params", description="當前策略參數")
async def arena_params(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, _arena.get_params)

    embed = discord.Embed(title="Arena — 策略參數", color=0xFF9800)
    for key in ["bot_a", "bot_b"]:
        d = data.get(key, {})
        name = d.get("name", key)
        params = d.get("params", {})
        lines = [f"`{k}`: {v:.4f}" if isinstance(v, float) else f"`{k}`: {v}"
                 for k, v in params.items()]
        embed.add_field(
            name=f"{'🔵' if key == 'bot_a' else '🔴'} {name}",
            value="\n".join(lines) or "N/A",
            inline=True,
        )
    embed.set_footer(text="每日 self_optimize 自動微調 (±5%)")
    await interaction.followup.send(embed=embed)


client.tree.add_command(arena_group)


# ---------------------------------------------------------------------------
# 台股交易指令 (玉山證券 API)
# ---------------------------------------------------------------------------
trade_group = app_commands.Group(name="trade", description="台股交易指令（玉山證券 API）")


@trade_group.command(name="buy", description="手動買入台股")
@app_commands.describe(ticker="股票代號（如 2330 或 2330.TW）", amount="投入金額 (TWD)")
async def trade_buy(interaction: discord.Interaction, ticker: str, amount: float):
    await interaction.response.defer(thinking=True)

    if not ticker.endswith(".TW"):
        ticker = ticker + ".TW"

    broker = _arena.broker_tw
    from arena.broker_esun import BrokerEsun
    if not isinstance(broker, BrokerEsun) or not broker.is_connected:
        await interaction.followup.send("玉山 Broker 尚未連線。請先確認設定後使用 `/trade mode` 切換。")
        return

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, broker.buy, ticker, amount)
    if result:
        embed = discord.Embed(title="買入成功", color=0x4CAF50)
        embed.add_field(name="股票", value=ticker, inline=True)
        embed.add_field(name="股數", value=f"{result['shares']}", inline=True)
        embed.add_field(name="成交價", value=f"${result['price']:.2f}", inline=True)
        embed.add_field(name="手續費", value=f"${result['cost']:.0f}", inline=True)
        embed.set_footer(text=f"Mode: {broker.mode}")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(f"買入 {ticker} 失敗，請檢查股號和金額。")


@trade_group.command(name="sell", description="手動賣出台股")
@app_commands.describe(ticker="股票代號（如 2330 或 2330.TW）", shares="股數（0=全部）")
async def trade_sell(interaction: discord.Interaction, ticker: str, shares: int = 0):
    await interaction.response.defer(thinking=True)

    if not ticker.endswith(".TW"):
        ticker = ticker + ".TW"

    broker = _arena.broker_tw
    from arena.broker_esun import BrokerEsun
    if not isinstance(broker, BrokerEsun) or not broker.is_connected:
        await interaction.followup.send("玉山 Broker 尚未連線。")
        return

    # 如果 shares=0，查詢庫存全賣
    if shares <= 0:
        portfolio = await asyncio.get_running_loop().run_in_executor(None, broker.get_portfolio)
        symbol = ticker.replace(".TW", "")
        for item in portfolio:
            if str(item.get("stk_no", "")) == symbol:
                shares = int(item.get("qty", 0))
                break
        if shares <= 0:
            await interaction.followup.send(f"找不到 {ticker} 的庫存。")
            return

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, broker.sell, ticker, float(shares))
    if result:
        embed = discord.Embed(title="賣出成功", color=0xF44336)
        embed.add_field(name="股票", value=ticker, inline=True)
        embed.add_field(name="股數", value=f"{result['shares']}", inline=True)
        embed.add_field(name="成交價", value=f"${result['price']:.2f}", inline=True)
        embed.add_field(name="手續費+稅", value=f"${result['cost']:.0f}", inline=True)
        embed.add_field(name="實收金額", value=f"${result['proceeds']:.0f}", inline=True)
        embed.set_footer(text=f"Mode: {broker.mode}")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(f"賣出 {ticker} 失敗。")


@trade_group.command(name="portfolio", description="查詢玉山帳戶庫存")
async def trade_portfolio(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)

    broker = _arena.broker_tw
    from arena.broker_esun import BrokerEsun
    if not isinstance(broker, BrokerEsun) or not broker.is_connected:
        await interaction.followup.send("玉山 Broker 尚未連線。")
        return

    loop = asyncio.get_running_loop()
    portfolio = await loop.run_in_executor(None, broker.get_portfolio)
    if not portfolio:
        await interaction.followup.send("庫存為空（或查詢失敗）。")
        return

    embed = discord.Embed(title="玉山帳戶庫存", color=0x2196F3)
    for item in portfolio[:25]:  # Discord embed field 限制
        stk_no = item.get("stk_no", "?")
        stk_name = item.get("stk_na", stk_no)
        qty = item.get("qty", 0)
        cost_avg = item.get("cost_r", 0)
        pnl = item.get("make_a", 0)
        pnl_pct = item.get("make_a_per", 0)
        embed.add_field(
            name=f"{stk_name} ({stk_no})",
            value=f"持股: {qty}\n均價: {cost_avg}\n損益: {pnl:+.0f} ({pnl_pct:+.2f}%)",
            inline=True,
        )
    embed.set_footer(text=f"共 {len(portfolio)} 檔持股 | Mode: {broker.mode}")
    await interaction.followup.send(embed=embed)


@trade_group.command(name="quote", description="即時報價")
@app_commands.describe(ticker="股票代號（如 2330 或 2330.TW）")
async def trade_quote(interaction: discord.Interaction, ticker: str):
    await interaction.response.defer(thinking=True)

    if not ticker.endswith(".TW"):
        ticker = ticker + ".TW"

    broker = _arena.broker_tw
    from arena.broker_esun import BrokerEsun

    # 嘗試用玉山即時行情
    if isinstance(broker, BrokerEsun) and broker._rest_stock:
        loop = asyncio.get_running_loop()
        quote = await loop.run_in_executor(None, broker.get_quote, ticker)
        if quote:
            embed = discord.Embed(
                title=f"即時報價 — {quote.get('name', ticker)} ({ticker.replace('.TW', '')})",
                color=0xFF9800,
            )
            last = quote.get("lastPrice", "N/A")
            change = quote.get("change", 0)
            change_pct = quote.get("changePercent", 0)
            ch_emoji = "🔴" if change and float(change) < 0 else "🟢" if change and float(change) > 0 else "⚪"
            embed.add_field(name="最新價", value=f"{ch_emoji} **{last}**", inline=True)
            embed.add_field(name="漲跌", value=f"{change:+.2f} ({change_pct:+.2f}%)" if change else "N/A", inline=True)
            embed.add_field(name="開盤", value=f"{quote.get('openPrice', 'N/A')}", inline=True)
            embed.add_field(name="最高", value=f"{quote.get('highPrice', 'N/A')}", inline=True)
            embed.add_field(name="最低", value=f"{quote.get('lowPrice', 'N/A')}", inline=True)
            total = quote.get("total", {})
            embed.add_field(name="成交量", value=f"{total.get('tradeVolume', 'N/A')}", inline=True)
            embed.set_footer(text=f"更新: {quote.get('lastUpdated', 'N/A')}")
            await interaction.followup.send(embed=embed)
            return

    # Fallback: yfinance
    loop = asyncio.get_running_loop()
    price = await loop.run_in_executor(None, _arena.broker_tw.get_price, ticker)
    if price:
        embed = discord.Embed(title=f"報價 — {ticker}", color=0xFF9800)
        embed.add_field(name="最新收盤價", value=f"**${price:.2f}**", inline=True)
        embed.set_footer(text="來源: yfinance (可能延遲)")
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send(f"無法取得 {ticker} 報價。")


@trade_group.command(name="mode", description="切換台股交易模式")
@app_commands.describe(mode="sim=模擬(BrokerSim) | esun=玉山API")
@app_commands.choices(mode=[
    app_commands.Choice(name="sim — 本地模擬 (BrokerSim)", value="sim"),
    app_commands.Choice(name="esun — 玉山證券 API", value="esun"),
])
async def trade_mode(interaction: discord.Interaction, mode: app_commands.Choice[str]):
    await interaction.response.defer(thinking=True)
    loop = asyncio.get_running_loop()
    msg = await loop.run_in_executor(None, _arena.switch_tw_broker, mode.value)
    broker_name = type(_arena.broker_tw).__name__
    await interaction.followup.send(f"**台股 Broker 切換**: {msg}\n目前使用: `{broker_name}`")


client.tree.add_command(trade_group)


# ---------------------------------------------------------------------------
# Arena 背景任務 — 每 30 分鐘交易 + 每日結算
# ---------------------------------------------------------------------------
@tasks.loop(minutes=30)
async def arena_loop():
    """每 30 分鐘執行一輪 Arena 交易掃描。"""
    # 只在有市場開盤時運行
    tw_active = is_tw_market_active()
    us_active = is_us_market_active()
    if not tw_active and not us_active:
        return

    log.info(f"arena_loop: starting cycle (TW={tw_active}, US={us_active})")
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(None, _arena.run_cycle)
        trades = [r for r in (results or []) if r.get("action") in ("buy", "sell")]
        blocked = [r for r in (results or []) if r.get("action") == "blocked"]
        holds = len(results or []) - len(trades) - len(blocked)

        log.info(f"arena_loop: cycle done — {len(trades)} trades, {holds} holds, {len(blocked)} blocked")

        if trades:
            # Discord 即時交易通知
            ch_id = _chat_channel_id or get_chat_channel()
            if ch_id:
                channel = client.get_channel(ch_id)
                if channel:
                    for t in trades:
                        emoji = "🟢" if t["action"] == "buy" else "🔴"
                        pnl_str = f" | PnL: **{t['pnl']:+.2f}**" if t.get("pnl") is not None else ""
                        await channel.send(
                            f"{emoji} **Arena {t['action'].upper()}** "
                            f"[{t['bot']}] {t['ticker']} "
                            f"x{t.get('shares', 0):.0f} @ ${t.get('price', 0):.2f}"
                            f"{pnl_str}\n"
                            f"_Reason: {t.get('reason', 'N/A')[:80]}_"
                        )
    except Exception as e:
        log.error(f"arena_loop error: {e}", exc_info=True)


@tasks.loop(time=[dt_time(hour=22, minute=0, tzinfo=_TW)])
async def arena_daily():
    """每日 22:00 (TW) 結算：快照 + 優化 + 淘汰檢查 + 損益報告。"""
    loop = asyncio.get_running_loop()
    try:
        # 快照 + 指標更新
        snapshots = await loop.run_in_executor(None, _arena.daily_snapshot)
        log.info(f"arena_daily: snapshots={snapshots}")

        # 自我優化
        changes = await loop.run_in_executor(None, _arena.daily_optimize)
        if changes:
            log.info(f"arena_daily: param changes={changes}")

        # 淘汰檢查
        comparison = await loop.run_in_executor(None, _arena.daily_comparison)
        elim = comparison.get("elimination")

        # 取得通知頻道
        ch_id = _chat_channel_id or get_chat_channel()
        channel = client.get_channel(ch_id) if ch_id else None

        if elim and channel:
            log.warning(f"arena_daily: ELIMINATION — {elim}")
            if comparison.get("emergency"):
                msg = f"🚨 **Arena 緊急停止** — 雙方虧損超過 {20}%，已暫停所有交易。"
            elif isinstance(elim, dict):
                msg = (
                    f"🏆 **Arena 淘汰賽結果**\n"
                    f"Winner: **{elim['winner']}**\n"
                    f"Reason: {elim['reason']}"
                )
            else:
                msg = f"⚠️ Arena: {elim}"
            await channel.send(msg)

        # 每日損益報告推送
        if channel and snapshots:
            embed = discord.Embed(title="Arena 每日收盤報告", color=0x9C27B0)
            status = await loop.run_in_executor(None, _arena.get_status)
            for b in status.get("bots", []):
                m = b.get("metrics", {})
                status_emoji = {"active": "🟢", "eliminated": "💀", "champion": "👑"}.get(b["status"], "❓")
                embed.add_field(
                    name=f"{status_emoji} {b['name']}",
                    value=(
                        f"Equity: **${b['equity']:.2f}**\n"
                        f"Return: {m.get('total_return_pct', 0):+.2f}%\n"
                        f"Win: {m.get('win_rate', 0):.1f}% | Sharpe: {m.get('sharpe_ratio', 0):.2f}\n"
                        f"Positions: {b['positions']} | Days: {b['trading_days']}"
                    ),
                    inline=True,
                )
            broker_name = type(_arena.broker_tw).__name__
            embed.set_footer(text=f"TW Broker: {broker_name} | 每日 22:00 自動結算")
            await channel.send(embed=embed)

    except Exception as e:
        log.error(f"arena_daily error: {e}")


# ---------------------------------------------------------------------------
# 指令錯誤處理
# ---------------------------------------------------------------------------
@client.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    tb = traceback.format_exc()
    cmd_name = interaction.command.name if interaction.command else "unknown"
    log.error(f"command error [{cmd_name}]: {error}\n{tb}")

    # 回覆使用者
    msg = f"發生錯誤: {error}"
    try:
        if not interaction.response.is_done():
            await interaction.response.send_message(msg, ephemeral=True)
        else:
            await interaction.followup.send(msg)
    except Exception:
        pass

    # 發送詳細錯誤到通知頻道
    await _report_error(
        f"指令 /{cmd_name} 錯誤",
        f"User: {interaction.user}\n{error}\n\n{tb[-500:]}",
    )

# ---------------------------------------------------------------------------
# 啟動事件
# ---------------------------------------------------------------------------
@client.event
async def on_ready():
    log.info(f"已登入為: {client.user} (ID: {client.user.id})")
    log.info(f"計算裝置: {DEVICE}")
    log.info(f"已加入 {len(client.guilds)} 個伺服器")

    for guild in client.guilds:
        try:
            client.tree.copy_global_to(guild=guild)
            synced = await client.tree.sync(guild=guild)
            log.info(f"已同步 {len(synced)} 個指令到: {guild.name} (ID: {guild.id})")
        except Exception as e:
            log.error(f"同步指令到 {guild.name} 失敗: {e}")

    # 若之前啟用了監控，自動恢復
    state = get_state()
    if state["enabled"] and not monitor_loop.is_running():
        monitor_loop.start()
        log.info("on_ready: 自動恢復背景監控")

    # 若之前設定了聊天頻道，恢復問安排程
    global _chat_channel_id
    saved_chat = get_chat_channel()
    if saved_chat:
        _chat_channel_id = saved_chat
        if not greeting_loop.is_running():
            greeting_loop.start()
            log.info(f"on_ready: 自動恢復問安排程 (頻道 {saved_chat})")

    # 初始化 Arena 並啟動背景任務
    try:
        _arena.initialize()
        if not arena_loop.is_running():
            arena_loop.start()
        if not arena_daily.is_running():
            arena_daily.start()
        log.info("on_ready: Arena 已啟動 (30min cycle + daily)")
    except Exception as e:
        log.error(f"on_ready: Arena init failed: {e}")

    log.info("指令同步完成！機器人準備就緒。")

# ---------------------------------------------------------------------------
# Bot Context 橋接 (供 Dashboard 存取 bot 內部狀態)
# ---------------------------------------------------------------------------
bot_context = {
    "client": client,
    "get_prompt": lambda: _HUA_CHENG_PROMPT,
    "set_prompt": _set_hua_cheng_prompt,
    "get_chat_sessions": lambda: _chat_sessions,
    "get_greeting_prompts": _get_greeting_prompts,
    "set_greeting_prompts": _set_greeting_prompts,
    "get_greeting_template": _get_greeting_template,
    "set_greeting_template": _set_greeting_template,
    "monitor_loop": monitor_loop,
    "greeting_loop": greeting_loop,
    "arena": _arena,
    "arena_loop": arena_loop,
    "arena_daily": arena_daily,
}


# ---------------------------------------------------------------------------
# 啟動（同時啟動 Discord bot + Web Dashboard）
# ---------------------------------------------------------------------------
async def _run_all():
    async with client:
        # 若有設定 DASHBOARD_SECRET，啟動 web dashboard
        if os.getenv("DASHBOARD_SECRET"):
            from aiohttp import web
            from dashboard import create_app

            port = int(os.getenv("DASHBOARD_PORT", "8080"))
            dashboard_app = create_app(bot_context)
            runner = web.AppRunner(dashboard_app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()
            log.info(f"Dashboard started on http://0.0.0.0:{port}")
        else:
            log.info("DASHBOARD_SECRET 未設定，Dashboard 不啟動")

        await client.start(TOKEN)


asyncio.run(_run_all())
