"""
花城 Performance Report — QuantStats 績效報表
基於 predictions_log.csv 產出績效統計 + tearsheet PNG。
import 失敗 → 回傳 None。
"""

import io
import os
import csv
import logging
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

_log = logging.getLogger("huacheng.perf_report")

_TW = timezone(timedelta(hours=8))
_PRED_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.csv")

# ---------------------------------------------------------------------------
# QuantStats import（graceful fallback）
# ---------------------------------------------------------------------------
_HAS_QS = False
try:
    import quantstats as qs
    _HAS_QS = True
    _log.info("perf_report: quantstats loaded")
except ImportError:
    _log.warning("perf_report: quantstats not installed")


def _load_prediction_returns() -> pd.Series | None:
    """
    從 predictions_log.csv 讀取預測記錄，計算每筆預測的預期日報酬率。
    回傳 pd.Series（index=date, values=daily_return），或 None。
    """
    if not os.path.exists(_PRED_LOG):
        _log.warning("perf_report: predictions_log.csv not found")
        return None

    with open(_PRED_LOG, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return None

    # 策略邏輯：若預測 Day 7 > last_price（看漲），持有部位 → 用實際報酬率衡量
    # 若預測看跌，不持有 → 報酬率為 0
    # 這模擬的是「依預測訊號做多」的假設策略
    import yfinance as yf

    # 收集所有預測記錄（需要 ticker + predict_time + last_price + pred_day7）
    signals = []
    for row in rows:
        try:
            pt = datetime.strptime(row["predict_time"], "%Y-%m-%d %H:%M:%S")
            last_price = float(row.get("last_price", 0))
            pred_day7 = float(row.get("pred_day7", 0))
            ticker = row.get("ticker", "")
            if last_price > 0 and pred_day7 > 0 and ticker:
                signals.append({
                    "date": pt.date(),
                    "ticker": ticker,
                    "last_price": last_price,
                    "pred_day7": pred_day7,
                    "signal": 1 if pred_day7 > last_price else 0,  # 1=看漲, 0=看跌
                })
        except (ValueError, KeyError):
            continue

    if not signals:
        return None

    # 按日期聚合信號（同一天多筆取平均信號強度）
    sig_df = pd.DataFrame(signals)

    # 為了計算實際報酬率，我們用 SPY 作為基準（簡化：不逐 ticker 追蹤）
    # 若預測看漲，假設持有市場 → 用 SPY 的實際日報酬率
    # 若預測看跌，不持有 → 報酬率 0
    try:
        spy = yf.download("SPY", period="1y", interval="1d", progress=False)
        if spy.empty:
            return None
        spy_close = spy["Close"].squeeze()
        spy_returns = spy_close.pct_change().dropna()
        spy_returns.index = spy_returns.index.normalize()
    except Exception:
        # SPY 下載失敗時，退回到用預測報酬率（比沒有好）
        daily_pred = sig_df.groupby("date").apply(
            lambda g: ((g["pred_day7"] - g["last_price"]) / g["last_price"]).mean()
        )
        daily_pred.index = pd.to_datetime(daily_pred.index)
        return daily_pred.sort_index()

    # 每日信號強度（0~1），用來乘以實際市場報酬
    daily_signal = sig_df.groupby("date")["signal"].mean()
    daily_signal.index = pd.to_datetime(daily_signal.index)

    # 對齊日期：信號日 T 的報酬率取 T+1 的 SPY 實際報酬（隔日持有）
    returns_data = []
    for sig_date, sig_strength in daily_signal.items():
        next_days = spy_returns.index[spy_returns.index > sig_date]
        if len(next_days) == 0:
            continue
        next_day = next_days[0]
        actual_return = float(spy_returns.loc[next_day])
        strategy_return = sig_strength * actual_return  # 按信號比例持有
        returns_data.append({"date": next_day, "return": strategy_return})

    if not returns_data:
        return None

    result_df = pd.DataFrame(returns_data)
    daily = result_df.groupby("date")["return"].mean()
    daily.index = pd.to_datetime(daily.index)
    return daily.sort_index()


def generate_tearsheet_png() -> io.BytesIO | None:
    """
    產出 QuantStats tearsheet PNG。
    回傳 BytesIO（PNG 圖片），或 None（失敗時）。
    """
    if not _HAS_QS:
        _log.warning("perf_report: quantstats not available")
        return None

    returns = _load_prediction_returns()
    if returns is None or len(returns) < 5:
        _log.warning("perf_report: insufficient data for tearsheet")
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # QuantStats snapshot plot
        fig = qs.plots.snapshot(returns, title="HuaCheng Prediction Performance", show=False)
        if fig is None:
            # Some versions return None and create current figure
            fig = plt.gcf()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)

        _log.info("perf_report: tearsheet PNG generated")
        return buf

    except Exception as e:
        _log.warning(f"perf_report: tearsheet generation failed ({e})")
        return None


def generate_stats_text() -> str | None:
    """
    產出文字版績效統計摘要。
    回傳格式化字串，或 None。
    """
    if not _HAS_QS:
        return None

    returns = _load_prediction_returns()
    if returns is None or len(returns) < 5:
        return None

    try:
        sharpe = qs.stats.sharpe(returns)
        sortino = qs.stats.sortino(returns)
        max_dd = qs.stats.max_drawdown(returns)
        total_return = qs.stats.comp(returns)
        win_rate = qs.stats.win_rate(returns)
        avg_win = qs.stats.avg_win(returns)
        avg_loss = qs.stats.avg_loss(returns)

        text = (
            f"📊 **花城預測績效報表**\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"累積報酬: **{total_return:.2%}**\n"
            f"Sharpe Ratio: **{sharpe:.2f}**\n"
            f"Sortino Ratio: **{sortino:.2f}**\n"
            f"最大回撤: **{max_dd:.2%}**\n"
            f"勝率: **{win_rate:.1%}**\n"
            f"平均獲利: **{avg_win:.2%}**\n"
            f"平均虧損: **{avg_loss:.2%}**\n"
            f"交易天數: **{len(returns)}**\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
        return text

    except Exception as e:
        _log.warning(f"perf_report: stats generation failed ({e})")
        return None
