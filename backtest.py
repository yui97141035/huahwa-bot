"""
花城 — 策略回測引擎
可擴展的 Strategy 框架，內建 8 種策略，支援台股/美股交易成本模型。
"""

import csv
import io
import logging
import os
import ssl
import certifi
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path

os.environ.setdefault("SSL_CERT_FILE", certifi.where())

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from prediction import _resolve_ticker, compute_analysis, _yf_download
from ta_modules import (
    compute_ta_overlay, SupportResistance, ChartPatterns, MACrossFilter,
    SignalConfidence, NoTradeZone, ProfitTarget,
)
from watchlist import _TW

# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    pnl: float          # 扣除手續費後的淨損益
    cost: float          # 手續費總額
    return_pct: float    # 單筆報酬率 %


@dataclass
class CostModel:
    buy_fee_pct: float   # 買入手續費 %
    sell_fee_pct: float  # 賣出手續費 % (含證交稅)

    def buy_cost(self, amount: float) -> float:
        return amount * self.buy_fee_pct / 100

    def sell_cost(self, amount: float) -> float:
        return amount * self.sell_fee_pct / 100


# 台股: 買 0.1425%, 賣 0.1425% + 0.3% 證交稅
TW_COST = CostModel(buy_fee_pct=0.1425, sell_fee_pct=0.4425)
# 美股: 買賣各 0.1% 滑價估算
US_COST = CostModel(buy_fee_pct=0.1, sell_fee_pct=0.1)


@dataclass
class BacktestResult:
    ticker: str
    strategy_name: str
    strategy_desc: str
    period: str
    initial_capital: float
    final_equity: float
    total_pnl: float
    total_return_pct: float
    num_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    buy_hold_return_pct: float
    alpha: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    dates: pd.DatetimeIndex = field(default_factory=lambda: pd.DatetimeIndex([]))
    prices: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# 策略基類
# ---------------------------------------------------------------------------

class Strategy(ABC):
    name: str = ""
    description: str = ""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """回傳與 df 同長的 Series: 1=買, -1=賣, 0=持有"""
        ...


# ---------------------------------------------------------------------------
# 內建策略
# ---------------------------------------------------------------------------

class RSI6Strategy(Strategy):
    name = "RSI_6"
    description = "RSI(6) < 20 買入, > 80 賣出"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"].squeeze()
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(6).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        signals = pd.Series(0, index=df.index)
        signals[rsi < 20] = 1
        signals[rsi > 80] = -1
        return signals


class MA3_100Strategy(Strategy):
    name = "MA_3_100"
    description = "MA3 黃金交叉 MA100 買, 死亡交叉賣"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"].squeeze()
        ma3 = close.rolling(3).mean()
        ma100 = close.rolling(100).mean()

        signals = pd.Series(0, index=df.index)
        # 黃金交叉: MA3 由下穿上 MA100
        cross_up = (ma3 > ma100) & (ma3.shift(1) <= ma100.shift(1))
        # 死亡交叉: MA3 由上穿下 MA100
        cross_down = (ma3 < ma100) & (ma3.shift(1) >= ma100.shift(1))
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals


class MomentumStrategy(Strategy):
    name = "MOM"
    description = "10 日動量由負轉正買, 由正轉負賣"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"].squeeze()
        mom = close - close.shift(10)

        signals = pd.Series(0, index=df.index)
        # 由負轉正
        cross_up = (mom > 0) & (mom.shift(1) <= 0)
        # 由正轉負
        cross_down = (mom < 0) & (mom.shift(1) >= 0)
        signals[cross_up] = 1
        signals[cross_down] = -1
        return signals


class HuaChengStrategy(Strategy):
    name = "花城"
    description = "花城評分 ≥60 買, <30 賣"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        # 需要至少 60 根 K 線才能計算完整指標
        min_bars = 60
        for i in range(min_bars, len(df)):
            sub_df = df.iloc[:i + 1]
            try:
                analysis = compute_analysis(sub_df)
                score = analysis["total_score"]
                if score >= 60:
                    signals.iloc[i] = 1
                elif score < 30:
                    signals.iloc[i] = -1
            except Exception as e:
                _log.debug(f"HuaCheng signal gen failed at bar {i}: {e}")
        return signals


class SRFlipStrategy(Strategy):
    name = "SR_Flip"
    description = "支撐壓力翻轉：壓力變支撐買入, 支撐跌破賣出"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        min_bars = 60
        for i in range(min_bars, len(df)):
            sub_df = df.iloc[:i + 1]
            try:
                sr = SupportResistance.analyze(sub_df)
                levels = sr.get("levels", [])
                flipped = [lv for lv in levels if lv.get("flipped") and lv.get("type") == "support"]
                close_val = float(sub_df["Close"].values.flatten()[-1])
                # 有翻轉支撐且價格在上方 → 買入
                if flipped:
                    signals.iloc[i] = 1
                # 若所有支撐都跌破 → 賣出
                elif levels:
                    supports = [lv for lv in levels if lv.get("type") == "support"]
                    if not supports:
                        signals.iloc[i] = -1
            except Exception:
                pass
        return signals


class PatternStrategy(Strategy):
    name = "Pattern"
    description = "形態辨識：看漲形態買入, 看跌形態賣出"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=df.index)
        min_bars = 60
        for i in range(min_bars, len(df)):
            sub_df = df.iloc[:i + 1]
            try:
                pattern = ChartPatterns.analyze(sub_df)
                if pattern.get("signal") == "bullish" and pattern.get("strength", 0) >= 3:
                    signals.iloc[i] = 1
                elif pattern.get("signal") == "bearish" and pattern.get("strength", 0) >= 3:
                    signals.iloc[i] = -1
            except Exception:
                pass
        return signals


class MAFilterStrategy(Strategy):
    name = "MA_Filter"
    description = "MA50/200 金叉買入, 死叉賣出 (含趨勢過濾)"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"].squeeze()
        signals = pd.Series(0, index=df.index)

        if len(df) < 200:
            return signals

        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # 金叉 (MA50 > MA200 且前一日 MA50 <= MA200)
        cross_up = (ma50 > ma200) & (ma50.shift(1) <= ma200.shift(1))
        # 死叉
        cross_down = (ma50 < ma200) & (ma50.shift(1) >= ma200.shift(1))

        # 趨勢過濾：只在 MA200 持平或上升時買入
        ma200_slope = ma200 - ma200.shift(20)
        valid_buy = cross_up & (ma200_slope >= 0)

        signals[valid_buy] = 1
        signals[cross_down] = -1
        return signals


class SignalGradeStrategy(Strategy):
    name = "SigGrade"
    description = "訊號分級策略：S/A級買入, 避險區暫停, 移動停利出場"

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        整合模組 6+7+8：
        - NoTradeZone 禁止時不開倉
        - SignalConfidence S/A 級才買入
        - ProfitTarget 強勢用 MA10 trailing stop, 弱勢用固定 R/R
        """
        close = df["Close"].squeeze()
        signals = pd.Series(0, index=df.index)
        min_bars = 60

        in_position = False
        entry_price = 0.0
        trailing_stop = False
        target_rr = 1.5
        ma10 = close.rolling(10).mean()

        for i in range(min_bars, len(df)):
            sub_df = df.iloc[:i + 1]
            price = float(close.iloc[i])

            try:
                # 出場邏輯（在持倉時）
                if in_position:
                    # 移動停利：跌破 MA10
                    if trailing_stop and not pd.isna(ma10.iloc[i]):
                        if price < float(ma10.iloc[i]):
                            signals.iloc[i] = -1
                            in_position = False
                            continue

                    # 固定停利：達到目標 R/R
                    if not trailing_stop and entry_price > 0:
                        gain_pct = (price - entry_price) / entry_price * 100
                        if gain_pct >= target_rr * 2:  # R/R target (assume 2% risk)
                            signals.iloc[i] = -1
                            in_position = False
                            continue

                    # 停損：跌超過 5%
                    if entry_price > 0 and price < entry_price * 0.95:
                        signals.iloc[i] = -1
                        in_position = False
                        continue

                    continue  # 持倉中，不重複買入

                # 入場邏輯
                notrade = NoTradeZone.analyze(sub_df)
                if not notrade.get("trade_allowed", True):
                    continue  # 禁止交易區，跳過

                pattern = ChartPatterns.analyze(sub_df)
                sr = SupportResistance.analyze(sub_df)
                from ta_modules import RetestConfirmation, KLineSentiment
                kline = KLineSentiment.analyze(sub_df)
                retest = RetestConfirmation.analyze(sub_df, kline, sr, pattern)
                confidence = SignalConfidence.analyze(pattern, retest)

                grade = confidence.get("grade", "none")
                action = confidence.get("action", "hold")

                # 只在 S 或 A 級買入訊號時進場
                if grade in ("S", "A") and action in ("strong_buy", "buy"):
                    signals.iloc[i] = 1
                    in_position = True
                    entry_price = price

                    # 決定出場策略
                    profit = ProfitTarget.analyze(pattern, sr)
                    exit_strat = profit.get("exit_strategy", "fixed_target")
                    trailing_stop = exit_strat in ("trailing_stop", "hybrid")
                    target_rr = profit.get("target_rr", 1.5) or 1.5

                # S 級賣出訊號（如果未持倉也標記，給持倉者參考）
                elif grade == "S" and action in ("strong_sell",):
                    signals.iloc[i] = -1

            except Exception:
                pass

        return signals


# 策略註冊表
STRATEGIES: dict[str, Strategy] = {
    s.name: s for s in [
        RSI6Strategy(),
        MA3_100Strategy(),
        MomentumStrategy(),
        HuaChengStrategy(),
        SRFlipStrategy(),
        PatternStrategy(),
        MAFilterStrategy(),
        SignalGradeStrategy(),
    ]
}

VALID_PERIODS = ["6mo", "1y", "2y", "5y"]
DEFAULT_PERIOD = "1y"
INITIAL_CAPITAL = 100_000.0


# ---------------------------------------------------------------------------
# 回測引擎
# ---------------------------------------------------------------------------

class Backtester:
    def __init__(self, cost_model: CostModel, initial_capital: float = INITIAL_CAPITAL):
        self.cost_model = cost_model
        self.initial_capital = initial_capital

    def run(self, df: pd.DataFrame, signals: pd.Series,
            ticker: str, strategy_name: str, strategy_desc: str,
            period: str) -> BacktestResult:
        close = df["Close"].squeeze().values.astype(float)
        dates = df.index

        cash = self.initial_capital
        shares = 0
        entry_price = 0.0
        entry_date = None

        trades: list[Trade] = []
        equity = np.zeros(len(close))

        for i in range(len(close)):
            price = close[i]
            sig = signals.iloc[i]

            # 買入: 持有現金且訊號 = 1
            if sig == 1 and shares == 0 and cash > 0:
                buy_amount = cash
                buy_cost = self.cost_model.buy_cost(buy_amount)
                available = buy_amount - buy_cost
                shares = int(available // price)
                if shares > 0:
                    actual_cost = shares * price
                    fee = self.cost_model.buy_cost(actual_cost)
                    cash -= actual_cost + fee
                    entry_price = price
                    entry_date = dates[i]

            # 賣出: 持有部位且訊號 = -1
            elif sig == -1 and shares > 0:
                sell_amount = shares * price
                fee = self.cost_model.sell_cost(sell_amount)
                buy_fee = self.cost_model.buy_cost(shares * entry_price)
                total_cost = buy_fee + fee
                pnl = (price - entry_price) * shares - total_cost
                ret_pct = pnl / (shares * entry_price) * 100

                trades.append(Trade(
                    entry_date=entry_date,
                    entry_price=entry_price,
                    exit_date=dates[i],
                    exit_price=price,
                    shares=shares,
                    pnl=pnl,
                    cost=total_cost,
                    return_pct=ret_pct,
                ))
                cash += sell_amount - fee
                shares = 0

            equity[i] = cash + shares * price

        # 若最後仍持有部位，以最後收盤價計算
        final_equity = cash + shares * close[-1]

        # Buy & Hold
        bh_shares = int(self.initial_capital // close[0])
        bh_buy_cost = self.cost_model.buy_cost(bh_shares * close[0])
        bh_sell_cost = self.cost_model.sell_cost(bh_shares * close[-1])
        bh_equity = (bh_shares * close[-1] - bh_sell_cost
                     + (self.initial_capital - bh_shares * close[0] - bh_buy_cost))
        bh_return = (bh_equity - self.initial_capital) / self.initial_capital * 100

        total_pnl = final_equity - self.initial_capital
        total_return = total_pnl / self.initial_capital * 100

        # Win rate
        wins = sum(1 for t in trades if t.pnl > 0)
        win_rate = (wins / len(trades) * 100) if trades else 0.0

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        # 避免除以零
        peak_safe = np.where(peak > 0, peak, 1)
        drawdown = (peak - equity) / peak_safe * 100
        max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        # Sharpe ratio (年化，假設 252 交易日)
        daily_returns = np.diff(equity) / np.where(equity[:-1] != 0, equity[:-1], 1)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))
        else:
            sharpe = 0.0

        return BacktestResult(
            ticker=ticker,
            strategy_name=strategy_name,
            strategy_desc=strategy_desc,
            period=period,
            initial_capital=self.initial_capital,
            final_equity=final_equity,
            total_pnl=total_pnl,
            total_return_pct=total_return,
            num_trades=len(trades),
            win_rate=win_rate,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            buy_hold_return_pct=bh_return,
            alpha=total_return - bh_return,
            trades=trades,
            equity_curve=pd.Series(equity, index=dates),
            dates=dates,
            prices=close,
        )


# ---------------------------------------------------------------------------
# 交易成本自動判斷
# ---------------------------------------------------------------------------

def _get_cost_model(resolved_ticker: str) -> CostModel:
    if resolved_ticker.endswith(".TW") or resolved_ticker.endswith(".TWO"):
        return TW_COST
    return US_COST


# ---------------------------------------------------------------------------
# 交易紀錄 CSV 持久化
# ---------------------------------------------------------------------------

_TRADE_LOG = Path("trades_log.csv")
_CSV_COLUMNS = [
    "run_time", "ticker", "strategy", "period",
    "entry_date", "entry_price", "exit_date", "exit_price",
    "shares", "pnl", "cost", "return_pct",
]
_log = logging.getLogger("huacheng.backtest")


def save_trades_csv(result: "BacktestResult") -> int:
    """將回測交易紀錄 append 寫入 trades_log.csv，回傳寫入筆數。"""
    if not result.trades:
        return 0

    write_header = not _TRADE_LOG.exists() or _TRADE_LOG.stat().st_size == 0
    run_time = datetime.now(_TW).strftime("%Y-%m-%d %H:%M:%S")

    with open(_TRADE_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        for t in result.trades:
            writer.writerow({
                "run_time":     run_time,
                "ticker":       result.ticker,
                "strategy":     result.strategy_name,
                "period":       result.period,
                "entry_date":   t.entry_date.strftime("%Y-%m-%d"),
                "entry_price":  f"{t.entry_price:.4f}",
                "exit_date":    t.exit_date.strftime("%Y-%m-%d"),
                "exit_price":   f"{t.exit_price:.4f}",
                "shares":       t.shares,
                "pnl":          f"{t.pnl:.2f}",
                "cost":         f"{t.cost:.2f}",
                "return_pct":   f"{t.return_pct:.2f}",
            })

    _log.info(f"trades_log.csv: +{len(result.trades)} trades ({result.ticker}/{result.strategy_name})")
    return len(result.trades)


def read_trades_csv(ticker: str | None = None,
                    strategy: str | None = None,
                    limit: int = 50) -> list[dict]:
    """讀取交易紀錄，可依 ticker / strategy 過濾，回傳最新 limit 筆。"""
    if not _TRADE_LOG.exists():
        return []
    with open(_TRADE_LOG, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if ticker:
        ticker_up = ticker.upper()
        rows = [r for r in rows if r["ticker"] == ticker_up]
    if strategy:
        rows = [r for r in rows if r["strategy"] == strategy]
    return rows[-limit:]


# ---------------------------------------------------------------------------
# 頂層 API
# ---------------------------------------------------------------------------

def run_backtest(ticker: str, strategy_name: str,
                 period: str = DEFAULT_PERIOD) -> BacktestResult:
    """單一策略回測。"""
    resolved = _resolve_ticker(ticker)
    if strategy_name not in STRATEGIES:
        raise ValueError(f"未知策略 **{strategy_name}**，可用: {', '.join(STRATEGIES)}")
    if period not in VALID_PERIODS:
        raise ValueError(f"無效期間 **{period}**，可用: {', '.join(VALID_PERIODS)}")

    df = _yf_download(resolved, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"找不到 **{resolved}** 的歷史資料")

    strategy = STRATEGIES[strategy_name]
    signals = strategy.generate_signals(df)

    cost_model = _get_cost_model(resolved)
    bt = Backtester(cost_model)
    result = bt.run(df, signals, resolved.upper(), strategy.name,
                    strategy.description, period)
    save_trades_csv(result)
    return result


def run_all_backtests(ticker: str,
                      period: str = DEFAULT_PERIOD) -> list[BacktestResult]:
    """所有策略批次回測（只下載一次數據）。"""
    resolved = _resolve_ticker(ticker)
    if period not in VALID_PERIODS:
        raise ValueError(f"無效期間 **{period}**，可用: {', '.join(VALID_PERIODS)}")

    df = _yf_download(resolved, period=period, interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"找不到 **{resolved}** 的歷史資料")

    cost_model = _get_cost_model(resolved)
    bt = Backtester(cost_model)
    results = []

    for strategy in STRATEGIES.values():
        signals = strategy.generate_signals(df)
        r = bt.run(df, signals, resolved.upper(), strategy.name,
                   strategy.description, period)
        save_trades_csv(r)
        results.append(r)

    # 依 return% 排序（高到低）
    results.sort(key=lambda r: r.total_return_pct, reverse=True)
    return results


# ---------------------------------------------------------------------------
# 圖表
# ---------------------------------------------------------------------------

def draw_backtest_chart(result: BacktestResult) -> io.BytesIO:
    """單策略圖表: 上圖價格+買賣標記, 下圖 equity curve。"""
    fig, (ax_price, ax_eq) = plt.subplots(
        2, 1, figsize=(11, 7), height_ratios=[2, 1],
        gridspec_kw={"hspace": 0.35},
    )

    dates = result.dates
    prices = result.prices

    # --- 上圖: 價格 + 買賣點 ---
    ax_price.plot(dates, prices, color="#1f77b4", linewidth=1.2, label="Close")

    for t in result.trades:
        ax_price.scatter(t.entry_date, t.entry_price, marker="^", color="green",
                         s=80, zorder=5)
        ax_price.scatter(t.exit_date, t.exit_price, marker="v", color="red",
                         s=80, zorder=5)

    # 標記圖例（只各加一次）
    if result.trades:
        ax_price.scatter([], [], marker="^", color="green", s=80, label="Buy")
        ax_price.scatter([], [], marker="v", color="red", s=80, label="Sell")

    ax_price.set_title(
        f"{result.ticker} — {result.strategy_name} Backtest ({result.period})  |  "
        f"Return: {result.total_return_pct:+.2f}%",
        fontsize=12,
    )
    ax_price.set_ylabel("Price")
    ax_price.legend(fontsize=8, loc="upper left")
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax_price.tick_params(axis="x", rotation=30)

    # --- 下圖: Equity Curve ---
    ax_eq.plot(dates, result.equity_curve.values, color="#2ca02c", linewidth=1.2,
               label=f"{result.strategy_name}")
    # Buy & Hold 基線
    bh_equity = INITIAL_CAPITAL * (1 + (prices / prices[0] - 1))
    ax_eq.plot(dates, bh_equity, color="gray", linestyle="--", linewidth=1,
               label="Buy & Hold", alpha=0.7)
    ax_eq.axhline(INITIAL_CAPITAL, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    ax_eq.set_title("Equity Curve", fontsize=11)
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.legend(fontsize=8, loc="upper left")
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax_eq.tick_params(axis="x", rotation=30)

    fig.subplots_adjust(top=0.93, bottom=0.10)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf


def draw_comparison_chart(results: list[BacktestResult]) -> io.BytesIO:
    """比較圖表: 所有策略 equity curve 疊加 + Buy & Hold 基線。"""
    fig, ax = plt.subplots(figsize=(11, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, r in enumerate(results):
        c = colors[i % len(colors)]
        ax.plot(r.dates, r.equity_curve.values, color=c, linewidth=1.3,
                label=f"{r.strategy_name} ({r.total_return_pct:+.1f}%)")

    # Buy & Hold 基線 (用第一個結果的價格資料)
    ref = results[0]
    bh_equity = INITIAL_CAPITAL * (1 + (ref.prices / ref.prices[0] - 1))
    ax.plot(ref.dates, bh_equity, color="gray", linestyle="--", linewidth=1.2,
            label=f"Buy & Hold ({ref.buy_hold_return_pct:+.1f}%)", alpha=0.7)

    ax.axhline(INITIAL_CAPITAL, color="black", linestyle=":", linewidth=0.5, alpha=0.5)

    ax.set_title(
        f"{ref.ticker} — Strategy Comparison ({ref.period})  |  "
        f"Initial: ${INITIAL_CAPITAL:,.0f}",
        fontsize=12,
    )
    ax.set_ylabel("Equity ($)")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))
    ax.tick_params(axis="x", rotation=30)

    fig.subplots_adjust(top=0.92, bottom=0.14)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf
