"""Arena 競技場調度器 — run_cycle + daily_comparison + elimination。"""

import io
import logging
from datetime import datetime, timedelta, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from . import arena_db as db
from .risk_manager import RiskManager, INITIAL_CAPITAL
from .broker_alpaca import BrokerAlpaca
from .broker_sim import BrokerSim
from .broker_esun import BrokerEsun
from .engine_a import EngineA
from .engine_b import EngineB

_log = logging.getLogger("arena")
_TW = timezone(timedelta(hours=8))

# 預設交易標的池
US_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
TW_TICKERS = [
    # ETF 打底（穩定配息，低波動）
    "0050.TW", "0056.TW", "00878.TW", "00919.TW",
    # 半導體 + 電子（價位適中，流動性好）
    "2303.TW",   # 聯電 ~57
    "2891.TW",   # 中信金 ~52（金融穩定股）
    "2882.TW",   # 國泰金 ~70
    # 航運 + 傳產（有波段空間）
    "2618.TW",   # 長榮航 ~35
    "1216.TW",   # 統一 ~72
]

# 淘汰規則
MIN_TRADING_DAYS = 30
WIN_RATE_CHAMPION = 85.0
PROFIT_FACTOR_CHAMPION = 2.0
EVALUATION_DAYS = 90
EMERGENCY_LOSS_PCT = 20.0


class Arena:
    """對抗式演算交易競技場。"""

    def __init__(self) -> None:
        self.engine_a = EngineA()
        self.engine_b = EngineB()
        self.risk = RiskManager()
        self.broker_us = BrokerAlpaca()
        # 台股 Broker: 優先使用玉山真實交易，fallback 到模擬
        self._broker_esun = BrokerEsun()
        self.broker_tw = self._broker_esun if self._broker_esun.is_connected else BrokerSim()
        self._initialized = False

    def initialize(self) -> None:
        """初始化 DB + 確保 Bot 存在 + 登入玉山 SDK。"""
        db.init_db()
        db.ensure_bot(self.engine_a.bot_id, self.engine_a.name, INITIAL_CAPITAL)
        db.ensure_bot(self.engine_b.bot_id, self.engine_b.name, INITIAL_CAPITAL)

        # 嘗試登入玉山 SDK
        if self._broker_esun._sdk and not self._broker_esun.is_connected:
            if self._broker_esun.login():
                self.broker_tw = self._broker_esun
                _log.info("Arena: using BrokerEsun for TW market")
            else:
                self.broker_tw = BrokerSim()
                _log.info("Arena: BrokerEsun login failed, fallback to BrokerSim")

        self._initialized = True
        _log.info(f"Arena initialized (TW broker: {type(self.broker_tw).__name__})")

    def switch_tw_broker(self, mode: str) -> str:
        """切換台股 broker 模式。mode: 'sim' | 'live' | 'esun'"""
        if mode == "sim":
            self.broker_tw = BrokerSim()
            return "Switched to BrokerSim (模擬)"
        elif mode in ("live", "esun"):
            if self._broker_esun.is_connected:
                self.broker_tw = self._broker_esun
                return f"Switched to BrokerEsun ({self._broker_esun.mode})"
            elif self._broker_esun.login():
                self.broker_tw = self._broker_esun
                return f"Switched to BrokerEsun ({self._broker_esun.mode})"
            else:
                return "BrokerEsun login failed — still using current broker"
        return f"Unknown mode: {mode}"

    def _get_broker(self, market: str):
        return self.broker_us if market == "US" else self.broker_tw

    # ------------------------------------------------------------------
    # 核心交易迴圈
    # ------------------------------------------------------------------

    def run_cycle(self) -> list[dict]:
        """執行一輪交易掃描（所有 active Bot × 所有標的）。"""
        if not self._initialized:
            self.initialize()

        results = []
        engines = [self.engine_a, self.engine_b]

        for engine in engines:
            bot = db.get_bot(engine.bot_id)
            if not bot or bot["status"] != "active":
                continue

            # 檢查日損限制
            if self.risk.check_daily_loss(engine.bot_id):
                results.append({
                    "bot": engine.bot_id, "action": "blocked",
                    "reason": "daily loss limit reached",
                })
                continue

            # 先檢查現有持倉的止損
            for pos in db.get_open_positions(engine.bot_id):
                broker = self._get_broker(pos["market"])
                price = broker.get_price(pos["ticker"])
                if price and self.risk.check_stop_loss(engine.bot_id, pos, price):
                    r = self._execute_sell(engine, pos, broker, price, "stop_loss")
                    if r:
                        results.append(r)

            # 掃描標的
            tickers_markets = [(t, "US") for t in US_TICKERS] + [(t, "TW") for t in TW_TICKERS]
            for ticker, market in tickers_markets:
                try:
                    decision = engine.decide(ticker, market)
                    if not decision or decision["action"] == "hold":
                        continue
                    r = self._execute_decision(engine, decision, market)
                    if r:
                        results.append(r)
                except Exception as e:
                    _log.error(f"[{engine.bot_id}] error on {ticker}: {e}")

        return results

    def _execute_decision(self, engine, decision: dict, market: str) -> dict | None:
        """執行買/賣決策。"""
        action = decision["action"]
        ticker = decision["ticker"]
        broker = self._get_broker(market)

        if action == "buy":
            return self._execute_buy(engine, decision, broker, market)
        elif action == "sell":
            pos = db.get_position_by_ticker(engine.bot_id, ticker)
            if pos:
                price = broker.get_price(ticker)
                if price:
                    return self._execute_sell(engine, pos, broker, price,
                                              decision.get("reason", "signal"))
        return None

    def _execute_buy(self, engine, decision: dict, broker, market: str) -> dict | None:
        ticker = decision["ticker"]
        cash = db.get_cash(engine.bot_id)
        if cash <= 0.01:
            return None

        # 計算投入金額
        equity = self._calc_equity(engine.bot_id)
        max_amount = self.risk.max_trade_amount(equity)
        desired = cash * decision.get("amount_pct", 50) / 100
        amount = min(desired, max_amount, cash)

        if amount < 0.01:
            return None

        if not self.risk.check_position_size(engine.bot_id, equity, amount):
            amount = self.risk.max_trade_amount(equity)

        result = broker.buy(ticker, amount)
        if not result:
            return None

        # 更新 DB
        total_cost = result["price"] * result["shares"] + result["cost"]
        db.update_cash(engine.bot_id, -total_cost)
        stop_loss = self.risk.calculate_stop_loss_price(result["price"])
        db.open_position(
            engine.bot_id, ticker, market,
            result["shares"], result["price"], stop_loss,
        )
        db.record_trade(
            engine.bot_id, ticker, market, "buy",
            result["shares"], result["price"], result["cost"],
            reason=decision.get("reason"),
        )

        _log.info(f"[{engine.bot_id}] BUY {ticker}: {result['shares']:.4f} @ ${result['price']:.2f}")
        return {
            "bot": engine.bot_id, "action": "buy", "ticker": ticker,
            "shares": result["shares"], "price": result["price"],
            "reason": decision.get("reason", ""),
        }

    def _execute_sell(self, engine, position: dict, broker,
                      current_price: float, reason: str) -> dict | None:
        ticker = position["ticker"]
        shares = position["shares"]

        result = broker.sell(ticker, shares)
        if not result:
            return None

        # 計算 PnL
        entry_cost = position["entry_price"] * shares
        proceeds = result["proceeds"]
        pnl = proceeds - entry_cost

        # 更新 DB
        db.update_cash(engine.bot_id, proceeds)
        db.close_position(position["id"])
        db.record_trade(
            engine.bot_id, ticker, position["market"], "sell",
            shares, result["price"], result["cost"],
            pnl=pnl, reason=reason,
        )

        _log.info(f"[{engine.bot_id}] SELL {ticker}: {shares:.4f} @ ${result['price']:.2f} PnL=${pnl:.2f}")
        return {
            "bot": engine.bot_id, "action": "sell", "ticker": ticker,
            "shares": shares, "price": result["price"], "pnl": pnl,
            "reason": reason,
        }

    # ------------------------------------------------------------------
    # 每日結算
    # ------------------------------------------------------------------

    def daily_snapshot(self) -> dict:
        """記錄每日權益快照 + 更新績效指標。"""
        if not self._initialized:
            self.initialize()

        result = {}
        for engine in [self.engine_a, self.engine_b]:
            bot = db.get_bot(engine.bot_id)
            if not bot:
                continue

            equity = self._calc_equity(engine.bot_id)
            cash = db.get_cash(engine.bot_id)
            db.save_snapshot(engine.bot_id, equity, cash)

            # 更新績效指標
            self._update_metrics(engine.bot_id)

            result[engine.bot_id] = {"equity": equity, "cash": cash}
            _log.info(f"[{engine.bot_id}] snapshot: equity=${equity:.2f} cash=${cash:.2f}")

        return result

    def daily_optimize(self) -> dict:
        """每日收盤後執行 self_optimize()。"""
        changes = {}
        for engine in [self.engine_a, self.engine_b]:
            bot = db.get_bot(engine.bot_id)
            if not bot or bot["status"] != "active":
                continue
            metrics = db.get_metrics(engine.bot_id)
            if metrics:
                c = engine.self_optimize(metrics)
                if c:
                    changes[engine.bot_id] = c
                    _log.info(f"[{engine.bot_id}] optimized: {c}")
        return changes

    def daily_comparison(self) -> dict:
        """每日對比 + 淘汰檢查。"""
        if not self._initialized:
            self.initialize()

        all_metrics = db.get_all_metrics()
        comparison = {"bots": all_metrics, "elimination": None, "emergency": False}

        active = [m for m in all_metrics if m.get("status") == "active"]
        if len(active) < 2:
            return comparison

        # 安全機制：兩者皆虧損超過 20%
        for m in active:
            ret = m.get("total_return_pct", 0)
            if ret < -EMERGENCY_LOSS_PCT:
                comparison["emergency"] = True
                _log.warning(f"EMERGENCY: {m['bot_id']} loss {ret:.1f}% > {EMERGENCY_LOSS_PCT}%")

        if comparison["emergency"]:
            for m in active:
                if m.get("total_return_pct", 0) < -EMERGENCY_LOSS_PCT:
                    db.set_bot_status(m["bot_id"], "eliminated")
                    db.log_risk_event(m["bot_id"], "emergency_stop",
                                      f"loss={m.get('total_return_pct', 0):.1f}%")
            comparison["elimination"] = "emergency_both_losing"
            return comparison

        # 主要條件：勝率 > 85% 且獲利因子 > 2.0
        for m in active:
            days = db.count_trading_days(m["bot_id"])
            if days < MIN_TRADING_DAYS:
                continue
            if (m.get("win_rate", 0) > WIN_RATE_CHAMPION
                    and m.get("profit_factor", 0) > PROFIT_FACTOR_CHAMPION):
                winner = m["bot_id"]
                loser = [x["bot_id"] for x in active if x["bot_id"] != winner][0]
                db.set_bot_status(winner, "champion")
                db.set_bot_status(loser, "eliminated")
                comparison["elimination"] = {
                    "winner": winner, "loser": loser,
                    "reason": f"win_rate={m['win_rate']:.1f}% pf={m['profit_factor']:.1f}",
                }
                _log.info(f"CHAMPION: {winner} eliminates {loser}")
                return comparison

        # 次要條件：90天後 Sharpe 調整報酬率
        days_a = db.count_trading_days("bot_a")
        days_b = db.count_trading_days("bot_b")
        if days_a >= EVALUATION_DAYS and days_b >= EVALUATION_DAYS:
            ma = next((m for m in active if m["bot_id"] == "bot_a"), None)
            mb = next((m for m in active if m["bot_id"] == "bot_b"), None)
            if ma and mb:
                score_a = ma.get("total_return_pct", 0) * max(0, ma.get("sharpe_ratio", 0))
                score_b = mb.get("total_return_pct", 0) * max(0, mb.get("sharpe_ratio", 0))
                if score_a != score_b:
                    winner = "bot_a" if score_a > score_b else "bot_b"
                    loser = "bot_b" if winner == "bot_a" else "bot_a"
                    db.set_bot_status(winner, "champion")
                    db.set_bot_status(loser, "eliminated")
                    comparison["elimination"] = {
                        "winner": winner, "loser": loser,
                        "reason": f"90d sharpe_adj: A={score_a:.2f} B={score_b:.2f}",
                    }

        return comparison

    # ------------------------------------------------------------------
    # 狀態查詢
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """取得當前對戰狀態。"""
        if not self._initialized:
            self.initialize()

        bots = db.get_all_bots()
        result = {"bots": []}
        for bot in bots:
            metrics = db.get_metrics(bot["bot_id"]) or {}
            positions = db.get_open_positions(bot["bot_id"])
            equity = self._calc_equity(bot["bot_id"])
            cash = db.get_cash(bot["bot_id"])
            days = db.count_trading_days(bot["bot_id"])
            result["bots"].append({
                "bot_id": bot["bot_id"],
                "name": bot["name"],
                "status": bot["status"],
                "equity": equity,
                "cash": cash,
                "positions": len(positions),
                "trading_days": days,
                "metrics": metrics,
            })
        return result

    def get_compare(self) -> dict:
        """詳細指標對比。"""
        if not self._initialized:
            self.initialize()

        metrics_a = db.get_metrics("bot_a") or {}
        metrics_b = db.get_metrics("bot_b") or {}
        return {
            "bot_a": {"name": self.engine_a.name, **metrics_a},
            "bot_b": {"name": self.engine_b.name, **metrics_b},
        }

    def get_params(self) -> dict:
        """取得兩個 Bot 的策略參數。"""
        return {
            "bot_a": {"name": self.engine_a.name, "params": self.engine_a.get_params()},
            "bot_b": {"name": self.engine_b.name, "params": self.engine_b.get_params()},
        }

    # ------------------------------------------------------------------
    # 圖表
    # ------------------------------------------------------------------

    def draw_equity_chart(self) -> io.BytesIO | None:
        """繪製權益曲線對比圖。"""
        snaps_a = db.get_snapshots("bot_a", limit=180)
        snaps_b = db.get_snapshots("bot_b", limit=180)

        if not snaps_a and not snaps_b:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        if snaps_a:
            dates_a = [datetime.strptime(s["date"], "%Y-%m-%d") for s in snaps_a]
            eq_a = [s["equity"] for s in snaps_a]
            ax.plot(dates_a, eq_a, label=f"Bot A ({self.engine_a.name})",
                    color="#2196F3", linewidth=2)

        if snaps_b:
            dates_b = [datetime.strptime(s["date"], "%Y-%m-%d") for s in snaps_b]
            eq_b = [s["equity"] for s in snaps_b]
            ax.plot(dates_b, eq_b, label=f"Bot B ({self.engine_b.name})",
                    color="#FF5722", linewidth=2)

        ax.axhline(y=INITIAL_CAPITAL, color="gray", linestyle="--", alpha=0.5,
                    label=f"Initial (${INITIAL_CAPITAL})")
        ax.set_title("Arena Equity Curve", fontsize=14)
        ax.set_ylabel("Equity ($)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        return buf

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------

    def _calc_equity(self, bot_id: str) -> float:
        """計算總權益 = 現金 + 持倉市值。"""
        cash = db.get_cash(bot_id)
        positions = db.get_open_positions(bot_id)
        market_value = 0.0
        for pos in positions:
            broker = self._get_broker(pos["market"])
            price = broker.get_price(pos["ticker"])
            if price:
                market_value += price * pos["shares"]
            else:
                market_value += pos["entry_price"] * pos["shares"]
        return cash + market_value

    def _update_metrics(self, bot_id: str) -> None:
        """從交易紀錄計算並更新績效指標。"""
        trades = db.get_closed_trades(bot_id)
        if not trades:
            db.update_metrics(bot_id, total_trades=0)
            return

        total = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        winning = len(wins)
        losing = len(losses)
        win_rate = winning / total * 100 if total > 0 else 0

        total_gain = sum(t["pnl"] for t in wins) if wins else 0
        total_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        profit_factor = total_gain / total_loss if total_loss > 0 else (
            999.0 if total_gain > 0 else 0
        )

        # 報酬率
        equity = self._calc_equity(bot_id)
        total_return_pct = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        # Sharpe ratio（簡化：用每日報酬率算）
        snapshots = db.get_snapshots(bot_id)
        sharpe = 0.0
        if len(snapshots) >= 5:
            equities = [s["equity"] for s in snapshots]
            daily_returns = []
            for i in range(1, len(equities)):
                if equities[i - 1] > 0:
                    daily_returns.append((equities[i] - equities[i - 1]) / equities[i - 1])
            if daily_returns:
                import numpy as np
                arr = np.array(daily_returns)
                mean_r = arr.mean()
                std_r = arr.std()
                if std_r > 0:
                    sharpe = (mean_r / std_r) * (252 ** 0.5)  # 年化

        # 最大回撤
        max_drawdown = 0.0
        if len(snapshots) >= 2:
            peak = snapshots[0]["equity"]
            for s in snapshots:
                eq = s["equity"]
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak * 100 if peak > 0 else 0
                if dd > max_drawdown:
                    max_drawdown = dd

        db.update_metrics(
            bot_id,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total,
            winning_trades=winning,
            losing_trades=losing,
        )
