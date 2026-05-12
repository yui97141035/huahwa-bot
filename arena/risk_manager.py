"""Arena 風控引擎 — 止損、日損限制、倉位上限 + 台股特殊風控。"""

import logging

from . import arena_db as db

_log = logging.getLogger("arena.risk")

# 風控參數（不可由 self_optimize 自動調整）
STOP_LOSS_PCT = 5.0       # 單筆止損 5%
DAILY_LOSS_LIMIT_PCT = 3.0  # 日損上限 3%（佔初始資金）
MAX_POSITION_PCT = 20.0   # 單股最大倉位 20%
INITIAL_CAPITAL = 4000.0  # 每個 Bot 初始資金 (TWD, 模擬帳戶 8675 分兩 Bot)

# 台股風控參數
INITIAL_CAPITAL_TW = 50000.0  # 台股初始資金 (TWD)
TW_LIMIT_UP_PCT = 10.0       # 台股漲停 10%
TW_LIMIT_DOWN_PCT = 10.0     # 台股跌停 10%
TW_MIN_ORDER_TWD = 1000.0    # 台股最低下單金額


class RiskManager:
    """共用風控引擎，在每次交易前/後檢查。"""

    def check_stop_loss(self, bot_id: str, position: dict,
                        current_price: float) -> bool:
        """檢查是否觸發止損。回傳 True = 應該平倉。"""
        entry = position["entry_price"]
        loss_pct = (entry - current_price) / entry * 100
        if loss_pct >= STOP_LOSS_PCT:
            db.log_risk_event(
                bot_id, "stop_loss",
                f"{position['ticker']}: entry={entry:.2f} current={current_price:.2f} loss={loss_pct:.1f}%",
            )
            _log.warning(f"[{bot_id}] stop_loss triggered: {position['ticker']} -{loss_pct:.1f}%")
            return True
        return False

    def check_daily_loss(self, bot_id: str) -> bool:
        """檢查當日虧損是否超過限制。回傳 True = 應該停止交易。"""
        daily_pnl = db.get_daily_pnl(bot_id)
        limit = INITIAL_CAPITAL * DAILY_LOSS_LIMIT_PCT / 100
        if daily_pnl < -limit:
            db.log_risk_event(
                bot_id, "daily_limit",
                f"daily_pnl={daily_pnl:.2f} limit=-{limit:.2f}",
            )
            _log.warning(f"[{bot_id}] daily loss limit reached: ${daily_pnl:.2f}")
            return True
        return False

    def check_position_size(self, bot_id: str, equity: float,
                            proposed_amount: float) -> bool:
        """檢查單筆交易是否超過倉位上限。回傳 True = 可以執行。"""
        if equity <= 0:
            return False
        pct = proposed_amount / equity * 100
        if pct > MAX_POSITION_PCT:
            db.log_risk_event(
                bot_id, "position_limit",
                f"proposed={proposed_amount:.2f} equity={equity:.2f} pct={pct:.1f}%",
            )
            _log.info(f"[{bot_id}] position size capped: {pct:.1f}% > {MAX_POSITION_PCT}%")
            return False
        return True

    def max_trade_amount(self, equity: float) -> float:
        """回傳單筆最大可用金額。"""
        return equity * MAX_POSITION_PCT / 100

    def calculate_stop_loss_price(self, entry_price: float, side: str = "long") -> float:
        """計算止損價。"""
        if side == "long":
            return entry_price * (1 - STOP_LOSS_PCT / 100)
        return entry_price * (1 + STOP_LOSS_PCT / 100)

    # ------------------------------------------------------------------
    # 台股特殊風控
    # ------------------------------------------------------------------

    def check_tw_limit_price(self, current_price: float, order_price: float,
                             action: str = "buy") -> bool:
        """檢查委託價是否在漲跌停範圍內。回傳 True = 價格合法。"""
        limit_up = current_price * (1 + TW_LIMIT_UP_PCT / 100)
        limit_down = current_price * (1 - TW_LIMIT_DOWN_PCT / 100)
        if order_price < limit_down or order_price > limit_up:
            _log.warning(
                f"TW price out of range: order={order_price:.2f} "
                f"limit=[{limit_down:.2f}, {limit_up:.2f}]"
            )
            return False
        return True

    def check_tw_min_order(self, amount_twd: float) -> bool:
        """檢查台股最低下單金額。"""
        return amount_twd >= TW_MIN_ORDER_TWD
