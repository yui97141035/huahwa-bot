"""Bot B — Mean Revert（布林帶 + RSI 均值回歸）。

刻意用不同邏輯，確保有意義的競爭。
"""

import logging

import numpy as np
import pandas as pd

from .engine_base import EngineBase
from . import arena_db as db

_log = logging.getLogger("arena.engine_b")


class EngineB(EngineBase):
    """均值回歸策略引擎。

    買入: RSI < rsi_buy 且 價格 < 布林下軌 且 量能確認
    賣出: RSI > rsi_sell 或 價格 > 布林上軌 或 持倉 > max_hold_days 天
    """

    def __init__(self) -> None:
        self._rsi_buy = 35.0       # 放寬（原 25）模擬期間觀察
        self._rsi_sell = 75.0
        self._bb_period = 20
        self._bb_std = 2.0
        self._vol_ratio_min = 0.8  # 放寬（原 1.2）模擬期間觀察
        self._max_hold_days = 10

    @property
    def bot_id(self) -> str:
        return "bot_b"

    @property
    def name(self) -> str:
        return "Mean Revert"

    def decide(self, ticker: str, market: str) -> dict | None:
        try:
            from prediction import _yf_download
        except ImportError:
            _log.error("Cannot import prediction module")
            return None

        df = _yf_download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty or len(df) < self._bb_period + 5:
            return None

        close = df["Close"].values.flatten().astype(float)
        volume = df["Volume"].values.flatten().astype(float)
        current = close[-1]

        # RSI(14)
        s_close = pd.Series(close)
        delta = s_close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if pd.isna(gain) or pd.isna(loss) or loss == 0:
            rsi = 50.0
        else:
            rsi = float(100 - (100 / (1 + gain / loss)))

        # 布林通道
        sma = s_close.rolling(self._bb_period).mean().iloc[-1]
        std = s_close.rolling(self._bb_period).std().iloc[-1]
        if pd.isna(sma) or pd.isna(std):
            return None
        sma = float(sma)
        std = float(std)
        bb_upper = sma + self._bb_std * std
        bb_lower = sma - self._bb_std * std

        # 量比
        s_vol = pd.Series(volume)
        vol_avg = s_vol.rolling(20).mean().iloc[-1]
        vol_avg = float(vol_avg) if not pd.isna(vol_avg) else float(s_vol.mean())
        vol_ratio = volume[-1] / vol_avg if vol_avg > 0 else 1.0

        # 檢查持倉
        position = db.get_position_by_ticker(self.bot_id, ticker)

        if position:
            # 持倉中 → 檢查賣出條件
            # 持倉天數
            from datetime import datetime, timezone, timedelta
            _TW = timezone(timedelta(hours=8))
            try:
                opened = datetime.strptime(position["opened_at"], "%Y-%m-%d %H:%M:%S")
                hold_days = (datetime.now(_TW).replace(tzinfo=None) - opened).days
            except Exception:
                hold_days = 0

            sell_reasons = []
            if rsi > self._rsi_sell:
                sell_reasons.append(f"RSI={rsi:.1f}>{self._rsi_sell}")
            if current > bb_upper:
                sell_reasons.append(f"price={current:.2f}>BB_upper={bb_upper:.2f}")
            if hold_days > self._max_hold_days:
                sell_reasons.append(f"hold_days={hold_days}>{self._max_hold_days}")

            if sell_reasons:
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "reason": " | ".join(sell_reasons),
                    "confidence": min(1.0, rsi / 100),
                    "amount_pct": 100,
                }

            return {"action": "hold", "ticker": ticker,
                    "reason": f"RSI={rsi:.1f} BB[{bb_lower:.2f},{bb_upper:.2f}] hold={hold_days}d",
                    "confidence": 0.5, "amount_pct": 0}

        else:
            # 無持倉 → 檢查買入條件
            if (rsi < self._rsi_buy
                    and current < bb_lower
                    and vol_ratio >= self._vol_ratio_min):
                confidence = max(0.3, 1 - rsi / 100)  # RSI 越低信心越高
                return {
                    "action": "buy",
                    "ticker": ticker,
                    "reason": f"RSI={rsi:.1f}<{self._rsi_buy} price={current:.2f}<BB_lower={bb_lower:.2f} vol_ratio={vol_ratio:.1f}",
                    "confidence": confidence,
                    "amount_pct": min(100, confidence * 100),
                }
            return None

    def get_params(self) -> dict:
        return {
            "rsi_buy": self._rsi_buy,
            "rsi_sell": self._rsi_sell,
            "bb_period": self._bb_period,
            "bb_std": self._bb_std,
            "vol_ratio_min": self._vol_ratio_min,
            "max_hold_days": self._max_hold_days,
        }

    def self_optimize(self, metrics: dict) -> dict:
        """根據績效微調參數。"""
        changes = {}
        win_rate = metrics.get("win_rate", 0)
        profit_factor = metrics.get("profit_factor", 0)

        # 勝率太低 → RSI 買入門檻更嚴格（更低）
        if win_rate < 35 and self._rsi_buy > 15:
            old = self._rsi_buy
            self._rsi_buy = self._clamp_adjust(old, -4.0, 15, 35)
            changes["rsi_buy"] = (old, self._rsi_buy)

        # 勝率高但獲利因子低 → 可能太早賣，提高賣出門檻
        if win_rate > 55 and profit_factor < 1.5 and self._rsi_sell < 85:
            old = self._rsi_sell
            self._rsi_sell = self._clamp_adjust(old, 3.0, 65, 85)
            changes["rsi_sell"] = (old, self._rsi_sell)

        # 持倉太久虧損 → 縮短持倉天數
        if profit_factor < 1.0 and self._max_hold_days > 5:
            old = float(self._max_hold_days)
            new = max(5, self._max_hold_days - 1)
            if new != self._max_hold_days:
                self._max_hold_days = new
                changes["max_hold_days"] = (old, float(new))

        # 記錄到 DB
        for key, (old, new) in changes.items():
            db.log_param_change(self.bot_id, key, old, new)

        return changes
