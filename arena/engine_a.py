"""Bot A — OpenClaw Fusion（趨勢跟隨 + 多信號融合）。

複用現有基礎設施：compute_analysis + predict_stock + 情緒 + TA Overlay。
"""

import logging

from .engine_base import EngineBase
from . import arena_db as db

_log = logging.getLogger("arena.engine_a")


class EngineA(EngineBase):
    """多信號融合策略引擎。

    Composite Score = Entry Score × w1
                    + TA Overlay  × w2
                    + LSTM 方向   × w3
                    + 市場情緒    × w4
                    + 反轉信號    × w5

    買入: composite > buy_threshold 且 SignalConfidence ≥ B 級
    賣出: composite < sell_threshold 或觸發止損
    """

    def __init__(self) -> None:
        # 可優化權重
        self._w1 = 0.35  # Entry Score (100pt)
        self._w2 = 0.25  # TA Overlay (40pt)
        self._w3 = 0.20  # LSTM 方向
        self._w4 = 0.10  # 市場情緒
        self._w5 = 0.10  # 反轉信號

        # 閾值（模擬期間放寬，觀察策略行為）
        self._buy_threshold = 0.45
        self._sell_threshold = 0.30

    @property
    def bot_id(self) -> str:
        return "bot_a"

    @property
    def name(self) -> str:
        return "OpenClaw Fusion"

    def decide(self, ticker: str, market: str) -> dict | None:
        try:
            from prediction import _yf_download, compute_analysis, predict_stock
        except ImportError:
            _log.error("Cannot import prediction module")
            return None

        # 1. 下載資料 + 基礎分析
        resolved = ticker
        df = _yf_download(resolved, period="1y", interval="1d", progress=False)
        if df.empty or len(df) < 30:
            return None

        analysis = compute_analysis(df, full=True)
        total_score = analysis["total_score"]  # 0-115
        ta_confidence = analysis.get("ta_confidence", 0)  # 0-40
        ta_overlay = analysis.get("ta_overlay", {})

        # 2. Entry Score 正規化 (0-1)
        entry_norm = min(total_score / 100.0, 1.0)

        # 3. TA Overlay 正規化 (0-1)
        ta_norm = min(ta_confidence / 40.0, 1.0)

        # 4. LSTM 方向 (0-1)：預測價格 vs 現價
        lstm_score = 0.5  # 預設中性
        try:
            pred = predict_stock(resolved)
            preds = pred.get("predictions", [])
            last_price = pred.get("last_price", analysis["current"])
            if preds and last_price > 0:
                avg_pred = sum(preds) / len(preds)
                pct_change = (avg_pred - last_price) / last_price
                lstm_score = max(0, min(1, 0.5 + pct_change * 5))  # ±10% → 0~1
        except Exception as e:
            _log.debug(f"LSTM failed for {ticker}: {e}")

        # 5. 市場情緒 (0-1)
        sentiment_score = 0.5
        try:
            from market_data import fetch_fear_greed, fetch_vix
            fg = fetch_fear_greed()
            if fg and "score" in fg:
                sentiment_score = fg["score"] / 100.0
            vix = fetch_vix()
            if vix and "value" in vix:
                vix_val = vix["value"]
                # VIX 高 → 恐慌 → 降低分數
                if vix_val > 30:
                    sentiment_score *= 0.7
                elif vix_val > 25:
                    sentiment_score *= 0.85
        except Exception:
            pass

        # 6. 反轉信號 (0-1)
        reversal = analysis["scores"].get("reversal", {})
        reversal_score = reversal.get("score", 0) / 15.0 if isinstance(reversal, dict) else 0

        # 合成分數
        composite = (
            entry_norm * self._w1
            + ta_norm * self._w2
            + lstm_score * self._w3
            + sentiment_score * self._w4
            + reversal_score * self._w5
        )

        # 檢查是否有現有持倉
        position = db.get_position_by_ticker(self.bot_id, ticker)

        # Signal Confidence 等級
        sig_conf = ta_overlay.get("signal_confidence", {})
        grade = sig_conf.get("grade", "C") if isinstance(sig_conf, dict) else "C"

        # 決策
        if position:
            # 有持倉 → 檢查是否賣出
            if composite < self._sell_threshold:
                return {
                    "action": "sell",
                    "ticker": ticker,
                    "reason": f"composite={composite:.2f}<{self._sell_threshold} (entry={entry_norm:.2f} ta={ta_norm:.2f} lstm={lstm_score:.2f})",
                    "confidence": 1 - composite,
                    "amount_pct": 100,
                }
            return {"action": "hold", "ticker": ticker,
                    "reason": f"composite={composite:.2f} — hold",
                    "confidence": composite, "amount_pct": 0}
        else:
            # 無持倉 → 檢查是否買入
            if composite > self._buy_threshold and grade in ("S", "A", "B", "C"):
                # 依信心度分配資金
                pct = min(100, composite * 100)
                return {
                    "action": "buy",
                    "ticker": ticker,
                    "reason": f"composite={composite:.2f}>{self._buy_threshold} grade={grade} (entry={entry_norm:.2f} ta={ta_norm:.2f} lstm={lstm_score:.2f} sent={sentiment_score:.2f})",
                    "confidence": composite,
                    "amount_pct": pct,
                }
            return None

    def get_params(self) -> dict:
        return {
            "w_entry": self._w1,
            "w_ta": self._w2,
            "w_lstm": self._w3,
            "w_sentiment": self._w4,
            "w_reversal": self._w5,
            "buy_threshold": self._buy_threshold,
            "sell_threshold": self._sell_threshold,
        }

    def self_optimize(self, metrics: dict) -> dict:
        """根據績效微調權重。"""
        changes = {}
        win_rate = metrics.get("win_rate", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        # 勝率偏低 → 提高門檻，減少進場
        if win_rate < 40 and self._buy_threshold < 0.80:
            old = self._buy_threshold
            self._buy_threshold = self._clamp_adjust(old, 3.0, 0.50, 0.85)
            changes["buy_threshold"] = (old, self._buy_threshold)

        # 勝率偏高但 Sharpe 低 → 可能持倉太久，降低賣出門檻
        if win_rate > 60 and sharpe < 0.5 and self._sell_threshold > 0.15:
            old = self._sell_threshold
            self._sell_threshold = self._clamp_adjust(old, -3.0, 0.15, 0.50)
            changes["sell_threshold"] = (old, self._sell_threshold)

        # Sharpe 好 → 稍微加重 LSTM 權重
        if sharpe > 1.0 and self._w3 < 0.30:
            old = self._w3
            self._w3 = self._clamp_adjust(old, 4.0, 0.10, 0.35)
            # 重新正規化權重總和=1
            self._rebalance_weights()
            changes["w_lstm"] = (old, self._w3)

        # 記錄到 DB
        for key, (old, new) in changes.items():
            db.log_param_change(self.bot_id, key, old, new)

        return changes

    def _rebalance_weights(self) -> None:
        """確保五個權重總和為 1.0。"""
        total = self._w1 + self._w2 + self._w3 + self._w4 + self._w5
        if total > 0:
            self._w1 /= total
            self._w2 /= total
            self._w3 /= total
            self._w4 /= total
            self._w5 /= total
