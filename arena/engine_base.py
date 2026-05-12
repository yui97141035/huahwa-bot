"""Arena 策略引擎抽象基底類。"""

from abc import ABC, abstractmethod


class EngineBase(ABC):
    """所有策略引擎的 ABC。"""

    @property
    @abstractmethod
    def bot_id(self) -> str:
        """唯一識別碼。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """策略名稱。"""

    @abstractmethod
    def decide(self, ticker: str, market: str) -> dict | None:
        """針對單一標的產生交易決策。

        回傳格式：
            {"action": "buy" | "sell" | "hold",
             "ticker": str,
             "reason": str,
             "confidence": float (0-1),
             "amount_pct": float (佔可用資金%)}
        或 None（無決策）。
        """

    @abstractmethod
    def get_params(self) -> dict:
        """回傳當前策略參數。"""

    @abstractmethod
    def self_optimize(self, metrics: dict) -> dict:
        """每日自我優化，回傳 {param_key: (old, new)} 調整紀錄。
        參數調整幅度限制 ±5%。"""

    def _clamp_adjust(self, current: float, delta_pct: float,
                      min_val: float, max_val: float) -> float:
        """限制參數調整幅度 ±5%，並 clamp 到範圍。"""
        delta_pct = max(-5.0, min(5.0, delta_pct))
        new = current * (1 + delta_pct / 100)
        return max(min_val, min(max_val, new))
