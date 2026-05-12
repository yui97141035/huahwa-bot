"""Arena Broker 抽象介面。"""

from abc import ABC, abstractmethod


class BrokerBase(ABC):
    """所有 Broker 實作的抽象介面。"""

    @abstractmethod
    def get_price(self, ticker: str) -> float | None:
        """取得即時/延遲報價。回傳 None 表示取不到。"""

    @abstractmethod
    def buy(self, ticker: str, amount_usd: float) -> dict | None:
        """市價買入。amount_usd 為投入金額。
        回傳 {"shares": float, "price": float, "cost": float} 或 None。"""

    @abstractmethod
    def sell(self, ticker: str, shares: float) -> dict | None:
        """市價賣出。
        回傳 {"shares": float, "price": float, "cost": float, "proceeds": float} 或 None。"""

    @abstractmethod
    def get_market_type(self) -> str:
        """回傳 'US' 或 'TW'。"""
