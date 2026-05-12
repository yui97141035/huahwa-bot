"""Arena Broker — 台股模擬交易（yfinance 報價 + 台股手續費）。"""

import logging

from .broker_base import BrokerBase

_log = logging.getLogger("arena.broker.sim")

# 台股手續費模型（複用 backtest.py 的 TW_COST 邏輯）
_TW_BUY_FEE_PCT = 0.1425   # 買入手續費 %
_TW_SELL_FEE_PCT = 0.4425  # 賣出手續費 % (含 0.3% 證交稅)


class BrokerSim(BrokerBase):
    """台股模擬交易 Broker。用 yfinance 取得報價，套用台股手續費。"""

    @staticmethod
    def _get_yf_price(ticker: str) -> float | None:
        try:
            from prediction import _yf_download
            df = _yf_download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as e:
            _log.debug(f"yfinance price failed for {ticker}: {e}")
        return None

    def get_price(self, ticker: str) -> float | None:
        return self._get_yf_price(ticker)

    def buy(self, ticker: str, amount_usd: float) -> dict | None:
        price = self.get_price(ticker)
        if not price or price <= 0:
            _log.warning(f"Cannot buy {ticker}: no price")
            return None

        cost = amount_usd * _TW_BUY_FEE_PCT / 100
        net_amount = amount_usd - cost
        shares = round(net_amount / price, 6)

        _log.info(f"SIM BUY {ticker}: {shares} shares @ ${price} (fee=${cost:.4f})")
        return {"shares": shares, "price": price, "cost": cost}

    def sell(self, ticker: str, shares: float) -> dict | None:
        price = self.get_price(ticker)
        if not price or price <= 0:
            _log.warning(f"Cannot sell {ticker}: no price")
            return None

        gross = price * shares
        cost = gross * _TW_SELL_FEE_PCT / 100
        proceeds = gross - cost

        _log.info(f"SIM SELL {ticker}: {shares} shares @ ${price} (fee=${cost:.4f})")
        return {"shares": shares, "price": price,
                "cost": cost, "proceeds": proceeds}

    def get_market_type(self) -> str:
        return "TW"
