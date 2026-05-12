"""Arena Broker — Alpaca 美股（Paper / Live）。"""

import logging
import os

from .broker_base import BrokerBase

_log = logging.getLogger("arena.broker.alpaca")


class BrokerAlpaca(BrokerBase):
    """Alpaca REST API 整合，支援 Paper 和 Live 模式。"""

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

        if not api_key or not secret_key:
            _log.warning("Alpaca API keys not configured — broker will be read-only")
            self._trading = None
            self._data = None
            return

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient

            self._trading = TradingClient(api_key, secret_key, paper=paper)
            self._data = StockHistoricalDataClient(api_key, secret_key)
            _log.info(f"Alpaca broker initialized (paper={paper})")
        except ImportError:
            _log.warning("alpaca-py not installed — using fallback pricing")
            self._trading = None
            self._data = None
        except Exception as e:
            _log.error(f"Alpaca init error: {e}")
            self._trading = None
            self._data = None

    def get_price(self, ticker: str) -> float | None:
        """取得最新報價。先嘗試 Alpaca，失敗則用 yfinance。"""
        # 嘗試 Alpaca snapshot
        if self._data:
            try:
                from alpaca.data.requests import StockLatestQuoteRequest
                req = StockLatestQuoteRequest(symbol_or_symbols=ticker)
                quotes = self._data.get_stock_latest_quote(req)
                if ticker in quotes:
                    q = quotes[ticker]
                    mid = (q.ask_price + q.bid_price) / 2
                    if mid > 0:
                        return round(mid, 4)
            except Exception as e:
                _log.debug(f"Alpaca quote failed for {ticker}: {e}")

        # fallback: yfinance
        return self._yf_price(ticker)

    @staticmethod
    def _yf_price(ticker: str) -> float | None:
        try:
            from prediction import _yf_download
            df = _yf_download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as e:
            _log.debug(f"yfinance price fallback failed for {ticker}: {e}")
        return None

    def buy(self, ticker: str, amount_usd: float) -> dict | None:
        price = self.get_price(ticker)
        if not price or price <= 0:
            _log.warning(f"Cannot buy {ticker}: no price available")
            return None

        if self._trading:
            try:
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce

                # Alpaca 支援碎股 (fractional shares)
                shares = round(amount_usd / price, 6)
                order = self._trading.submit_order(
                    MarketOrderRequest(
                        symbol=ticker,
                        qty=shares,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else price
                fill_shares = float(order.filled_qty) if order.filled_qty else shares
                cost = fill_price * fill_shares * 0.001  # 估算滑價
                _log.info(f"Alpaca BUY {ticker}: {fill_shares} shares @ ${fill_price}")
                return {"shares": fill_shares, "price": fill_price, "cost": cost}
            except Exception as e:
                _log.error(f"Alpaca buy error {ticker}: {e}")

        # Paper fallback: 模擬成交
        shares = round(amount_usd / price, 6)
        cost = amount_usd * 0.001
        _log.info(f"Simulated BUY {ticker}: {shares} shares @ ${price}")
        return {"shares": shares, "price": price, "cost": cost}

    def sell(self, ticker: str, shares: float) -> dict | None:
        price = self.get_price(ticker)
        if not price or price <= 0:
            _log.warning(f"Cannot sell {ticker}: no price available")
            return None

        if self._trading:
            try:
                from alpaca.trading.requests import MarketOrderRequest
                from alpaca.trading.enums import OrderSide, TimeInForce

                order = self._trading.submit_order(
                    MarketOrderRequest(
                        symbol=ticker,
                        qty=round(shares, 6),
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else price
                fill_shares = float(order.filled_qty) if order.filled_qty else shares
                proceeds = fill_price * fill_shares
                cost = proceeds * 0.001
                _log.info(f"Alpaca SELL {ticker}: {fill_shares} shares @ ${fill_price}")
                return {"shares": fill_shares, "price": fill_price,
                        "cost": cost, "proceeds": proceeds - cost}
            except Exception as e:
                _log.error(f"Alpaca sell error {ticker}: {e}")

        # Paper fallback
        proceeds = price * shares
        cost = proceeds * 0.001
        _log.info(f"Simulated SELL {ticker}: {shares} shares @ ${price}")
        return {"shares": shares, "price": price,
                "cost": cost, "proceeds": proceeds - cost}

    def get_market_type(self) -> str:
        return "US"
