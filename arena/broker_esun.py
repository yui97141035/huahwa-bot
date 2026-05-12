"""Arena Broker — 玉山證券 API 真實交易（esun_trade + esun_marketdata SDK）。

支援模擬 (simulation) 及正式 (live) 環境，透過 config.ini 切換。
SDK 只支援 macOS / Windows / Linux x86_64（不支援 Linux ARM64）。
"""

import json
import logging
import os
import threading
from configparser import ConfigParser
from pathlib import Path

from .broker_base import BrokerBase

_log = logging.getLogger("arena.broker.esun")

# 台股手續費
_TW_BUY_FEE_PCT = 0.1425    # 買入手續費 0.1425%
_TW_SELL_FEE_PCT = 0.1425    # 賣出手續費 0.1425%
_TW_TAX_PCT = 0.3000         # 證交稅 0.3%（ETF 0.1%，此處簡化取 0.3%）
_TW_SELL_TOTAL_PCT = _TW_SELL_FEE_PCT + _TW_TAX_PCT  # 0.4425%


class BrokerEsun(BrokerBase):
    """玉山證券 API Broker — 整合 esun_trade + esun_marketdata。

    初始化時會讀取 config.ini 並登入 SDK。
    若 SDK 不可用（例如在不支援的平台上），會自動 fallback 到模擬模式。
    """

    def __init__(self) -> None:
        self._sdk = None             # esun_trade SDK instance
        self._marketdata = None      # esun_marketdata instance
        self._rest_stock = None      # marketdata REST stock client
        self._ws_stock = None        # marketdata WebSocket stock client
        self._config = None
        self._connected = False
        self._ws_connected = False
        self._mode = os.getenv("ESUN_MODE", "simulation")  # simulation / live
        self._lock = threading.Lock()

        # 回報暫存
        self._order_reports: list[dict] = []
        self._dealt_reports: list[dict] = []

        self._init_sdk()

    def _init_sdk(self) -> None:
        """讀取 config.ini + 憑證，初始化 SDK。"""
        config_path = os.getenv("ESUN_CONFIG_PATH", "")
        if not config_path or not Path(config_path).exists():
            _log.warning(f"ESUN_CONFIG_PATH not found: {config_path!r} — broker disabled")
            return

        try:
            from esun_trade.sdk import SDK
            from esun_marketdata import EsunMarketdata

            self._config = ConfigParser()
            self._config.read(config_path)

            # 設定憑證路徑（覆蓋 config.ini 中的 Cert.Path）
            cert_path = os.getenv("ESUN_CERT_PATH", "")
            if cert_path and Path(cert_path).exists():
                if not self._config.has_section("Cert"):
                    self._config.add_section("Cert")
                self._config.set("Cert", "Path", cert_path)

            # 交易 SDK
            self._sdk = SDK(self._config)
            _log.info(f"Esun Trade SDK initialized (mode={self._mode})")

            # 行情 SDK
            self._marketdata = EsunMarketdata(self._config)
            _log.info("Esun MarketData SDK initialized")

        except ImportError:
            _log.error("esun_trade / esun_marketdata SDK not installed")
        except Exception as e:
            _log.error(f"Failed to init Esun SDK: {e}")

    def login(self) -> bool:
        """登入交易 + 行情 SDK。

        SDK 使用 keyring 儲存密碼。首次使用時需先透過環境變數
        預存密碼到 keyring（避免互動式 getpass）。
        """
        if not self._sdk:
            _log.warning("SDK not initialized, cannot login")
            return False

        try:
            # 預存密碼到 keyring（SDK 的 login 會從 keyring 讀取）
            account = self._config.get("User", "Account", fallback="")
            cert_password = os.getenv("ESUN_CERT_PASSWORD", "")
            login_password = os.getenv("ESUN_LOGIN_PASSWORD", "")

            if account and cert_password and login_password:
                try:
                    from esun_trade.util import set_password, get_password
                    if not get_password("esun_trade_sdk:account", account):
                        set_password("esun_trade_sdk:account", account, login_password)
                    if not get_password("esun_trade_sdk:cert", account):
                        set_password("esun_trade_sdk:cert", account, cert_password)
                except Exception as e:
                    _log.warning(f"Failed to pre-store keyring credentials: {e}")

            # SDK login（從 keyring 讀取密碼）
            self._sdk.login()

            self._connected = True
            _log.info("Esun Trade SDK login success")

            # 註冊 WebSocket 回報
            self._setup_ws_callbacks()

            # 行情 SDK 登入
            if self._marketdata:
                self._marketdata.login()
                self._rest_stock = self._marketdata.rest_client.stock
                _log.info("Esun MarketData SDK login success")

            return True

        except Exception as e:
            _log.error(f"Esun SDK login failed: {e}")
            return False

    def _setup_ws_callbacks(self) -> None:
        """註冊交易 WebSocket 回報（委託回報 + 成交回報）。

        connect_websocket() 是 blocking call (run_forever)，
        必須在背景 thread 執行，否則會卡住 async event loop。
        """
        if not self._sdk:
            return

        @self._sdk.on("error")
        def _on_error(err):
            _log.warning(f"Esun WS error: {err}")

        @self._sdk.on("order")
        def _on_order(data):
            _log.info(f"Esun order report: {data}")
            with self._lock:
                self._order_reports.append(data)

        @self._sdk.on("dealt")
        def _on_dealt(data):
            _log.info(f"Esun dealt report: {data}")
            with self._lock:
                self._dealt_reports.append(data)

        def _ws_thread():
            try:
                self._sdk.connect_websocket()
                self._ws_connected = True
                _log.info("Esun Trade WebSocket connected")
            except Exception as e:
                _log.warning(f"Esun Trade WS connect failed (non-critical): {e}")

        ws_t = threading.Thread(target=_ws_thread, daemon=True, name="esun-ws")
        ws_t.start()
        _log.info("Esun Trade WebSocket thread started")

    # ------------------------------------------------------------------
    # BrokerBase 介面實作
    # ------------------------------------------------------------------

    def get_price(self, ticker: str) -> float | None:
        """透過 esun_marketdata REST API 取得即時報價。

        ticker 格式: "2330.TW" → 轉換為 SDK 用的 "2330"
        """
        symbol = self._to_symbol(ticker)

        # 優先用玉山即時行情
        if self._rest_stock:
            try:
                quote = self._rest_stock.intraday.quote(symbol=symbol)
                # quote 包含 closePrice, lastPrice 等
                price = quote.get("lastPrice") or quote.get("closePrice")
                if price and float(price) > 0:
                    return float(price)
            except Exception as e:
                _log.debug(f"Esun quote failed for {symbol}: {e}")

        # Fallback: yfinance
        return self._yf_fallback(ticker)

    def buy(self, ticker: str, amount_twd: float) -> dict | None:
        """買入台股。amount_twd 為投入金額（TWD）。

        自動判斷整股 vs 零股：
        - 金額夠買 1 張（1000 股）→ 整股委託
        - 金額不夠 1 張 → 盤中零股委託
        """
        price = self.get_price(ticker)
        if not price or price <= 0:
            _log.warning(f"Cannot buy {ticker}: no price")
            return None

        # 手續費計算
        fee = amount_twd * _TW_BUY_FEE_PCT / 100
        fee = max(fee, 20)  # 最低手續費 20 元
        net_amount = amount_twd - fee

        # 計算可買股數
        total_shares = int(net_amount / price)
        if total_shares <= 0:
            _log.warning(f"Cannot buy {ticker}: insufficient amount ({amount_twd:.0f} TWD)")
            return None

        # 真實下單
        if self._connected and self._sdk:
            success = self._place_buy_order(ticker, price, total_shares)
            if not success:
                return None

        actual_cost = price * total_shares + fee
        _log.info(
            f"ESUN BUY {ticker}: {total_shares} shares @ {price:.2f} "
            f"(fee={fee:.0f}, total={actual_cost:.0f} TWD)"
        )
        return {"shares": total_shares, "price": price, "cost": fee}

    def sell(self, ticker: str, shares: float) -> dict | None:
        """賣出台股。"""
        price = self.get_price(ticker)
        if not price or price <= 0:
            _log.warning(f"Cannot sell {ticker}: no price")
            return None

        int_shares = int(shares)
        if int_shares <= 0:
            return None

        gross = price * int_shares
        broker_fee = max(gross * _TW_SELL_FEE_PCT / 100, 20)
        tax = gross * _TW_TAX_PCT / 100
        total_fee = broker_fee + tax
        proceeds = gross - total_fee

        # 真實下單
        if self._connected and self._sdk:
            success = self._place_sell_order(ticker, price, int_shares)
            if not success:
                return None

        _log.info(
            f"ESUN SELL {ticker}: {int_shares} shares @ {price:.2f} "
            f"(fee={total_fee:.0f}, proceeds={proceeds:.0f} TWD)"
        )
        return {
            "shares": int_shares, "price": price,
            "cost": total_fee, "proceeds": proceeds,
        }

    def get_market_type(self) -> str:
        return "TW"

    # ------------------------------------------------------------------
    # 下單核心
    # ------------------------------------------------------------------

    def _place_buy_order(self, ticker: str, price: float, shares: int) -> bool:
        """透過 SDK 下買單。"""
        try:
            from esun_trade.order import OrderObject
            from esun_trade.constant import (
                APCode, Trade, PriceFlag, BSFlag, Action,
            )

            symbol = self._to_symbol(ticker)
            lots = shares // 1000       # 整張
            odd_shares = shares % 1000  # 零股

            # 整股委託（以張為單位）
            if lots > 0:
                order = OrderObject(
                    buy_sell=Action.Buy,
                    price_flag=PriceFlag.Limit,
                    price=price,
                    stock_no=symbol,
                    quantity=lots,
                    ap_code=APCode.Common,
                    trade=Trade.Cash,
                    bs_flag=BSFlag.ROD,
                )
                result = self._sdk.place_order(order)
                _log.info(f"Esun buy order (lots={lots}): {result}")

            # 零股委託（盤中零股，以股為單位）
            if odd_shares > 0:
                order = OrderObject(
                    buy_sell=Action.Buy,
                    price_flag=PriceFlag.Limit,
                    price=price,
                    stock_no=symbol,
                    quantity=odd_shares,
                    ap_code=APCode.IntradayOdd,
                    trade=Trade.Cash,
                    bs_flag=BSFlag.ROD,
                )
                result = self._sdk.place_order(order)
                _log.info(f"Esun buy order (odd={odd_shares}): {result}")

            return True

        except Exception as e:
            _log.error(f"Esun buy order failed for {ticker}: {e}")
            return False

    def _place_sell_order(self, ticker: str, price: float, shares: int) -> bool:
        """透過 SDK 下賣單。"""
        try:
            from esun_trade.order import OrderObject
            from esun_trade.constant import (
                APCode, Trade, PriceFlag, BSFlag, Action,
            )

            symbol = self._to_symbol(ticker)
            lots = shares // 1000
            odd_shares = shares % 1000

            if lots > 0:
                order = OrderObject(
                    buy_sell=Action.Sell,
                    price_flag=PriceFlag.Limit,
                    price=price,
                    stock_no=symbol,
                    quantity=lots,
                    ap_code=APCode.Common,
                    trade=Trade.Cash,
                    bs_flag=BSFlag.ROD,
                )
                result = self._sdk.place_order(order)
                _log.info(f"Esun sell order (lots={lots}): {result}")

            if odd_shares > 0:
                order = OrderObject(
                    buy_sell=Action.Sell,
                    price_flag=PriceFlag.Limit,
                    price=price,
                    stock_no=symbol,
                    quantity=odd_shares,
                    ap_code=APCode.IntradayOdd,
                    trade=Trade.Cash,
                    bs_flag=BSFlag.ROD,
                )
                result = self._sdk.place_order(order)
                _log.info(f"Esun sell order (odd={odd_shares}): {result}")

            return True

        except Exception as e:
            _log.error(f"Esun sell order failed for {ticker}: {e}")
            return False

    # ------------------------------------------------------------------
    # 擴充方法（不在 BrokerBase 中但 Arena 可用）
    # ------------------------------------------------------------------

    def get_portfolio(self) -> list[dict]:
        """查詢玉山帳戶庫存。"""
        if not self._connected or not self._sdk:
            return []
        try:
            inventories = self._sdk.get_inventories()
            return inventories if isinstance(inventories, list) else []
        except Exception as e:
            _log.error(f"get_inventories failed: {e}")
            return []

    def get_balance(self) -> dict | None:
        """查詢銀行餘額（每 180 秒限查一次）。"""
        if not self._connected or not self._sdk:
            return None
        try:
            return self._sdk.get_balance()
        except Exception as e:
            _log.error(f"get_balance failed: {e}")
            return None

    def get_order_results(self) -> list:
        """查詢當日委託結果。"""
        if not self._connected or not self._sdk:
            return []
        try:
            return self._sdk.get_order_results()
        except Exception as e:
            _log.error(f"get_order_results failed: {e}")
            return []

    def cancel_order(self, order_result, cel_qty: int | None = None) -> bool:
        """取消委託。"""
        if not self._connected or not self._sdk:
            return False
        try:
            if cel_qty:
                self._sdk.cancel_order(order_result, cel_qty=cel_qty)
            else:
                self._sdk.cancel_order(order_result)
            _log.info(f"Esun cancel order: {order_result}")
            return True
        except Exception as e:
            _log.error(f"cancel_order failed: {e}")
            return False

    def get_transactions(self, range_str: str = "0d") -> list:
        """查詢成交明細。range_str: '0d'=當日, '3d', '1m', '3m'"""
        if not self._connected or not self._sdk:
            return []
        try:
            return self._sdk.get_transactions(range_str)
        except Exception as e:
            _log.error(f"get_transactions failed: {e}")
            return []

    def get_market_status(self) -> dict | None:
        """查詢市場狀態。"""
        if not self._connected or not self._sdk:
            return None
        try:
            return self._sdk.get_market_status()
        except Exception as e:
            _log.error(f"get_market_status failed: {e}")
            return None

    def get_quote(self, ticker: str) -> dict | None:
        """取得完整即時報價資料（含五檔、成交量等）。"""
        symbol = self._to_symbol(ticker)
        if not self._rest_stock:
            return None
        try:
            return self._rest_stock.intraday.quote(symbol=symbol)
        except Exception as e:
            _log.error(f"get_quote failed for {symbol}: {e}")
            return None

    def get_ticker_info(self, ticker: str) -> dict | None:
        """取得股票資訊（含是否可當沖等）。"""
        symbol = self._to_symbol(ticker)
        if not self._rest_stock:
            return None
        try:
            return self._rest_stock.intraday.ticker(symbol=symbol)
        except Exception as e:
            _log.error(f"get_ticker_info failed for {symbol}: {e}")
            return None

    def pop_reports(self) -> tuple[list[dict], list[dict]]:
        """取出並清空所有 WebSocket 回報。回傳 (order_reports, dealt_reports)。"""
        with self._lock:
            orders = list(self._order_reports)
            dealts = list(self._dealt_reports)
            self._order_reports.clear()
            self._dealt_reports.clear()
        return orders, dealts

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------

    @staticmethod
    def _to_symbol(ticker: str) -> str:
        """'2330.TW' → '2330', '00878.TW' → '00878'"""
        return ticker.replace(".TW", "").replace(".TWO", "")

    @staticmethod
    def _yf_fallback(ticker: str) -> float | None:
        """yfinance 報價 fallback。"""
        try:
            from prediction import _yf_download
            df = _yf_download(ticker, period="5d", interval="1d", progress=False)
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as e:
            _log.debug(f"yfinance fallback failed for {ticker}: {e}")
        return None
