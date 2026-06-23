"""
花城 Feature Engine — 統一技術指標計算 + LSTM 特徵矩陣
使用 pandas-ta 計算 30+ 指標；import 失敗時 graceful fallback 到手寫計算。
"""

import logging
import numpy as np
import pandas as pd

_log = logging.getLogger("huacheng.feature_engine")

# ---------------------------------------------------------------------------
# pandas-ta import（graceful fallback）
# ---------------------------------------------------------------------------
_HAS_PANDAS_TA = False
try:
    import pandas_ta as ta
    _HAS_PANDAS_TA = True
    _log.info("feature_engine: pandas-ta loaded")
except ImportError:
    _log.warning("feature_engine: pandas-ta not installed, using fallback calculations")


# ---------------------------------------------------------------------------
# 手寫 fallback 指標（當 pandas-ta 不可用時）
# ---------------------------------------------------------------------------
def _fallback_rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(length).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _fallback_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast).mean()
    ema_slow = close.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        "MACD_12_26_9": macd_line,
        "MACDs_12_26_9": macd_signal,
        "MACDh_12_26_9": macd_hist,
    })


def _fallback_bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    sma = close.rolling(length).mean()
    stdev = close.rolling(length).std()
    upper = sma + std * stdev
    lower = sma - std * stdev
    mid = sma
    # BB %B = (close - lower) / (upper - lower)
    bandwidth = upper - lower
    pct_b = (close - lower) / bandwidth.replace(0, np.nan)
    return pd.DataFrame({
        "BBU_20_2.0": upper,
        "BBM_20_2.0": mid,
        "BBL_20_2.0": lower,
        "BBB_20_2.0": bandwidth / mid * 100,  # bandwidth %
        "BBP_20_2.0": pct_b,
    })


def _fallback_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()


def _fallback_adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Simplified ADX calculation."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    atr = _fallback_atr(high, low, close, length)
    atr_safe = atr.replace(0, np.nan)

    plus_di = 100 * plus_dm.rolling(length).mean() / atr_safe
    minus_di = 100 * minus_dm.rolling(length).mean() / atr_safe

    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.rolling(length).mean()
    return adx


# ---------------------------------------------------------------------------
# compute_features(df) — 計算所有技術指標
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算技術指標並附加到 DataFrame。回傳帶有 30+ 指標欄位的 DataFrame 副本。
    需要 OHLCV 欄位: Open, High, Low, Close, Volume。
    """
    out = df.copy()

    close = out["Close"].squeeze()
    high = out["High"].squeeze() if "High" in out.columns else close
    low = out["Low"].squeeze() if "Low" in out.columns else close
    volume = out["Volume"].squeeze() if "Volume" in out.columns else pd.Series(0, index=close.index)

    if _HAS_PANDAS_TA:
        try:
            return _compute_features_pandas_ta(out, close, high, low, volume)
        except Exception as e:
            _log.warning(f"feature_engine: pandas-ta failed ({e}), using fallback")

    return _compute_features_fallback(out, close, high, low, volume)


def _compute_features_pandas_ta(out: pd.DataFrame, close: pd.Series,
                                 high: pd.Series, low: pd.Series,
                                 volume: pd.Series) -> pd.DataFrame:
    """使用 pandas-ta 計算完整指標集。"""
    # 均線
    out["ma5"] = ta.sma(close, length=5)
    out["ma10"] = ta.sma(close, length=10)
    out["ma20"] = ta.sma(close, length=20)
    out["ma60"] = ta.sma(close, length=60)
    out["ma200"] = ta.sma(close, length=200)
    out["ema12"] = ta.ema(close, length=12)
    out["ema26"] = ta.ema(close, length=26)

    # RSI
    out["rsi_14"] = ta.rsi(close, length=14)

    # MACD
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None:
        out = pd.concat([out, macd_df], axis=1)
        out["macd_hist"] = out.get("MACDh_12_26_9")

    # Bollinger Bands
    bb_df = ta.bbands(close, length=20, std=2.0)
    if bb_df is not None:
        out = pd.concat([out, bb_df], axis=1)
        out["bb_pct"] = out.get("BBP_20_2.0")

    # Volume
    vol_ma20 = volume.rolling(20).mean()
    out["vol_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # ATR
    atr = ta.atr(high, low, close, length=14)
    if atr is not None:
        out["atr_14"] = atr
        # ATR 正規化（除以收盤價，百分比形式）
        out["atr_14_pct"] = atr / close * 100

    # ADX
    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None:
        out = pd.concat([out, adx_df], axis=1)
        out["adx_14"] = out.get("ADX_14")

    # Stochastic
    stoch_df = ta.stoch(high, low, close, k=14, d=3)
    if stoch_df is not None:
        out = pd.concat([out, stoch_df], axis=1)

    # OBV
    obv = ta.obv(close, volume)
    if obv is not None:
        out["obv"] = obv

    # Williams %R
    willr = ta.willr(high, low, close, length=14)
    if willr is not None:
        out["willr_14"] = willr

    # CCI
    cci = ta.cci(high, low, close, length=20)
    if cci is not None:
        out["cci_20"] = cci

    # MFI (Money Flow Index)
    mfi = ta.mfi(high, low, close, volume, length=14)
    if mfi is not None:
        out["mfi_14"] = mfi

    # ROC (Rate of Change)
    roc = ta.roc(close, length=10)
    if roc is not None:
        out["roc_10"] = roc

    # VWAP (if intraday index, otherwise skip)
    try:
        vwap = ta.vwap(high, low, close, volume)
        if vwap is not None:
            out["vwap"] = vwap
    except Exception:
        pass

    # Close normalized (for LSTM — legacy, kept for backward compat)
    c_min, c_max = close.min(), close.max()
    if c_max - c_min > 0:
        out["close_norm"] = (close - c_min) / (c_max - c_min)
    else:
        out["close_norm"] = 0.5

    # Daily return + 5-day rolling return（LSTM 預測 target）
    out["close_return"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)

    return out


def _compute_features_fallback(out: pd.DataFrame, close: pd.Series,
                                high: pd.Series, low: pd.Series,
                                volume: pd.Series) -> pd.DataFrame:
    """不用 pandas-ta 的手寫指標 fallback。"""
    # 均線
    out["ma5"] = close.rolling(5).mean()
    out["ma10"] = close.rolling(10).mean()
    out["ma20"] = close.rolling(20).mean()
    out["ma60"] = close.rolling(60).mean()
    out["ma200"] = close.rolling(200).mean()
    out["ema12"] = close.ewm(span=12).mean()
    out["ema26"] = close.ewm(span=26).mean()

    # RSI
    out["rsi_14"] = _fallback_rsi(close, 14)

    # MACD
    macd_df = _fallback_macd(close)
    out = pd.concat([out, macd_df], axis=1)
    out["macd_hist"] = out["MACDh_12_26_9"]

    # Bollinger
    bb_df = _fallback_bbands(close)
    out = pd.concat([out, bb_df], axis=1)
    out["bb_pct"] = out["BBP_20_2.0"]

    # Volume ratio
    vol_ma20 = volume.rolling(20).mean()
    out["vol_ratio"] = volume / vol_ma20.replace(0, np.nan)

    # ATR
    out["atr_14"] = _fallback_atr(high, low, close, 14)
    out["atr_14_pct"] = out["atr_14"] / close * 100

    # ADX
    out["adx_14"] = _fallback_adx(high, low, close, 14)

    # Close normalized (legacy)
    c_min, c_max = close.min(), close.max()
    if c_max - c_min > 0:
        out["close_norm"] = (close - c_min) / (c_max - c_min)
    else:
        out["close_norm"] = 0.5

    # Daily return + 5-day rolling return
    out["close_return"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)

    return out


# ---------------------------------------------------------------------------
# get_lstm_feature_matrix(df) — 給多變量 LSTM 用的特徵矩陣
# ---------------------------------------------------------------------------
def get_lstm_feature_matrix(df: pd.DataFrame,
                            featured: pd.DataFrame | None = None,
                            ) -> tuple[np.ndarray, dict]:
    """
    從 OHLCV DataFrame 計算特徵矩陣。
    回傳:
      matrix: shape (n_days, n_features) 的 min-max 正規化矩陣
      scalers: dict，包含:
        - 每個特徵的 (vmin, vmax) 用於反正規化
        - "_last_close": 最後一天的收盤價（LSTM 用來從 return 推算價格）

    特徵順序由 prediction_config.LSTM_FEATURES 定義。
    第 0 欄 (close_return) 是日報酬率，LSTM 預測它後乘以前日收盤價得到預測價格。

    featured: 若已經呼叫過 compute_features()，可直接傳入避免重複計算。
    """
    from prediction_config import LSTM_FEATURES

    if featured is None:
        featured = compute_features(df)

    # 確保所有特徵欄位存在
    for col in LSTM_FEATURES:
        if col not in featured.columns:
            featured[col] = 0.0

    # 取出特徵子集
    feat_df = featured[LSTM_FEATURES].copy()

    # forward-fill 然後 back-fill 處理開頭的 NaN（MA/RSI 需要暖機期）
    feat_df = feat_df.ffill().bfill()

    # 還有殘留 NaN 就填 0
    feat_df = feat_df.fillna(0.0)

    # Min-Max 正規化（逐欄）
    scalers = {}
    matrix = np.zeros((len(feat_df), len(LSTM_FEATURES)), dtype=np.float32)
    for i, col in enumerate(LSTM_FEATURES):
        vals = feat_df[col].values.astype(np.float32)
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax - vmin > 0:
            matrix[:, i] = (vals - vmin) / (vmax - vmin)
        else:
            matrix[:, i] = 0.5
        scalers[col] = (vmin, vmax)

    # 存最後收盤價，供 predict_stock() 從 return 推算未來價格
    raw_close = df["Close"].squeeze()
    scalers["_last_close"] = float(raw_close.iloc[-1])

    return matrix, scalers
