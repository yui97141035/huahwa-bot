"""
小龍蝦 OpenClaw 1.32 — 股票趨勢預測 + 技術分析 + 進場評分模組
使用 LSTM 模型預測 + 多指標共振評分系統，支援 Apple MPS GPU 加速。
v1.30: 新增八大進階技術分析模組（K線形態/均線過濾/支撐壓力/形態辨識/回測確認/訊號分級/避險過濾/獲利排序）

============================================================================
進場評分系統（基礎 100 分 + 反轉加分 15 分）
============================================================================
1. 趨勢分數 (25分) — 均線排列 + 價格與均線的相對位置
   ‧ 多頭排列 Price > MA5 > MA20 > MA60 → 25
   ‧ Price > MA5 > MA20                  → 18
   ‧ Price > MA20 (中期趨勢向上)         → 12
   ‧ Price > MA5 (短期止穩回升)          →  8
   ‧ Price < MA20 但 > MA60              →  6
   ‧ Price < MA20 且 < MA60 (空頭)       →  0

2. 動能分數 (20分) — RSI(14) 判斷超買超賣
   ‧ RSI 30~40  超賣反彈區 (最佳進場)    → 20
   ‧ RSI <30    極度超賣 (反轉機會)      → 15
   ‧ RSI 40~50  蓄力區                   → 15
   ‧ RSI 50~60  健康上漲                 → 10
   ‧ RSI 60~70  偏強但追高風險漸增       →  5
   ‧ RSI >70 超買                        →  0

3. MACD 分數 (20分) — 趨勢轉折確認
   ‧ 金叉且柱狀體連續 3 日放大           → 20
   ‧ 金叉 (MACD > Signal)               → 14
   ‧ 死叉但柱狀體縮小 (即將金叉)         →  8
   ‧ 死叉且柱狀體放大                    →  0

4. 布林通道分數 (15分) — 判斷價格位置
   ‧ 股價在下軌附近 (< lower+10%帶寬)   → 15  (超跌反彈)
   ‧ 股價在中下區 (< middle)             → 10
   ‧ 股價在中上區 (< upper-10%帶寬)      →  5
   ‧ 股價突破上軌                        →  0  (過熱)

5. 量能分數 (20分) — 量價配合
   ‧ 量縮回調後放量上攻 (量比>1.5 且漲)  → 20
   ‧ 溫和放量上漲 (量比1.0~1.5 且漲)     → 15
   ‧ 量縮整理 (量比<0.8)                 → 10  (健康回調)
   ‧ 放量下跌 (量比>1.5 且跌)            →  0  (危險)

6. 反轉訊號 (15分 Bonus) — 底部反轉偵測
   ‧ 超跌止穩 (RSI<35 且今日收漲)        → +5
   ‧ MACD 空方動能減弱 (柱狀體回升)      → +5
   ‧ 布林下軌反彈 (下軌附近且收漲)       → +5
   ‧ 無反轉訊號                          →  0

============================================================================
進階技術分析疊加層 (ta_overlay, 最高 40 分)
============================================================================
  模組一: K 線力道 (0-5)  — 單根/雙根/三根 K 線形態
  模組二: 均線過濾 (0-5)  — MA50/100/200 交叉 + 趨勢
  模組三: 支撐壓力 (0-5)  — Pivot + 水平線 + 翻轉偵測
  模組四: 形態辨識 (0-5)  — 雙底/頭肩/楔形/三角/旗形/菱形
  模組五: 回測確認 (0-5)  — 拉回支撐守住 + 反轉 K 線
  模組六: 訊號分級 (0-5)  — S/A/B 級信心度 + 部位建議
  模組七: 避險過濾 (0-5)  — 盤整偵測 + 禁止交易區
  模組八: 獲利排序 (0-5)  — 移動停利 vs 固定停利
============================================================================
加碼條件（同時滿足以下條件）
============================================================================
  ① 總分 ≥ 60（大趨勢偏多）
  ② 股價回踩 MA20 附近（±2%）
  ③ RSI 在 40~55 之間（回調未過深）
  ④ 成交量萎縮（量比 < 0.8，代表賣壓減輕）
============================================================================
"""

import io
import os
import csv
import time as _time
import logging
import threading
from datetime import datetime, timezone, timedelta
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from ta_modules import compute_ta_overlay
from prediction_config import (
    ENABLE_MULTIVARIATE_LSTM, ENABLE_PANDAS_TA,
    LSTM_INPUT_SIZE, LSTM_HIDDEN_SIZE, LSTM_DROPOUT,
    LSTM_LOOK_BACK, LSTM_PRED_DAYS, LSTM_MAX_EPOCHS, LSTM_PATIENCE, LSTM_LR,
    LSTM_GRAD_CLIP, LSTM_WARMUP_EPOCHS,
    ENABLE_NHITS, ENABLE_SENTIMENT, ENABLE_AGENT_GATE,
)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()
_log = logging.getLogger("openclaw.prediction")

# ---------------------------------------------------------------------------
# 預測記錄
# ---------------------------------------------------------------------------
_TW = timezone(timedelta(hours=8))
_PRED_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions_log.csv")
_pred_log_lock = threading.Lock()
_PRED_LOG_FIELDS = [
    "predict_time", "ticker", "last_price",
    "pred_day1", "pred_day2", "pred_day3", "pred_day4",
    "pred_day5", "pred_day6", "pred_day7",
    "model_val_mae",
]


def log_prediction(ticker: str, last_price: float, predictions: list[float],
                   val_mae: float | None) -> None:
    """Append one prediction record to predictions_log.csv (thread-safe)."""
    row = {
        "predict_time": datetime.now(_TW).strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker.upper(),
        "last_price": f"{last_price:.4f}",
    }
    for i, p in enumerate(predictions[:7], 1):
        row[f"pred_day{i}"] = f"{p:.4f}"
    row["model_val_mae"] = f"{val_mae:.4f}" if val_mae is not None else ""

    with _pred_log_lock:
        write_header = not os.path.exists(_PRED_LOG)
        with open(_PRED_LOG, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_PRED_LOG_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    _log.info(f"log_prediction: {ticker} logged to {_PRED_LOG}")


# ---------------------------------------------------------------------------
# yfinance 下載（含 retry + thread lock）
# ---------------------------------------------------------------------------
_yf_lock = threading.Lock()

def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 回傳 MultiIndex columns 時，降為單層。"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
        # 若有重複欄位名（不應發生在單 ticker），取第一個
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
    return df

def _yf_download(ticker: str, **kwargs) -> pd.DataFrame:
    """yf.download with retry (max 3 attempts, exponential backoff).
    自動處理 401 Invalid Crumb：清除 yfinance cookie 快取後重試。
    使用 lock 避免平行化時 yfinance 內部狀態衝突。
    """
    for attempt in range(3):
        try:
            with _yf_lock:
                df = yf.download(ticker, **kwargs)
            if not df.empty:
                return _flatten_yf_columns(df)
        except Exception as e:
            msg = str(e)
            _log.warning(f"yf.download({ticker}) attempt {attempt+1} failed: {e}")
            # 401 Invalid Crumb / Unauthorized → 清除 yfinance session 快取再重試
            if "401" in msg or "Unauthorized" in msg or "Invalid Crumb" in msg:
                try:
                    import yfinance.data as _yfdata
                    if hasattr(_yfdata, '_YfData__session'):
                        _yfdata._YfData__session = None
                    # 也清除 cache_manager
                    if hasattr(yf, 'cache'):
                        yf.cache.clear()
                except Exception:
                    pass
        if attempt < 2:
            _time.sleep(1.5 ** (attempt + 1))
    return pd.DataFrame()

# ---------------------------------------------------------------------------
# LSTM 模型
# ---------------------------------------------------------------------------
class StockLSTM(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ---------------------------------------------------------------------------
# 資料前處理
# ---------------------------------------------------------------------------
LOOK_BACK = 30
PRED_DAYS = 7
_PRED_CACHE_TTL = 1800  # 30 分鐘快取

# ---------------------------------------------------------------------------
# 預測結果快取（不含 chart_buf，hit 時重新產圖）
# ---------------------------------------------------------------------------
_pred_cache: dict[str, dict] = {}   # ticker -> {"ts": float, "data": dict}

def _pred_cache_get(ticker: str) -> dict | None:
    entry = _pred_cache.get(ticker.upper())
    if entry and (_time.time() - entry["ts"]) < _PRED_CACHE_TTL:
        _log.info(f"predict_stock({ticker}): cache hit")
        return entry["data"]
    return None

def _pred_cache_set(ticker: str, data: dict) -> None:
    _pred_cache[ticker.upper()] = {"ts": _time.time(), "data": data}

def _normalize(series: np.ndarray):
    vmin, vmax = series.min(), series.max()
    if vmax - vmin == 0:
        return series, vmin, vmax
    return (series - vmin) / (vmax - vmin), vmin, vmax

def _denormalize(arr: np.ndarray, vmin: float, vmax: float):
    return arr * (vmax - vmin) + vmin

def _make_sequences(data: np.ndarray, look_back: int):
    xs, ys = [], []
    for i in range(len(data) - look_back):
        xs.append(data[i : i + look_back])
        ys.append(data[i + look_back])
    return np.array(xs), np.array(ys)


def _make_sequences_multi(matrix: np.ndarray, look_back: int):
    """Create sequences from multi-feature matrix.
    matrix: shape (n_days, n_features)
    Returns:
      X: shape (n_samples, look_back, n_features)
      y: shape (n_samples,) — target is first column (close_norm)
    """
    xs, ys = [], []
    for i in range(len(matrix) - look_back):
        xs.append(matrix[i : i + look_back])
        ys.append(matrix[i + look_back, 0])  # close_norm is column 0
    return np.array(xs), np.array(ys)

# ---------------------------------------------------------------------------
# 代碼解析
# ---------------------------------------------------------------------------
_TICKER_ALIASES = {
    "RGT": "RGTI",
}

# 快取已解析過的代碼，避免每次監控都重複查詢 yfinance（消除 .TW→404 噪音）
_resolved_cache: dict[str, str] = {}

def _resolve_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker in _resolved_cache:
        return _resolved_cache[ticker]
    raw = ticker
    # 去除 .US 後綴（使用者可能從看盤軟體複製帶 .US 的代碼）
    if ticker.endswith(".US"):
        ticker = ticker[:-3]
    # 別名對照
    if ticker in _TICKER_ALIASES:
        ticker = _TICKER_ALIASES[ticker]
    # 已有明確後綴（.TW / .TWO）→ 直接回傳
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        _resolved_cache[raw] = ticker
        return ticker
    # 台股代碼（純數字）→ 先試上市 .TW，查不到再試上櫃 .TWO
    if ticker.isdigit():
        tw = ticker + ".TW"
        try:
            info = yf.Ticker(tw).info
            if info.get("exchange") or info.get("shortName"):
                _resolved_cache[raw] = tw
                return tw
        except Exception:
            pass
        result = ticker + ".TWO"
        _resolved_cache[raw] = result
        return result
    _resolved_cache[raw] = ticker
    return ticker

# ===========================================================================
# 進場評分系統
# ===========================================================================

def _score_trend(price: float, ma5: float, ma20: float, ma60) -> tuple[int, str]:
    """趨勢分數 (滿分 25)"""
    if ma60 is not None and price > ma5 > ma20 > ma60:
        return 25, "✅ 完美多頭排列 (Price>MA5>MA20>MA60)"
    if price > ma5 > ma20:
        return 18, "✅ 短中期多頭排列 (Price>MA5>MA20)"
    if price > ma20:
        return 12, "🔶 中期趨勢向上 (Price>MA20)"
    if price > ma5:
        return 8, "🔶 短期止穩回升 (Price>MA5)，中期仍偏弱"
    if ma60 is not None and price > ma60:
        return 6, "🔶 長期趨勢尚可，短期偏弱"
    return 0, "❌ 空頭排列，不建議進場"


def _score_momentum(rsi: float) -> tuple[int, str]:
    """動能分數 (滿分 20)"""
    if 30 <= rsi < 40:
        return 20, f"✅ RSI={rsi:.1f} 超賣反彈區 (最佳進場)"
    if rsi < 30:
        return 15, f"🔄 RSI={rsi:.1f} 極度超賣，留意反轉機會"
    if 40 <= rsi < 50:
        return 15, f"✅ RSI={rsi:.1f} 蓄力區 (適合佈局)"
    if 50 <= rsi < 60:
        return 10, f"🔶 RSI={rsi:.1f} 健康上漲"
    if 60 <= rsi < 70:
        return 5, f"🔶 RSI={rsi:.1f} 偏強，追高風險漸增"
    return 0, f"❌ RSI={rsi:.1f} 超買，注意回調"


def _score_macd(macd_line: float, signal_line: float, hist_3d: list[float]) -> tuple[int, str]:
    """MACD 分數 (滿分 20)"""
    golden = macd_line > signal_line
    # 柱狀體連續 3 日放大
    hist_expanding = len(hist_3d) >= 3 and hist_3d[-1] > hist_3d[-2] > hist_3d[-3]
    # 柱狀體縮小（死叉中但即將金叉）
    hist_shrinking = len(hist_3d) >= 3 and abs(hist_3d[-1]) < abs(hist_3d[-2]) < abs(hist_3d[-3])

    if golden and hist_expanding:
        return 20, "✅ MACD 金叉且動能持續增強"
    if golden:
        return 14, "✅ MACD 金叉 (多方佔優)"
    if not golden and hist_shrinking:
        return 8, "🔶 MACD 死叉但空方動能減弱，可能即將金叉"
    return 0, "❌ MACD 死叉，空方主導"


def _score_bollinger(price: float, upper: float, lower: float, middle: float) -> tuple[int, str]:
    """布林通道分數 (滿分 15)"""
    bandwidth = upper - lower
    if bandwidth == 0:
        return 7, "🔶 布林帶寬為零，無法判斷"

    lower_zone = lower + bandwidth * 0.1
    upper_zone = upper - bandwidth * 0.1

    if price <= lower_zone:
        return 15, f"✅ 股價接近布林下軌 (超跌反彈機會)"
    if price <= middle:
        return 10, f"✅ 股價在布林中下區 (相對低位)"
    if price <= upper_zone:
        return 5, f"🔶 股價在布林中上區"
    return 0, f"❌ 股價突破布林上軌 (過熱警戒)"


def _score_volume(vol_ratio: float, price_change_pct: float) -> tuple[int, str]:
    """量能分數 (滿分 20)"""
    if vol_ratio > 1.5 and price_change_pct > 0:
        return 20, f"✅ 放量上漲 (量比{vol_ratio:.1f}x)，多方強勢進攻"
    if 1.0 <= vol_ratio <= 1.5 and price_change_pct > 0:
        return 15, f"✅ 溫和放量上漲 (量比{vol_ratio:.1f}x)"
    if vol_ratio < 0.8:
        return 10, f"🔶 量縮整理 (量比{vol_ratio:.1f}x)，等待方向"
    if vol_ratio > 1.5 and price_change_pct < 0:
        return 0, f"❌ 放量下跌 (量比{vol_ratio:.1f}x)，空方強勢"
    return 5, f"🔶 量能普通 (量比{vol_ratio:.1f}x)"


def _score_reversal(rsi: float, change_pct: float,
                    hist_3d: list[float],
                    price: float, boll_lower: float) -> tuple[int, str]:
    """反轉訊號加分 (最高 +15 Bonus)。偵測底部反轉特徵。"""
    bonus = 0
    details = []

    # ① 超跌止穩 (+5): RSI < 35 且今日收漲 → 止跌訊號
    if rsi < 35 and change_pct > 0:
        bonus += 5
        details.append("超跌止穩 (RSI<35且收漲)")

    # ② MACD 空方動能減弱 (+5): 死叉中但柱狀體回升
    if len(hist_3d) >= 2 and hist_3d[-1] < 0 and hist_3d[-1] > hist_3d[-2]:
        bonus += 5
        details.append("MACD 空方動能減弱")

    # ③ 布林下軌反彈 (+5): 股價接近下軌且收漲
    if boll_lower > 0 and price <= boll_lower * 1.03 and change_pct > 0:
        bonus += 5
        details.append("布林下軌反彈")

    if bonus > 0:
        return bonus, "🔄 " + "、".join(details)
    return 0, "— 無反轉訊號"


def _check_add_position(total_score: int, price: float, ma20: float,
                         rsi: float, vol_ratio: float) -> tuple[bool, str]:
    """判斷是否適合加碼"""
    reasons = []
    conditions_met = 0

    # ① 總分 ≥ 60
    if total_score >= 60:
        conditions_met += 1
        reasons.append(f"總分{total_score}≥60")
    # ② 股價回踩 MA20 附近 (±2%)
    if ma20 > 0 and abs(price - ma20) / ma20 <= 0.02:
        conditions_met += 1
        reasons.append(f"股價接近MA20 (差距{abs(price-ma20)/ma20*100:.1f}%)")
    # ③ RSI 40~55
    if 40 <= rsi <= 55:
        conditions_met += 1
        reasons.append(f"RSI={rsi:.1f}在回調健康區間")
    # ④ 量縮
    if vol_ratio < 0.8:
        conditions_met += 1
        reasons.append(f"量縮(量比{vol_ratio:.1f}x)，賣壓減輕")

    if conditions_met >= 3:
        return True, "🟢 符合加碼條件：" + "、".join(reasons)
    return False, "🔴 不符合加碼條件 (需同時滿足≥3項)"


# ---------------------------------------------------------------------------
# 綜合技術分析 + 評分
# ---------------------------------------------------------------------------
def compute_analysis(df: pd.DataFrame, full: bool = False) -> dict:
    """計算所有技術指標並產出進場評分。full=True 時執行全部 5 個 TA 模組。"""
    close = df["Close"].values.flatten()
    volume = df["Volume"].values.flatten()
    current = float(close[-1])
    prev = float(close[-2]) if len(close) >= 2 else current
    change_pct = (current - prev) / prev * 100

    # --- 使用 feature_engine 計算指標（fallback 到手寫） ---
    # feat 暫存供 predict_stock() 的多變量 LSTM 復用，避免重複計算
    _featured_df = None
    if ENABLE_PANDAS_TA:
        try:
            from feature_engine import compute_features
            feat = compute_features(df)
            _featured_df = feat
            _last = feat.iloc[-1]

            _ma5 = _last.get("ma5", np.nan)
            _ma20 = _last.get("ma20", np.nan)
            ma5 = float(_ma5) if not pd.isna(_ma5) else current
            ma20 = float(_ma20) if not pd.isna(_ma20) else current
            _ma60 = _last.get("ma60", np.nan)
            ma60 = float(_ma60) if (not pd.isna(_ma60)) else (None if len(close) < 60 else current)

            _rsi = _last.get("rsi_14", np.nan)
            rsi = float(_rsi) if not pd.isna(_rsi) else 50.0

            macd_val = float(_last.get("MACD_12_26_9", 0))
            signal_val = float(_last.get("MACDs_12_26_9", 0))
            _hist_col = "MACDh_12_26_9"
            if _hist_col in feat.columns:
                hist_3d = feat[_hist_col].iloc[-3:].tolist() if len(feat) >= 3 else []
            else:
                hist_3d = []

            _bbu = _last.get("BBU_20_2.0", np.nan)
            _bbl = _last.get("BBL_20_2.0", np.nan)
            _bbm = _last.get("BBM_20_2.0", np.nan)
            sma20 = float(_bbm) if not pd.isna(_bbm) else current
            boll_upper = float(_bbu) if not pd.isna(_bbu) else sma20
            boll_lower = float(_bbl) if not pd.isna(_bbl) else sma20

            s_vol = pd.Series(volume)
            _vol_avg_raw = s_vol.rolling(20).mean().iloc[-1]
            vol_avg20 = float(_vol_avg_raw) if not pd.isna(_vol_avg_raw) else float(s_vol.mean())
            vol_today = float(volume[-1])
            vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0
        except Exception as _fe_err:
            _log.warning(f"compute_analysis: feature_engine failed ({_fe_err}), fallback to hand-written")
            ma5, ma20, ma60, rsi, macd_val, signal_val, hist_3d = None, None, None, None, None, None, None
            sma20, boll_upper, boll_lower, vol_avg20, vol_today, vol_ratio = None, None, None, None, None, None
    else:
        ma5 = None  # trigger fallback

    # --- 手寫 fallback（feature_engine 失敗或停用時） ---
    if ma5 is None:
        s_close = pd.Series(close)
        _ma5 = s_close.rolling(5).mean().iloc[-1]
        _ma20 = s_close.rolling(20).mean().iloc[-1]
        ma5 = float(_ma5) if not pd.isna(_ma5) else current
        ma20 = float(_ma20) if not pd.isna(_ma20) else current
        ma60 = float(s_close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else None

        delta = s_close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
        if pd.isna(gain) or pd.isna(loss) or loss == 0:
            rsi = 50.0
        else:
            rsi = float(100 - (100 / (1 + gain / loss)))

        ema12 = s_close.ewm(span=12).mean()
        ema26 = s_close.ewm(span=26).mean()
        macd_line_series = ema12 - ema26
        signal_series = macd_line_series.ewm(span=9).mean()
        hist_series = macd_line_series - signal_series
        macd_val = float(macd_line_series.iloc[-1])
        signal_val = float(signal_series.iloc[-1])
        hist_3d = hist_series.iloc[-3:].tolist() if len(hist_series) >= 3 else []

        _sma20 = s_close.rolling(20).mean().iloc[-1]
        _std20 = s_close.rolling(20).std().iloc[-1]
        sma20 = float(_sma20) if not pd.isna(_sma20) else current
        std20 = float(_std20) if not pd.isna(_std20) else 0.0
        boll_upper = sma20 + 2 * std20
        boll_lower = sma20 - 2 * std20

        s_vol = pd.Series(volume)
        _vol_avg_raw = s_vol.rolling(20).mean().iloc[-1]
        vol_avg20 = float(_vol_avg_raw) if not pd.isna(_vol_avg_raw) else float(s_vol.mean())
        vol_today = float(volume[-1])
        vol_ratio = vol_today / vol_avg20 if vol_avg20 > 0 else 1.0

    # ===== 評分 =====
    trend_score, trend_msg = _score_trend(current, ma5, ma20, ma60)
    momentum_score, momentum_msg = _score_momentum(rsi)
    macd_score, macd_msg = _score_macd(macd_val, signal_val, hist_3d)
    boll_score, boll_msg = _score_bollinger(current, boll_upper, boll_lower, sma20)
    vol_score, vol_msg = _score_volume(vol_ratio, change_pct)
    reversal_score, reversal_msg = _score_reversal(rsi, change_pct, hist_3d,
                                                    current, boll_lower)

    total = (trend_score + momentum_score + macd_score
             + boll_score + vol_score + reversal_score)

    # 進場建議
    if total >= 75:
        verdict = "🟢 強烈買入訊號"
    elif total >= 60:
        verdict = "🟢 適合進場佈局"
    elif total >= 45:
        verdict = "🟡 觀望，等待更好時機"
    elif total >= 30:
        verdict = "🟠 偏弱，不建議進場"
    else:
        verdict = "🔴 遠離，趨勢不佳"

    # 加碼判斷
    add_ok, add_msg = _check_add_position(total, current, ma20, rsi, vol_ratio)

    scores = {
        "trend": {"score": trend_score, "max": 25, "detail": trend_msg},
        "momentum": {"score": momentum_score, "max": 20, "detail": momentum_msg},
        "macd": {"score": macd_score, "max": 20, "detail": macd_msg},
        "bollinger": {"score": boll_score, "max": 15, "detail": boll_msg},
        "volume": {"score": vol_score, "max": 20, "detail": vol_msg},
        "reversal": {"score": reversal_score, "max": 15, "detail": reversal_msg},
    }

    # ===== 進階技術分析疊加層 =====
    ta = compute_ta_overlay(df, full=full)

    result = {
        "current": current,
        "change_pct": change_pct,
        "ma5": ma5, "ma20": ma20, "ma60": ma60,
        "rsi": rsi,
        "macd": macd_val, "macd_signal": signal_val,
        "boll_upper": boll_upper, "boll_lower": boll_lower,
        "vol_today": int(vol_today), "vol_avg20": int(vol_avg20),
        "vol_ratio": vol_ratio,
        "scores": scores,
        "total_score": total,
        "verdict": verdict,
        "add_position": add_ok,
        "add_position_msg": add_msg,
        "_featured_df": _featured_df,  # 內部用：多變量 LSTM 復用已計算的特徵
    }
    result.update(ta)
    return result

# ---------------------------------------------------------------------------
# 快速技術分析（不跑 LSTM，用於高頻監控）
# ---------------------------------------------------------------------------
def quick_analysis(ticker: str) -> dict:
    """只做技術分析評分，不訓練 LSTM 模型。回傳 dict 含 ticker / analysis。"""
    resolved = _resolve_ticker(ticker)
    df = _yf_download(resolved, period="1y", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"找不到股票代碼 **{resolved}** 的資料")
    analysis = compute_analysis(df, full=False)
    return {
        "ticker": resolved.upper(),
        "analysis": analysis,
    }


# ---------------------------------------------------------------------------
# 批次快速分析（一次 yfinance 請求抓多檔）
# ---------------------------------------------------------------------------
def batch_quick_analysis(tickers: list[str]) -> list[dict]:
    """用 yf.download(tickers=[...]) 一次下載多檔，逐一計算技術分析。
    回傳 [{"ticker": str, "analysis": dict | None, "error": str | None}, ...]"""
    if not tickers:
        return []

    resolved = [_resolve_ticker(t) for t in tickers]
    unique = list(dict.fromkeys(resolved))  # 去重保序

    # 批次下載（加 lock 避免與其他 yf 呼叫衝突）
    try:
        with _yf_lock:
            raw = yf.download(unique, period="1y", interval="1d", progress=False, group_by="ticker")
    except Exception as e:
        _log.warning(f"batch_quick_analysis: yf.download failed: {e}")
        raw = pd.DataFrame()

    # 若批次下載完全失敗，限制 fallback 個別下載數量以避免 FD 耗盡
    batch_failed = raw.empty if isinstance(raw, pd.DataFrame) else True
    fallback_count = 0
    _MAX_FALLBACK = 5  # 批次失敗時最多個別下載 5 檔

    results = []
    for tkr in resolved:
        try:
            if len(unique) == 1:
                df = _flatten_yf_columns(raw.copy())
            else:
                df = raw[tkr] if tkr in raw.columns.get_level_values(0) else pd.DataFrame()
                df = _flatten_yf_columns(df.copy()) if not df.empty else df

            if isinstance(df, pd.DataFrame) and not df.empty:
                df = df.dropna(subset=["Close"])

            if df.empty:
                if batch_failed and fallback_count >= _MAX_FALLBACK:
                    _log.warning(f"batch_quick_analysis({tkr}): 跳過 (批次失敗，已達 fallback 上限)")
                    results.append({"ticker": tkr.upper(), "analysis": None, "error": "skipped (batch failed)"})
                    continue
                # fallback: 單獨下載
                df = _yf_download(tkr, period="1y", interval="1d", progress=False)
                fallback_count += 1

            if df.empty:
                _log.warning(f"batch_quick_analysis({tkr}): no data (yfinance 回傳空資料)")
                results.append({"ticker": tkr.upper(), "analysis": None, "error": "no data"})
                continue

            analysis = compute_analysis(df, full=False)
            results.append({"ticker": tkr.upper(), "analysis": analysis, "error": None})
        except Exception as e:
            _log.error(f"batch_quick_analysis({tkr}): {e}")
            results.append({"ticker": tkr.upper(), "analysis": None, "error": str(e)})

    return results


# ---------------------------------------------------------------------------
# 核心預測函式
# ---------------------------------------------------------------------------
def predict_stock(ticker: str) -> dict:
    ticker = _resolve_ticker(ticker)

    # --- 快取檢查 ---
    cached = _pred_cache_get(ticker)
    if cached is not None:
        # 從快取資料重新產圖（matplotlib 快，LSTM 才慢）
        df = _yf_download(ticker, period="1y", interval="1d", progress=False)
        if not df.empty:
            close = df["Close"].values.flatten().astype(np.float32)
            volume = df["Volume"].values.flatten().astype(np.float64)
            chart_buf = _draw_chart(ticker, df.index, close,
                                     cached["predictions"], cached["analysis"], volume)
            return {**cached, "chart_buf": chart_buf}

    df = _yf_download(ticker, period="1y", interval="1d", progress=False)
    if df.empty:
        raise ValueError(f"找不到股票代碼 **{ticker}** 的資料，請確認代碼是否正確。")

    close = df["Close"].values.flatten().astype(np.float32)
    dates = df.index

    # 技術分析 + 評分 (full mode for predict)
    analysis = compute_analysis(df, full=True)

    # LSTM 預測 — 資料不足時跳過訓練，直接用最後收盤價作為預測
    val_mae_price: float | None = None
    use_multivariate = False  # track which mode was used
    min_rows = LOOK_BACK + 5  # 至少需要 35 筆才能建立有效訓練集
    if len(close) < min_rows:
        _log.warning(f"predict_stock({ticker}): 僅有 {len(close)} 筆資料 (需 {min_rows})，略過 LSTM")
        predictions = [float(close[-1])] * PRED_DAYS
    else:
        # 選擇裝置 — MPS 對小 tensor 不穩定，資料量少時回退到 CPU
        device = DEVICE if len(close) >= 60 else torch.device("cpu")

        # --- 嘗試多變量 LSTM（return-based） ---
        if ENABLE_MULTIVARIATE_LSTM:
            try:
                from feature_engine import get_lstm_feature_matrix
                # 復用 compute_analysis() 已計算的特徵，避免重複呼叫 compute_features()
                _cached_feat = analysis.get("_featured_df")
                feat_matrix, scalers = get_lstm_feature_matrix(df, featured=_cached_feat)
                n_features = feat_matrix.shape[1]  # 8

                X_m, y_m = _make_sequences_multi(feat_matrix, LOOK_BACK)
                n_samples = len(X_m)
                split_idx = int(n_samples * 0.8)
                if n_samples - split_idx < 5:
                    split_idx = n_samples
                has_val = split_idx < n_samples

                X_train = torch.tensor(X_m[:split_idx], dtype=torch.float32).to(device)
                y_train = torch.tensor(y_m[:split_idx], dtype=torch.float32).unsqueeze(-1).to(device)
                if has_val:
                    X_val = torch.tensor(X_m[split_idx:], dtype=torch.float32).to(device)
                    y_val = torch.tensor(y_m[split_idx:], dtype=torch.float32).unsqueeze(-1).to(device)

                model = StockLSTM(
                    input_size=n_features,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    dropout=LSTM_DROPOUT,
                ).to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-5
                )

                best_val_loss = float("inf")
                patience_counter = 0
                best_state = None

                model.train()
                for epoch in range(LSTM_MAX_EPOCHS):
                    pred = model(X_train)
                    loss = criterion(pred, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), LSTM_GRAD_CLIP)
                    optimizer.step()

                    if has_val:
                        model.eval()
                        with torch.no_grad():
                            val_pred = model(X_val)
                            val_loss = criterion(val_pred, y_val).item()
                        model.train()
                        scheduler.step(val_loss)

                        # Warmup: 前 N epochs 不做 early stopping
                        if epoch < LSTM_WARMUP_EPOCHS:
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                            continue

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        else:
                            patience_counter += 1
                            if patience_counter >= LSTM_PATIENCE:
                                _log.info(f"predict_stock({ticker}): multi-LSTM early stop at epoch {epoch+1}")
                                break

                if best_state is not None:
                    model.load_state_dict(best_state)

                # Validation MAE（return → 實際價格差）
                ret_vmin, ret_vmax = scalers["close_return"]
                last_close = scalers["_last_close"]
                if has_val:
                    model.eval()
                    with torch.no_grad():
                        val_pred_norm = model(X_val).cpu().numpy().flatten()
                    val_actual_norm = y_val.cpu().numpy().flatten()
                    val_pred_ret = _denormalize(val_pred_norm, ret_vmin, ret_vmax)
                    val_actual_ret = _denormalize(val_actual_norm, ret_vmin, ret_vmax)
                    # MAE in price terms: return * last_close ≈ 價格偏差
                    val_mae_price = float(np.mean(np.abs(val_pred_ret - val_actual_ret)) * last_close)

                # --- 7 天預測（return-based：預測日報酬率，從最後收盤價累積） ---
                model.eval()
                last_window = feat_matrix[-LOOK_BACK:].copy()
                predictions = []
                current_price = last_close
                with torch.no_grad():
                    for _ in range(PRED_DAYS):
                        inp = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).to(device)
                        out_norm = model(inp).item()
                        # 反正規化得到日報酬率
                        pred_return = out_norm * (ret_vmax - ret_vmin) + ret_vmin
                        # 累積到價格
                        current_price = current_price * (1.0 + pred_return)
                        predictions.append(current_price)
                        # Roll window: LOCF + 更新 close_return
                        new_row = last_window[-1].copy()
                        new_row[0] = out_norm  # 正規化後的 return
                        last_window = np.vstack([last_window[1:], new_row])

                use_multivariate = True
                _log.info(f"predict_stock({ticker}): multi-LSTM val_mae=${val_mae_price:.2f}" if val_mae_price else
                           f"predict_stock({ticker}): multi-LSTM no val set")
            except Exception as e:
                _log.warning(f"predict_stock({ticker}): multi-LSTM 失敗 ({e})，回退到單變量")
                use_multivariate = False

        # --- 單變量 LSTM fallback ---
        if not use_multivariate:
            try:
                norm_close, vmin, vmax = _normalize(close)
                X, y = _make_sequences(norm_close, LOOK_BACK)

                n_samples = len(X)
                split_idx = int(n_samples * 0.8)
                if n_samples - split_idx < 5:
                    split_idx = n_samples
                has_val = split_idx < n_samples

                X_train = torch.tensor(X[:split_idx], dtype=torch.float32).unsqueeze(-1).to(device)
                y_train = torch.tensor(y[:split_idx], dtype=torch.float32).unsqueeze(-1).to(device)
                if has_val:
                    X_val = torch.tensor(X[split_idx:], dtype=torch.float32).unsqueeze(-1).to(device)
                    y_val = torch.tensor(y[split_idx:], dtype=torch.float32).unsqueeze(-1).to(device)

                model = StockLSTM().to(device)
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                MAX_EPOCHS = 200
                PATIENCE = 15
                best_val_loss = float("inf")
                patience_counter = 0
                best_state = None

                model.train()
                for epoch in range(MAX_EPOCHS):
                    pred = model(X_train)
                    loss = criterion(pred, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if has_val:
                        model.eval()
                        with torch.no_grad():
                            val_pred = model(X_val)
                            val_loss = criterion(val_pred, y_val).item()
                        model.train()

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_state = {k: v.clone() for k, v in model.state_dict().items()}
                        else:
                            patience_counter += 1
                            if patience_counter >= PATIENCE:
                                _log.info(f"predict_stock({ticker}): early stop at epoch {epoch+1}")
                                break

                if best_state is not None:
                    model.load_state_dict(best_state)

                if has_val:
                    model.eval()
                    with torch.no_grad():
                        val_pred_norm = model(X_val).cpu().numpy().flatten()
                    val_actual_norm = y_val.cpu().numpy().flatten()
                    val_pred_real = _denormalize(val_pred_norm, vmin, vmax)
                    val_actual_real = _denormalize(val_actual_norm, vmin, vmax)
                    val_mae_price = float(np.mean(np.abs(val_pred_real - val_actual_real)))

                model.eval()
                last_seq = norm_close[-LOOK_BACK:].copy()
                predictions_norm = []
                with torch.no_grad():
                    for _ in range(PRED_DAYS):
                        inp = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
                        out = model(inp).item()
                        predictions_norm.append(out)
                        last_seq = np.append(last_seq[1:], out)

                predictions = _denormalize(np.array(predictions_norm), vmin, vmax).tolist()
                _log.info(f"predict_stock({ticker}): uni-LSTM val_mae=${val_mae_price:.2f}" if val_mae_price else
                           f"predict_stock({ticker}): uni-LSTM no val set")
            except Exception as e:
                _log.warning(f"predict_stock({ticker}): LSTM 失敗 ({e})，回退到最後收盤價")
                predictions = [float(close[-1])] * PRED_DAYS

    # --- LSTM 預測完成，存入 lstm_predictions ---
    lstm_predictions = predictions[:]

    # --- Phase 1C: NeuralForecast ensemble ---
    ensemble_method = "lstm"
    nhits_predictions = None
    if ENABLE_NHITS:
        try:
            from forecast_ensemble import forecast_nhits, ensemble_predictions
            nhits_preds = forecast_nhits(df, horizon=PRED_DAYS)
            if nhits_preds is not None:
                predictions = ensemble_predictions(lstm_predictions, nhits_preds)
                nhits_predictions = nhits_preds
                ensemble_method = "lstm+nhits"
                _log.info(f"predict_stock({ticker}): ensemble={ensemble_method}")
        except Exception as e:
            _log.warning(f"predict_stock({ticker}): N-HiTS 失敗 ({e})，使用純 LSTM")

    # --- Phase 1D: FinBERT 新聞情緒 ---
    news_sentiment = {"available": False}
    if ENABLE_SENTIMENT:
        try:
            from sentiment_engine import analyze_sentiment
            news_sentiment = analyze_sentiment(ticker)
        except Exception as e:
            _log.warning(f"predict_stock({ticker}): sentiment 失敗 ({e})")

    # --- Phase 3: Multi-Agent Consensus (Gate 4 pre-computation) ---
    agent_analysis = {"available": False}
    if ENABLE_AGENT_GATE:
        try:
            from agent_gate import gate_agent_consensus, get_agent_summary, get_agent_rating
            g4 = gate_agent_consensus(analysis, ticker)
            rating = get_agent_rating(ticker)
            summary = get_agent_summary(ticker)
            agent_analysis = {
                "available": True,
                "rating": rating or "Hold",
                "summary": summary,
                "gate_result": g4,
            }
        except Exception as e:
            _log.warning(f"predict_stock({ticker}): agent analysis 失敗 ({e})")

    # 記錄預測
    try:
        log_prediction(ticker, float(close[-1]), predictions, val_mae_price)
    except Exception as e:
        _log.warning(f"predict_stock({ticker}): log_prediction failed: {e}")

    volume = df["Volume"].values.flatten().astype(np.float64)
    chart_buf = _draw_chart(ticker, dates, close, predictions, analysis, volume)

    result = {
        "ticker": ticker.upper(),
        "last_price": float(close[-1]),
        "predictions": predictions,
        "lstm_predictions": lstm_predictions,
        "nhits_predictions": nhits_predictions,
        "ensemble_method": ensemble_method,
        "news_sentiment": news_sentiment,
        "agent_analysis": agent_analysis,
        "chart_buf": chart_buf,
        "device_used": str(DEVICE),
        "analysis": analysis,
        "val_mae": val_mae_price,
    }

    # 清除內部暫存欄位（不需要傳到 bot 或快取）
    analysis.pop("_featured_df", None)

    # 快取結果（不含 chart_buf）
    cache_data = {k: v for k, v in result.items() if k != "chart_buf"}
    _pred_cache_set(ticker, cache_data)

    return result

# ---------------------------------------------------------------------------
# 預測準確度追蹤
# ---------------------------------------------------------------------------
def check_prediction_accuracy(ticker: str | None = None) -> dict:
    """
    讀取 predictions_log.csv，針對 ≥10 天前的記錄對照實際價格，回傳準確度統計。
    ticker: 篩選特定股票，None 表示全部。
    回傳:
      { "has_data": bool,
        "overall": { "count", "mae", "mape", "direction_accuracy" },
        "by_ticker": { TICKER: { "count", "mae", "mape", "direction_accuracy" }, ... },
        "details": [ { "predict_time", "ticker", "mae", "mape", "direction_correct", ... }, ... ] }
    """
    if not os.path.exists(_PRED_LOG):
        return {"has_data": False, "overall": {}, "by_ticker": {}, "details": []}

    with open(_PRED_LOG, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return {"has_data": False, "overall": {}, "by_ticker": {}, "details": []}

    # 篩選 ≥10 天前的預測（確保 7 個營業日已過）
    cutoff = datetime.now(_TW) - timedelta(days=10)
    eligible = []
    for row in rows:
        try:
            pt = datetime.strptime(row["predict_time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=_TW)
        except (ValueError, KeyError):
            continue
        if pt > cutoff:
            continue
        if ticker and row.get("ticker", "").upper() != ticker.upper():
            continue
        eligible.append(row)

    if not eligible:
        return {"has_data": False, "overall": {}, "by_ticker": {}, "details": []}

    # 按 ticker 分組，每組只下載一次 yfinance 資料
    from collections import defaultdict
    by_ticker = defaultdict(list)
    for row in eligible:
        by_ticker[row["ticker"]].append(row)

    details = []
    ticker_stats = {}

    for tkr, tkr_rows in by_ticker.items():
        # 找出需要的日期範圍
        earliest_str = min(r["predict_time"] for r in tkr_rows)
        earliest_dt = datetime.strptime(earliest_str, "%Y-%m-%d %H:%M:%S")
        start_date = (earliest_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        end_date = (datetime.now(_TW) + timedelta(days=1)).strftime("%Y-%m-%d")

        try:
            hist = _yf_download(tkr, start=start_date, end=end_date, progress=False)
        except Exception as e:
            _log.warning(f"check_prediction_accuracy: 無法下載 {tkr}: {e}")
            continue
        if hist.empty:
            continue

        hist_close = hist["Close"].values.flatten()
        hist_dates = hist.index.normalize()

        tkr_maes, tkr_mapes, tkr_dirs = [], [], []

        for row in tkr_rows:
            pred_dt = datetime.strptime(row["predict_time"], "%Y-%m-%d %H:%M:%S")
            pred_date = pd.Timestamp(pred_dt.date())

            # 找預測日之後的營業日
            future_mask = hist_dates > pred_date
            future_closes = hist_close[future_mask]
            if len(future_closes) < PRED_DAYS:
                continue  # 還沒有足夠的實際資料

            actual_7d = future_closes[:PRED_DAYS]
            last_price = float(row.get("last_price", 0))

            pred_7d = []
            for i in range(1, PRED_DAYS + 1):
                val = row.get(f"pred_day{i}", "")
                if not val:
                    break
                pred_7d.append(float(val))
            if len(pred_7d) < PRED_DAYS:
                continue

            # MAE
            errors = [abs(p - a) for p, a in zip(pred_7d, actual_7d)]
            mae = sum(errors) / len(errors)

            # MAPE
            pct_errors = [abs(p - a) / a * 100 for p, a in zip(pred_7d, actual_7d) if a != 0]
            mape = sum(pct_errors) / len(pct_errors) if pct_errors else 0.0

            # 方向準確率：第 7 天預測漲/跌 vs 實際漲/跌
            pred_direction = pred_7d[-1] > last_price
            actual_direction = float(actual_7d[-1]) > last_price
            direction_correct = pred_direction == actual_direction

            detail = {
                "predict_time": row["predict_time"],
                "ticker": tkr,
                "last_price": last_price,
                "pred_day7": pred_7d[-1],
                "actual_day7": float(actual_7d[-1]),
                "mae": round(mae, 2),
                "mape": round(mape, 2),
                "direction_correct": direction_correct,
                "model_val_mae": row.get("model_val_mae", ""),
            }
            details.append(detail)
            tkr_maes.append(mae)
            tkr_mapes.append(mape)
            tkr_dirs.append(direction_correct)

        if tkr_maes:
            ticker_stats[tkr] = {
                "count": len(tkr_maes),
                "mae": round(sum(tkr_maes) / len(tkr_maes), 2),
                "mape": round(sum(tkr_mapes) / len(tkr_mapes), 2),
                "direction_accuracy": round(sum(tkr_dirs) / len(tkr_dirs) * 100, 1),
            }

    if not details:
        return {"has_data": False, "overall": {}, "by_ticker": {}, "details": []}

    # 整體統計
    all_maes = [d["mae"] for d in details]
    all_mapes = [d["mape"] for d in details]
    all_dirs = [d["direction_correct"] for d in details]
    overall = {
        "count": len(details),
        "mae": round(sum(all_maes) / len(all_maes), 2),
        "mape": round(sum(all_mapes) / len(all_mapes), 2),
        "direction_accuracy": round(sum(all_dirs) / len(all_dirs) * 100, 1),
    }

    # 按時間倒序
    details.sort(key=lambda d: d["predict_time"], reverse=True)

    return {
        "has_data": True,
        "overall": overall,
        "by_ticker": ticker_stats,
        "details": details,
    }


# ---------------------------------------------------------------------------
# 繪圖
# ---------------------------------------------------------------------------
def _draw_chart(ticker: str, dates, close: np.ndarray, predictions: list,
                ta: dict, volume: np.ndarray) -> io.BytesIO:
    future_dates = pd.bdate_range(start=dates[-1], periods=PRED_DAYS + 1)[1:]
    s_close = pd.Series(close)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), height_ratios=[3, 1.2, 1],
                              gridspec_kw={"hspace": 0.35})
    ax_price, ax_rsi, ax_vol = axes

    # === 上圖：股價 + 均線 + 布林 + 預測 ===
    ax_price.plot(dates, close, label="Close", color="#1f77b4", linewidth=1.5)
    ax_price.plot(future_dates, predictions, label="Predicted (7d)", color="#ff7f0e",
                  linestyle="--", marker="o", markersize=4)

    ma5 = s_close.rolling(5).mean()
    ma20 = s_close.rolling(20).mean()
    ma200 = s_close.rolling(200).mean()
    ax_price.plot(dates, ma5, label="MA5", color="#e377c2", linewidth=0.8, alpha=0.7)
    ax_price.plot(dates, ma20, label="MA20", color="#2ca02c", linewidth=0.8, alpha=0.7)
    ax_price.plot(dates, ma200, label="MA200", color="#ff6600", linewidth=1.0, alpha=0.8,
                  linestyle="--")

    # S/R 水平線
    sr_levels = ta.get("ta_overlay", {}).get("support_resistance", {}).get("levels", [])
    for lv in sr_levels[:6]:  # 最多畫 6 條
        sr_color = "#2ca02c" if lv.get("type") == "support" else "#d62728"
        sr_style = "-" if lv.get("flipped") else ":"
        ax_price.axhline(lv["price"], color=sr_color, linestyle=sr_style,
                         linewidth=0.7, alpha=0.5)

    # 布林通道陰影
    boll_mid = s_close.rolling(20).mean()
    boll_std = s_close.rolling(20).std()
    boll_up = boll_mid + 2 * boll_std
    boll_dn = boll_mid - 2 * boll_std
    ax_price.fill_between(dates, boll_dn, boll_up, alpha=0.08, color="blue", label="Bollinger")

    ax_price.axvline(dates[-1], color="gray", linestyle=":", alpha=0.6)
    score = ta["total_score"]
    # 圖表標題用英文避免中文字型缺失
    if score >= 75:
        verdict_en = "STRONG BUY"
    elif score >= 60:
        verdict_en = "BUY"
    elif score >= 45:
        verdict_en = "HOLD"
    elif score >= 30:
        verdict_en = "WEAK"
    else:
        verdict_en = "AVOID"
    ta_conf = ta.get("ta_confidence", 0)
    ta_conf_max = ta.get("ta_confidence_max", 40)
    ax_price.set_title(
        f"{ticker.upper()} — OpenClaw 1.32  |  Entry Score: {score}/100  [{verdict_en}]  TA: {ta_conf}/{ta_conf_max}",
        fontsize=12,
    )
    ax_price.set_ylabel("Price")
    ax_price.legend(fontsize=7, loc="upper left")
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # === 中圖：RSI ===
    delta = s_close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / loss_s))
    ax_rsi.plot(dates, rsi, color="#9467bd", linewidth=1)
    ax_rsi.axhline(70, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_rsi.axhline(30, color="green", linestyle="--", linewidth=0.7, alpha=0.5)
    ax_rsi.axhspan(30, 40, alpha=0.1, color="green", label="Entry Zone")
    ax_rsi.fill_between(dates, 30, 70, alpha=0.03, color="gray")
    ax_rsi.set_ylabel("RSI(14)")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(fontsize=7, loc="upper left")
    ax_rsi.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    # === 下圖：成交量 (紅漲綠跌) ===
    price_diff = s_close.diff().fillna(0)
    colors = ["#d62728" if d >= 0 else "#2ca02c" for d in price_diff]
    ax_vol.bar(dates, volume, color=colors, alpha=0.7, width=0.8)
    # 20日均量線
    vol_ma20 = pd.Series(volume).rolling(20).mean()
    ax_vol.plot(dates, vol_ma20, color="orange", linewidth=0.8, label="Vol MA20")
    ax_vol.set_ylabel("Volume")
    ax_vol.legend(fontsize=7, loc="upper left")
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    fig.subplots_adjust(hspace=0.4, top=0.94, bottom=0.08)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf
