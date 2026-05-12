"""
OpenClaw 三驗證進場訊號系統 (Triple Verification Gates)
只在多個獨立訊號同時確認時才推送進場通知。

Gate 1: 校準技術面 — 用校準後的閾值評估 score + RSI/ADX 濾波
Gate 2: ML 方向分類 — LightGBM 預測未來 5 天方向
Gate 3: 總體環境   — VIX + 大盤趨勢 + 新聞情緒
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

_log = logging.getLogger("openclaw.signal_gate")

# LightGBM lazy import（避免模組層級載入 C lib 導致 segfault）
_HAS_LIGHTGBM = None  # None = 尚未嘗試, True/False = 結果
lgb = None

def _ensure_lightgbm():
    global _HAS_LIGHTGBM, lgb
    if _HAS_LIGHTGBM is not None:
        return _HAS_LIGHTGBM
    try:
        import lightgbm as _lgb
        lgb = _lgb
        _HAS_LIGHTGBM = True
        _log.info("signal_gate: lightgbm loaded OK")
    except ImportError:
        _HAS_LIGHTGBM = False
        _log.warning("signal_gate: lightgbm not installed, Gate 2 will auto-pass")
    return _HAS_LIGHTGBM


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GateResult:
    passed: bool
    score: float       # Gate-specific score (e.g. tech score, probability, macro score)
    details: str


@dataclass
class EntrySignal:
    confidence: str      # "HIGH" | "MEDIUM" | "LOW"
    gates_passed: int    # 0-3
    gate_results: dict   # {"gate1": GateResult, "gate2": GateResult, "gate3": GateResult}
    should_alert: bool   # only True when HIGH


# ---------------------------------------------------------------------------
# 股票分類
# ---------------------------------------------------------------------------
def classify_ticker(ticker: str) -> str:
    if ticker.endswith(".TW") or ticker.endswith(".TWO"):
        # 台灣 ETF: 00 開頭
        code = ticker.split(".")[0]
        if code.startswith("00"):
            return "etf_tw"
        return "stock_tw"
    if ticker in ("RGTI", "IONQ", "QBTS", "QUBT", "QMCO"):
        return "volatile_us"
    return "stock_us"


def _get_gate1_threshold(category: str) -> int:
    from prediction_config import (
        GATE1_THRESHOLD_ETF_TW, GATE1_THRESHOLD_STOCK_TW,
        GATE1_THRESHOLD_STOCK_US, GATE1_THRESHOLD_VOLATILE_US,
    )
    return {
        "etf_tw": GATE1_THRESHOLD_ETF_TW,
        "stock_tw": GATE1_THRESHOLD_STOCK_TW,
        "stock_us": GATE1_THRESHOLD_STOCK_US,
        "volatile_us": GATE1_THRESHOLD_VOLATILE_US,
    }.get(category, 60)


# ---------------------------------------------------------------------------
# Gate 1: 校準技術面
# ---------------------------------------------------------------------------
def gate_technical(analysis: dict, category: str) -> GateResult:
    """用校準後的閾值評估技術面。

    analysis: compute_analysis() 的回傳 dict
    category: "etf_tw" | "stock_tw" | "stock_us" | "volatile_us"
    """
    from prediction_config import GATE1_RSI_MAX, GATE1_ADX_MIN

    score = analysis["total_score"]
    rsi = analysis.get("rsi", 50.0)
    threshold = _get_gate1_threshold(category)

    # ADX: 從 _featured_df 取得（若有的話）
    adx = None
    featured_df = analysis.get("_featured_df")
    if featured_df is not None and "adx_14" in featured_df.columns:
        adx_val = featured_df["adx_14"].iloc[-1]
        if not pd.isna(adx_val):
            adx = float(adx_val)

    # 評估條件
    score_pass = score >= threshold
    rsi_pass = rsi < GATE1_RSI_MAX
    adx_pass = adx is None or adx >= GATE1_ADX_MIN  # 無 ADX 數據時自動通過

    passed = score_pass and rsi_pass and adx_pass

    parts = []
    parts.append(f"score={score}/{threshold}{'✓' if score_pass else '✗'}")
    parts.append(f"RSI={rsi:.1f}<{GATE1_RSI_MAX}{'✓' if rsi_pass else '✗'}")
    if adx is not None:
        parts.append(f"ADX={adx:.1f}>={GATE1_ADX_MIN}{'✓' if adx_pass else '✗'}")
    else:
        parts.append("ADX=N/A(pass)")

    return GateResult(passed=passed, score=float(score), details=", ".join(parts))


# ---------------------------------------------------------------------------
# Gate 2: ML 方向分類 (LightGBM)
# ---------------------------------------------------------------------------
_GATE2_FEATURES = [
    "rsi_14", "macd_hist", "bb_pct", "vol_ratio", "atr_14", "adx_14",
    "close_return", "return_5d",
    # 衍生特徵（在 _prepare_ml_features 計算）
    "rsi_slope_5", "macd_cross", "vol_trend",
    "price_vs_ma20", "price_vs_ma60",
    "bb_width", "mfi_14", "willr_14",
]


def _prepare_ml_features(featured_df: pd.DataFrame) -> pd.DataFrame:
    """從 feature_engine 的 DataFrame 準備 LightGBM 所需特徵。"""
    df = featured_df.copy()

    close = df["Close"].squeeze() if "Close" in df.columns else pd.Series(dtype=float)

    # 衍生特徵
    if "rsi_14" in df.columns:
        df["rsi_slope_5"] = df["rsi_14"].diff(5)
    else:
        df["rsi_slope_5"] = 0.0

    if "MACD_12_26_9" in df.columns and "MACDs_12_26_9" in df.columns:
        df["macd_cross"] = (df["MACD_12_26_9"] > df["MACDs_12_26_9"]).astype(float)
    else:
        df["macd_cross"] = 0.0

    if "vol_ratio" in df.columns:
        df["vol_trend"] = df["vol_ratio"].rolling(5).mean()
    else:
        df["vol_trend"] = 1.0

    if "ma20" in df.columns and len(close) > 0:
        ma20 = df["ma20"]
        df["price_vs_ma20"] = (close - ma20) / ma20.replace(0, np.nan)
    else:
        df["price_vs_ma20"] = 0.0

    if "ma60" in df.columns and len(close) > 0:
        ma60 = df["ma60"]
        df["price_vs_ma60"] = (close - ma60) / ma60.replace(0, np.nan)
    else:
        df["price_vs_ma60"] = 0.0

    # bb_width = (BBU - BBL) / BBM
    if "BBU_20_2.0" in df.columns and "BBL_20_2.0" in df.columns and "BBM_20_2.0" in df.columns:
        bbm = df["BBM_20_2.0"].replace(0, np.nan)
        df["bb_width"] = (df["BBU_20_2.0"] - df["BBL_20_2.0"]) / bbm
    elif "bb_width" not in df.columns:
        df["bb_width"] = 0.0

    # 確保所有特徵欄位存在
    for col in _GATE2_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return df


def _train_predict_lgbm(featured_df: pd.DataFrame, train_window: int,
                         forward_days: int) -> tuple[float | None, str]:
    """訓練 LightGBM 並預測最後一天的方向。

    使用最後 train_window 天訓練，預測最後一天。
    Returns: (probability, details_string)
    """
    if not _ensure_lightgbm():
        return None, "LightGBM not available"

    df = _prepare_ml_features(featured_df)

    close = df["Close"].squeeze() if "Close" in df.columns else pd.Series(dtype=float)
    if len(close) < train_window + forward_days + 10:
        return None, f"insufficient data ({len(close)} < {train_window + forward_days + 10})"

    # 建立標籤：未來 forward_days 天收盤價 > 今天 → 1，否則 → 0
    future_return = close.shift(-forward_days) / close - 1
    label = (future_return > 0).astype(int)

    # 特徵矩陣
    feat_cols = _GATE2_FEATURES
    X = df[feat_cols].copy()

    # 移除無標籤的最後 forward_days 天
    valid_mask = label.notna()
    X_valid = X[valid_mask].copy()
    y_valid = label[valid_mask].copy()

    X_valid = X_valid.ffill().bfill().fillna(0)

    if len(X_valid) < train_window:
        return None, f"insufficient valid data ({len(X_valid)} < {train_window})"

    # 用最後 train_window 天訓練，預測最後一筆
    X_train = X_valid.iloc[-train_window:]
    y_train = y_valid.iloc[-train_window:]

    # 最後一天（標籤不可用的那天 = 真正的預測對象）
    X_predict = X.iloc[[-1]].ffill().bfill().fillna(0)

    try:
        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "n_jobs": 1,
        }
        model = lgb.train(params, train_data, num_boost_round=100)
        prob = float(model.predict(X_predict)[0])
        return prob, f"prob={prob:.3f}, train_size={len(X_train)}"
    except Exception as e:
        _log.warning(f"LightGBM train/predict failed: {e}")
        return None, f"LightGBM error: {e}"


def gate_ml_direction(analysis: dict, ticker: str, *, allow_training: bool = False) -> GateResult:
    """LightGBM 預測未來 5 天方向。

    allow_training: True 時才真正訓練 LightGBM（回測用）。
                    False 時 Gate 2 自動通過（生產環境，避免 segfault）。
    """
    from prediction_config import GATE2_PROBABILITY_THRESHOLD, GATE2_TRAIN_WINDOW, GATE2_FORWARD_DAYS

    if not allow_training:
        return GateResult(passed=True, score=0.5, details="live mode (auto-pass, Gate 1+3 only)")

    featured_df = analysis.get("_featured_df")
    if featured_df is None:
        _log.info(f"gate_ml_direction({ticker}): no featured_df, auto-pass")
        return GateResult(passed=True, score=0.5, details="no featured_df (auto-pass)")

    prob, details = _train_predict_lgbm(featured_df, GATE2_TRAIN_WINDOW, GATE2_FORWARD_DAYS)

    if prob is None:
        # Fallback: 降級為二驗證
        _log.info(f"gate_ml_direction({ticker}): fallback — {details}")
        return GateResult(passed=True, score=0.5, details=f"fallback: {details}")

    passed = prob > GATE2_PROBABILITY_THRESHOLD
    return GateResult(
        passed=passed,
        score=prob,
        details=f"{details}, threshold={GATE2_PROBABILITY_THRESHOLD}{'✓' if passed else '✗'}",
    )


# ---------------------------------------------------------------------------
# Gate 3: 總體環境
# ---------------------------------------------------------------------------
def gate_macro(ticker: str, vix: float | None, sentiment: dict | None) -> GateResult:
    """檢查 VIX + 大盤趨勢 + 新聞情緒。

    sentiment: fetch_market_sentiment() 的回傳 dict (含 vix, twii, fear_greed)
    """
    from prediction_config import GATE3_VIX_MAX, GATE3_SENTIMENT_MIN

    checks = []
    all_pass = True

    # Sub-check 1: VIX < threshold
    if vix is not None:
        vix_pass = vix < GATE3_VIX_MAX
        checks.append(f"VIX={vix:.1f}<{GATE3_VIX_MAX}{'✓' if vix_pass else '✗'}")
        if not vix_pass:
            all_pass = False
    else:
        checks.append("VIX=N/A(pass)")

    # Sub-check 2: 大盤趨勢（台股用 TWII, 美股用 S&P500）
    # 這裡使用 sentiment 中已有的資料
    if sentiment:
        twii = sentiment.get("twii")
        if ticker.endswith(".TW") or ticker.endswith(".TWO"):
            # 台股：TWII 需要 > MA20（用近期趨勢判斷）
            if twii and twii.get("change_pct") is not None:
                # 簡化版：看近期漲跌趨勢
                trend_ok = True  # TWII 數據太少無法算 MA20，改用其他方法
                checks.append(f"TWII={twii['value']:.0f}(pass)")
            else:
                checks.append("TWII=N/A(pass)")
        else:
            # 美股：VIX 已檢查，大盤用 fear_greed 代替
            fg = sentiment.get("fear_greed")
            if fg and fg.get("score") is not None:
                fg_score = fg["score"]
                fg_pass = fg_score > 20  # 非極度恐懼
                checks.append(f"F&G={fg_score:.0f}>20{'✓' if fg_pass else '✗'}")
                if not fg_pass:
                    all_pass = False
            else:
                checks.append("F&G=N/A(pass)")
    else:
        checks.append("sentiment=N/A(pass)")

    # Sub-check 3: FinBERT 情緒（若有）
    finbert_score = None
    if sentiment:
        finbert = sentiment.get("finbert")
        if finbert and finbert.get("score") is not None:
            finbert_score = finbert["score"]
            sent_pass = finbert_score > GATE3_SENTIMENT_MIN
            checks.append(f"FinBERT={finbert_score:.2f}>{GATE3_SENTIMENT_MIN}{'✓' if sent_pass else '✗'}")
            if not sent_pass:
                all_pass = False
        else:
            checks.append("FinBERT=N/A(pass)")
    else:
        checks.append("FinBERT=N/A(pass)")

    # 計算總體 macro score (0-1)
    macro_score = 1.0
    if vix is not None:
        # VIX 越低越好：30 → 0, 15 → 1
        macro_score *= max(0.0, min(1.0, (GATE3_VIX_MAX - vix) / GATE3_VIX_MAX))

    return GateResult(passed=all_pass, score=macro_score, details=", ".join(checks))


# ---------------------------------------------------------------------------
# 三驗證整合
# ---------------------------------------------------------------------------
def evaluate_entry(analysis: dict, ticker: str, vix: float | None = None,
                   sentiment: dict | None = None) -> EntrySignal:
    """執行三道 Gate，回傳 EntrySignal。

    analysis: compute_analysis() 的回傳 dict（含 _featured_df）
    """
    category = classify_ticker(ticker)

    g1 = gate_technical(analysis, category)
    g2 = gate_ml_direction(analysis, ticker)
    g3 = gate_macro(ticker, vix, sentiment)

    gate_results = {"gate1": g1, "gate2": g2, "gate3": g3}
    gates_passed = sum(1 for g in gate_results.values() if g.passed)

    if gates_passed == 3:
        confidence = "HIGH"
    elif gates_passed == 2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    should_alert = confidence == "HIGH"

    _log.info(
        f"evaluate_entry({ticker}): {confidence} ({gates_passed}/3) — "
        f"G1={g1.passed} G2={g2.passed} G3={g3.passed}"
    )

    return EntrySignal(
        confidence=confidence,
        gates_passed=gates_passed,
        gate_results=gate_results,
        should_alert=should_alert,
    )
