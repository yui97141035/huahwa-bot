"""
OpenClaw 預測系統設定 — 功能開關 + 超參數
所有設定可透過環境變數覆寫（大寫，前綴 OPENCLAW_）。
"""

import os

def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(f"OPENCLAW_{key}", "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default

def _env_int(key: str, default: int) -> int:
    val = os.environ.get(f"OPENCLAW_{key}", "").strip()
    if val.isdigit():
        return int(val)
    return default

def _env_float(key: str, default: float) -> float:
    val = os.environ.get(f"OPENCLAW_{key}", "").strip()
    try:
        return float(val) if val else default
    except ValueError:
        return default

def _env_str(key: str, default: str) -> str:
    val = os.environ.get(f"OPENCLAW_{key}", "").strip()
    return val if val else default

# ---------------------------------------------------------------------------
# Phase 1A: Feature Engine
# ---------------------------------------------------------------------------
ENABLE_PANDAS_TA = _env_bool("ENABLE_PANDAS_TA", True)

# ---------------------------------------------------------------------------
# Phase 1B: Multivariate LSTM
# ---------------------------------------------------------------------------
ENABLE_MULTIVARIATE_LSTM = _env_bool("ENABLE_MULTIVARIATE_LSTM", True)

LSTM_INPUT_SIZE = _env_int("LSTM_INPUT_SIZE", 8)
LSTM_HIDDEN_SIZE = _env_int("LSTM_HIDDEN_SIZE", 128)
LSTM_DROPOUT = _env_float("LSTM_DROPOUT", 0.2)
LSTM_LOOK_BACK = _env_int("LSTM_LOOK_BACK", 30)
LSTM_PRED_DAYS = _env_int("LSTM_PRED_DAYS", 7)
LSTM_MAX_EPOCHS = _env_int("LSTM_MAX_EPOCHS", 200)
LSTM_PATIENCE = _env_int("LSTM_PATIENCE", 20)
LSTM_LR = _env_float("LSTM_LR", 0.002)
LSTM_GRAD_CLIP = _env_float("LSTM_GRAD_CLIP", 1.0)
LSTM_WARMUP_EPOCHS = _env_int("LSTM_WARMUP_EPOCHS", 30)

# LSTM 特徵欄位（順序固定，第 0 欄 = 預測 target）
# close_return: 日報酬率（取代 close_norm，消除 mean-reversion bias）
# return_5d: 5 日累計報酬率（動量信號）
LSTM_FEATURES = [
    "close_return", "return_5d", "rsi_14", "macd_hist", "bb_pct",
    "vol_ratio", "atr_14", "adx_14",
]

# ---------------------------------------------------------------------------
# Phase 1C: NeuralForecast Ensemble
# ---------------------------------------------------------------------------
ENABLE_NHITS = _env_bool("ENABLE_NHITS", True)

NHITS_MAX_STEPS = _env_int("NHITS_MAX_STEPS", 100)
NHITS_TIMEOUT = _env_int("NHITS_TIMEOUT", 15)  # seconds
ENSEMBLE_LSTM_WEIGHT = _env_float("ENSEMBLE_LSTM_WEIGHT", 0.6)
ENSEMBLE_NHITS_WEIGHT = _env_float("ENSEMBLE_NHITS_WEIGHT", 0.4)

# ---------------------------------------------------------------------------
# Phase 1D: FinBERT Sentiment
# ---------------------------------------------------------------------------
ENABLE_SENTIMENT = _env_bool("ENABLE_SENTIMENT", True)

SENTIMENT_MAX_HEADLINES = _env_int("SENTIMENT_MAX_HEADLINES", 10)
SENTIMENT_DISPLAY_TOP = _env_int("SENTIMENT_DISPLAY_TOP", 3)

# ---------------------------------------------------------------------------
# Phase 1E: QuantStats
# ---------------------------------------------------------------------------
ENABLE_QUANTSTATS = _env_bool("ENABLE_QUANTSTATS", True)

# ---------------------------------------------------------------------------
# Phase 2: Triple Verification Gates (三驗證進場訊號)
# ---------------------------------------------------------------------------
ENABLE_TRIPLE_GATE = _env_bool("ENABLE_TRIPLE_GATE", True)

# Gate 1: 校準技術面閾值（per category）
GATE1_THRESHOLD_ETF_TW = _env_int("GATE1_THRESHOLD_ETF_TW", 45)
GATE1_THRESHOLD_STOCK_TW = _env_int("GATE1_THRESHOLD_STOCK_TW", 45)
GATE1_THRESHOLD_STOCK_US = _env_int("GATE1_THRESHOLD_STOCK_US", 45)
GATE1_THRESHOLD_VOLATILE_US = _env_int("GATE1_THRESHOLD_VOLATILE_US", 45)
GATE1_RSI_MAX = _env_int("GATE1_RSI_MAX", 75)
GATE1_ADX_MIN = _env_int("GATE1_ADX_MIN", 15)

# Gate 2: ML 方向分類器
GATE2_PROBABILITY_THRESHOLD = _env_float("GATE2_PROBABILITY_THRESHOLD", 0.7)
GATE2_TRAIN_WINDOW = _env_int("GATE2_TRAIN_WINDOW", 120)   # 訓練用天數
GATE2_FORWARD_DAYS = _env_int("GATE2_FORWARD_DAYS", 5)      # 預測 N 天後漲跌

# Gate 3: 總體環境
GATE3_VIX_MAX = _env_float("GATE3_VIX_MAX", 30.0)
GATE3_SENTIMENT_MIN = _env_float("GATE3_SENTIMENT_MIN", -0.3)

# ---------------------------------------------------------------------------
# Phase 3: Multi-Agent Consensus Gate (Gate 4)
# ---------------------------------------------------------------------------
ENABLE_AGENT_GATE = _env_bool("ENABLE_AGENT_GATE", True)
AGENT_GATE_TIMEOUT = _env_int("AGENT_GATE_TIMEOUT", 180)   # seconds
AGENT_CACHE_TTL = _env_int("AGENT_CACHE_TTL", 14400)       # 4 hours
AGENT_LLM_PROVIDER = _env_str("AGENT_LLM_PROVIDER", "google")
AGENT_LLM_MODEL = _env_str("AGENT_LLM_MODEL", "gemini-2.5-flash-lite")
