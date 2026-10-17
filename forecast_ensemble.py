"""
花城 Forecast Ensemble — N-HiTS 預測 + LSTM/N-HiTS 加權平均
使用 NeuralForecast 框架訓練 N-HiTS 模型，與 LSTM 組合為 ensemble。
import 失敗或訓練逾時 → 靜默回傳 None，讓 caller fallback 到純 LSTM。
"""

import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

_log = logging.getLogger("huacheng.forecast_ensemble")
_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_nhits_worker.py")

# ---------------------------------------------------------------------------
# NeuralForecast import（graceful fallback）
# ---------------------------------------------------------------------------
_HAS_NF = False
try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    _HAS_NF = True
    _log.info("forecast_ensemble: neuralforecast loaded")
except ImportError:
    _log.warning("forecast_ensemble: neuralforecast not installed")


def forecast_nhits(df: pd.DataFrame, horizon: int = 7) -> list[float] | None:
    """
    使用 N-HiTS 模型預測未來 horizon 天收盤價。
    df: 含 Close 欄位的 OHLCV DataFrame（index = DatetimeIndex）。
    回傳: list[float] 長度 = horizon，或 None（失敗時）。

    訓練交給獨立子程序(_nhits_worker.py)執行，而不是 ProcessPoolExecutor：
    macOS spawn 模式下 ProcessPoolExecutor 的 worker 會重新 import 啟動模組
    (bot.py)，觸發 "bootstrap phase" 錯誤，導致 N-HiTS 永遠失敗
    （同 agent_gate.py 用 subprocess 解決 TradingAgents 的方式一致）。
    """
    if not _HAS_NF:
        return None

    from prediction_config import NHITS_MAX_STEPS, NHITS_TIMEOUT

    try:
        # NeuralForecast 需要特定的 DataFrame 格式
        close = df["Close"].squeeze()
        ds = pd.to_datetime(close.index).strftime("%Y-%m-%d").tolist()
        y = close.values.astype(float).tolist()

        payload = json.dumps({
            "ds": ds,
            "y": y,
            "horizon": horizon,
            "max_steps": NHITS_MAX_STEPS,
        })

        proc = subprocess.run(
            [sys.executable, _WORKER_SCRIPT],
            input=payload,
            capture_output=True,
            text=True,
            timeout=NHITS_TIMEOUT,
            env={**os.environ},
        )

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip()[-500:] if proc.stderr else f"exit code {proc.returncode}")

        result = json.loads(proc.stdout)["predictions"]
        if len(result) == horizon:
            _log.info(f"forecast_nhits: success, {horizon} days predicted")
            return result
        else:
            _log.warning(f"forecast_nhits: unexpected output length {len(result)}")
            return None

    except subprocess.TimeoutExpired:
        _log.warning(f"forecast_nhits: timeout ({NHITS_TIMEOUT}s)")
        return None

    except Exception as e:
        _log.warning(f"forecast_nhits: failed ({e})")
        return None


def ensemble_predictions(lstm_preds: list[float], nhits_preds: list[float]) -> list[float]:
    """
    加權平均 LSTM 和 N-HiTS 預測。
    權重由 prediction_config 控制（預設 60/40）。
    """
    from prediction_config import ENSEMBLE_LSTM_WEIGHT, ENSEMBLE_NHITS_WEIGHT

    # 確保長度一致
    n = min(len(lstm_preds), len(nhits_preds))
    result = []
    for i in range(n):
        combined = lstm_preds[i] * ENSEMBLE_LSTM_WEIGHT + nhits_preds[i] * ENSEMBLE_NHITS_WEIGHT
        result.append(combined)

    # 如果 lstm_preds 比較長，補上剩餘（不太可能但防禦性處理）
    for i in range(n, len(lstm_preds)):
        result.append(lstm_preds[i])

    return result
