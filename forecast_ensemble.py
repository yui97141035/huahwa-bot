"""
花城 Forecast Ensemble — N-HiTS 預測 + LSTM/N-HiTS 加權平均
使用 NeuralForecast 框架訓練 N-HiTS 模型，與 LSTM 組合為 ensemble。
import 失敗或訓練逾時 → 靜默回傳 None，讓 caller fallback 到純 LSTM。
"""

import logging
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError

# macOS + PyTorch: fork() 會複製 CUDA/MPS context 導致 crash，
# 必須用 "spawn" 建立乾淨的子進程。
try:
    multiprocessing.set_start_method("spawn", force=False)
except RuntimeError:
    pass  # 已經被設定過（例如在 __main__ 中）

_log = logging.getLogger("huacheng.forecast_ensemble")

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


def _nhits_train_predict(nf_df: pd.DataFrame, horizon: int, max_steps: int) -> list[float]:
    """在子進程中執行 N-HiTS 訓練+預測（必須是 top-level function 才能被 pickle）。"""
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS
    model = NHITS(
        h=horizon,
        input_size=30,
        max_steps=max_steps,
        scaler_type="standard",
    )
    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(df=nf_df)
    forecast = nf.predict()
    return forecast["NHITS"].values.tolist()


def forecast_nhits(df: pd.DataFrame, horizon: int = 7) -> list[float] | None:
    """
    使用 N-HiTS 模型預測未來 horizon 天收盤價。
    df: 含 Close 欄位的 OHLCV DataFrame（index = DatetimeIndex）。
    回傳: list[float] 長度 = horizon，或 None（失敗時）。
    """
    if not _HAS_NF:
        return None

    from prediction_config import NHITS_MAX_STEPS, NHITS_TIMEOUT

    try:
        # NeuralForecast 需要特定的 DataFrame 格式
        close = df["Close"].squeeze()
        nf_df = pd.DataFrame({
            "unique_id": "stock",
            "ds": close.index,
            "y": close.values.astype(float),
        })
        nf_df["ds"] = pd.to_datetime(nf_df["ds"])

        # 用 ProcessPoolExecutor + timeout 控制訓練時間
        # ProcessPoolExecutor 可以在超時後真正終止 worker（ThreadPool 做不到）
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                _nhits_train_predict, nf_df, horizon, NHITS_MAX_STEPS
            )
            try:
                result = future.result(timeout=NHITS_TIMEOUT)
                if len(result) == horizon:
                    _log.info(f"forecast_nhits: success, {horizon} days predicted")
                    return result
                else:
                    _log.warning(f"forecast_nhits: unexpected output length {len(result)}")
                    return None
            except FuturesTimeoutError:
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
