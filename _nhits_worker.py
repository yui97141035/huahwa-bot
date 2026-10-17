#!/usr/bin/env python3
"""獨立子程序 worker — 訓練+預測 N-HiTS 模型，避免 macOS spawn 問題。

ProcessPoolExecutor 在 macOS spawn 模式下會重新 import 啟動模組(bot.py)，
觸發 bootstrap 錯誤（"An attempt has been made to start a new process
before the current process has finished its bootstrapping phase"）。
改用 subprocess 執行這個獨立腳本，繞開 re-import 整個 bot 的問題
（同 agent_gate.py 用 _agent_worker.py 解決 TradingAgents 的方式）。

用法 (由 forecast_ensemble.py 透過 subprocess 呼叫):
    echo '{"ds":[...],"y":[...],"horizon":7,"max_steps":100}' | python3 _nhits_worker.py

輸入: JSON (stdin) — ds (ISO date strings), y (floats), horizon, max_steps
輸出: JSON (stdout) — {"predictions": [...]}
"""

import json
import sys


def main():
    data = json.loads(sys.stdin.read())
    horizon = data["horizon"]
    max_steps = data["max_steps"]

    import pandas as pd
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    nf_df = pd.DataFrame({
        "unique_id": "stock",
        "ds": pd.to_datetime(data["ds"]),
        "y": [float(v) for v in data["y"]],
    })

    model = NHITS(
        h=horizon,
        input_size=30,
        max_steps=max_steps,
        scaler_type="standard",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq="B")
    nf.fit(df=nf_df)
    forecast = nf.predict()
    predictions = forecast["NHITS"].values.tolist()

    json.dump({"predictions": predictions}, sys.stdout)


if __name__ == "__main__":
    main()
