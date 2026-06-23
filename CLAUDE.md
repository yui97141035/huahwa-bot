# 花城 (huahwa-bot) 開發規則

## 已踩過的坑

### 1. 特徵雙重正規化 Bug (2026-05-11)
**場景**: LSTM 特徵矩陣中的 `close_norm` 先被 `compute_features()` min-max 正規化到 [0,1]，再被 `get_lstm_feature_matrix()` 做第二次 min-max。結果 scaler 存的是 (~0, ~1) 而非原始價格範圍，`_denormalize()` 無法還原為真實價格。
**教訓**: 當 pipeline 有多層正規化時，必須追蹤 scaler 對應的原始值域。寫完正規化鏈後，立刻寫一個 round-trip test 驗證：`denormalize(normalize(x)) ≈ x`。

### 2. 重複計算浪費效能 (2026-05-11)
**場景**: `compute_analysis()` 和 `get_lstm_feature_matrix()` 各自呼叫 `compute_features()`，同一組指標算了兩次。
**教訓**: 把中間計算結果暫存傳遞（但記得清除，不要讓大 DataFrame 跑進快取或跨模組傳輸）。

### 3. ThreadPoolExecutor timeout 不殺執行緒 (2026-05-11)
**場景**: N-HiTS 用 ThreadPoolExecutor + timeout，超時後 thread 繼續跑，白耗資源。
**教訓**: Python ThreadPool 的 timeout 只是放棄等待，不殺 worker。需要真正終止計算時用 `ProcessPoolExecutor`（注意 macOS 要 `set_start_method("spawn")`）。

### 4. pandas-ta Python 版本限制 (2026-05-11)
**場景**: pandas-ta 0.4.x 要求 Python 3.12+，在 3.11 上完全無法安裝。
**教訓**: 所有可選依賴必須有 fallback 路徑，requirements.txt 用 environment marker（`; python_version>="3.12"`）限制。

## 架構規則

- 每個新元件都必須有 `try/except` + feature flag，失敗不影響現有功能
- `predict_stock()` 回傳 dict 結構只做 additive 擴充，不改動既有欄位
- 不動的檔案: `ta_modules.py`, `market_data.py`, `arena.py`
- 測試: 有改動就跑 `python3 -m pytest tests/ -v`
- Phase 2 三驗證系統由 `ENABLE_TRIPLE_GATE` 控制，False 時回退到舊 score≥60 邏輯
- `signal_gate.py` 的四道 Gate 各自有 fallback（LightGBM 不裝 → Gate2 auto-pass, VIX/情緒取不到 → 子條件 auto-pass, TradingAgents 超時/錯誤 → Gate4 auto-pass）
- Phase 3 Gate 4 由 `ENABLE_AGENT_GATE` 控制，False 或異常時 auto-pass
- Gate 4 使用 ProcessPoolExecutor + 180s timeout，結果快取 4 小時
- TradingAgents 安裝為 editable package（`pip install -e /Users/yui/TradingAgents`）
- 部署四驗證前必須先跑 `python3 gate_backtest.py` 確認準確率 > 70%
