#!/usr/bin/env python3
"""
OpenClaw 三驗證回測腳本 — Walk-forward 驗證準確率

用法:
    python3 gate_backtest.py                  # 全部監控股票
    python3 gate_backtest.py 0050.TW 2330.TW  # 指定股票
    python3 gate_backtest.py --calibrate      # 校準閾值
"""

import sys
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
warnings.filterwarnings("ignore", category=FutureWarning)
_log = logging.getLogger("openclaw.gate_backtest")

# LightGBM graceful fallback
_HAS_LIGHTGBM = False
try:
    import lightgbm as lgb
    _HAS_LIGHTGBM = True
except ImportError:
    pass


@dataclass
class GateBacktestResult:
    ticker: str
    total_days: int = 0
    gate1_accuracy: float = 0.0
    gate2_accuracy: float = 0.0
    gate3_accuracy: float = 0.0
    combined_accuracy: float = 0.0
    combined_signals: int = 0
    avg_return_on_signal: float = 0.0
    baseline_accuracy: float = 0.0
    baseline_avg_return: float = 0.0


def _download_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """下載歷史資料並計算特徵。"""
    from feature_engine import compute_features
    df = yf.download(ticker, period=period, interval="1d", progress=False)
    if df.empty:
        return pd.DataFrame()
    # 扁平化多層 columns（yfinance 有時回傳 MultiIndex）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    if len(df) < 150:
        return pd.DataFrame()
    featured = compute_features(df)
    return featured


def _evaluate_gate1(row: pd.Series, threshold: int,
                    rsi_max: int, adx_min: int, score: float) -> bool:
    """評估 Gate 1 通過與否（不含 compute_analysis，用簡化版）。"""
    rsi = row.get("rsi_14", 50.0)
    if pd.isna(rsi):
        rsi = 50.0
    adx = row.get("adx_14")
    adx_pass = True
    if adx is not None and not pd.isna(adx):
        adx_pass = float(adx) >= adx_min
    return score >= threshold and rsi < rsi_max and adx_pass


def _compute_simple_score(row: pd.Series, close: float) -> float:
    """簡化的技術評分（模擬 compute_analysis 的 total_score）。"""
    score = 0.0
    rsi = row.get("rsi_14", 50.0)
    if pd.isna(rsi):
        rsi = 50.0

    # 趨勢 (0-25)
    ma5 = row.get("ma5", close)
    ma20 = row.get("ma20", close)
    ma60 = row.get("ma60", close)
    if pd.isna(ma5):
        ma5 = close
    if pd.isna(ma20):
        ma20 = close
    if pd.isna(ma60):
        ma60 = close

    if close > ma5 > ma20 > ma60:
        score += 25
    elif close > ma5 > ma20:
        score += 18
    elif close > ma20:
        score += 12
    elif close > ma5:
        score += 8

    # RSI 動能 (0-20)
    if 30 <= rsi <= 40:
        score += 20
    elif rsi < 30:
        score += 15
    elif 40 < rsi <= 50:
        score += 15
    elif 50 < rsi <= 60:
        score += 10
    elif 60 < rsi <= 70:
        score += 5

    # MACD (0-20)
    macd_hist = row.get("macd_hist", 0)
    if pd.isna(macd_hist):
        macd_hist = 0
    macd_val = row.get("MACD_12_26_9", 0)
    signal_val = row.get("MACDs_12_26_9", 0)
    if pd.isna(macd_val):
        macd_val = 0
    if pd.isna(signal_val):
        signal_val = 0
    if macd_val > signal_val:
        score += 14
    elif macd_hist > 0:
        score += 8

    # 布林 (0-15)
    bb_pct = row.get("bb_pct", 0.5)
    if pd.isna(bb_pct):
        bb_pct = 0.5
    if bb_pct < 0.1:
        score += 15
    elif bb_pct < 0.5:
        score += 10
    elif bb_pct < 0.9:
        score += 5

    # 量能 (0-20)
    vol_ratio = row.get("vol_ratio", 1.0)
    if pd.isna(vol_ratio):
        vol_ratio = 1.0
    close_return = row.get("close_return", 0)
    if pd.isna(close_return):
        close_return = 0
    if vol_ratio > 1.5 and close_return > 0:
        score += 20
    elif 1.0 <= vol_ratio <= 1.5 and close_return > 0:
        score += 15
    elif vol_ratio < 0.8:
        score += 10

    return score


def _evaluate_gate3_historical(vix_value: float | None) -> bool:
    """簡化版 Gate 3，回測用（只看 VIX）。"""
    from prediction_config import GATE3_VIX_MAX
    if vix_value is None:
        return True
    return vix_value < GATE3_VIX_MAX


def backtest_gates(ticker: str, period: str = "2y") -> GateBacktestResult | None:
    """Walk-forward 回測三驗證系統。"""
    from prediction_config import (
        GATE1_RSI_MAX, GATE1_ADX_MIN,
        GATE2_PROBABILITY_THRESHOLD, GATE2_TRAIN_WINDOW, GATE2_FORWARD_DAYS,
    )
    from signal_gate import classify_ticker, _get_gate1_threshold, _prepare_ml_features, _GATE2_FEATURES

    result = GateBacktestResult(ticker=ticker)

    featured = _download_data(ticker, period)
    if featured.empty:
        print(f"  {ticker}: 無資料，跳過")
        return None

    category = classify_ticker(ticker)
    gate1_threshold = _get_gate1_threshold(category)
    train_window = GATE2_TRAIN_WINDOW
    forward_days = GATE2_FORWARD_DAYS

    close = featured["Close"].squeeze()
    n = len(featured)

    # 下載 VIX 歷史（用於 Gate 3）
    vix_df = yf.download("^VIX", period=period, interval="1d", progress=False)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_df.columns = vix_df.columns.get_level_values(0)
    vix_close = vix_df["Close"].squeeze() if not vix_df.empty else pd.Series(dtype=float)

    # 準備 ML 特徵
    ml_df = _prepare_ml_features(featured)

    # Walk-forward: 從第 train_window 天起
    start_idx = train_window + 20  # 留暖機期
    if start_idx + forward_days >= n:
        print(f"  {ticker}: 資料不足 ({n} 天)，跳過")
        return None

    gate1_correct = 0
    gate1_total = 0
    gate2_correct = 0
    gate2_total = 0
    gate3_correct = 0
    gate3_total = 0
    combined_correct = 0
    combined_total = 0
    combined_returns = []
    baseline_correct = 0
    baseline_returns = []
    total_days = 0

    for i in range(start_idx, n - forward_days):
        total_days += 1

        today_close = float(close.iloc[i])
        future_close = float(close.iloc[i + forward_days])
        actual_up = future_close > today_close
        actual_return = (future_close - today_close) / today_close

        row = featured.iloc[i]

        # Baseline
        if actual_up:
            baseline_correct += 1
        baseline_returns.append(actual_return)

        # Gate 1: 技術面
        score = _compute_simple_score(row, today_close)
        g1_pass = _evaluate_gate1(row, gate1_threshold, GATE1_RSI_MAX, GATE1_ADX_MIN, score)
        if g1_pass:
            gate1_total += 1
            if actual_up:
                gate1_correct += 1

        # Gate 2: LightGBM
        g2_pass = True  # fallback
        g2_prob = 0.5
        if _HAS_LIGHTGBM and i >= train_window:
            try:
                X_train = ml_df[_GATE2_FEATURES].iloc[i - train_window:i].copy()
                X_train = X_train.ffill().bfill().fillna(0)

                # 訓練標籤
                future_ret = close.shift(-forward_days) / close - 1
                y_train_all = (future_ret > 0).astype(int)
                y_train = y_train_all.iloc[i - train_window:i].copy()

                # 確保 train data 有效
                valid = y_train.notna()
                X_t = X_train[valid]
                y_t = y_train[valid]

                if len(X_t) >= 30:
                    train_data = lgb.Dataset(X_t, label=y_t, free_raw_data=False)
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
                    X_pred = ml_df[_GATE2_FEATURES].iloc[[i]].ffill().bfill().fillna(0)
                    g2_prob = float(model.predict(X_pred)[0])
                    g2_pass = g2_prob > GATE2_PROBABILITY_THRESHOLD
            except Exception:
                g2_pass = True
                g2_prob = 0.5

        if g2_pass:
            gate2_total += 1
            if actual_up:
                gate2_correct += 1

        # Gate 3: VIX
        today_date = featured.index[i]
        vix_val = None
        if not vix_close.empty:
            # 找最近的 VIX 日期
            mask = vix_close.index <= today_date
            if mask.any():
                vix_val = float(vix_close[mask].iloc[-1])
        g3_pass = _evaluate_gate3_historical(vix_val)

        if g3_pass:
            gate3_total += 1
            if actual_up:
                gate3_correct += 1

        # 三驗證
        if g1_pass and g2_pass and g3_pass:
            combined_total += 1
            combined_returns.append(actual_return)
            if actual_up:
                combined_correct += 1

    # 統計
    result.total_days = total_days
    result.gate1_accuracy = gate1_correct / gate1_total * 100 if gate1_total > 0 else 0
    result.gate2_accuracy = gate2_correct / gate2_total * 100 if gate2_total > 0 else 0
    result.gate3_accuracy = gate3_correct / gate3_total * 100 if gate3_total > 0 else 0
    result.combined_accuracy = combined_correct / combined_total * 100 if combined_total > 0 else 0
    result.combined_signals = combined_total
    result.avg_return_on_signal = np.mean(combined_returns) * 100 if combined_returns else 0
    result.baseline_accuracy = baseline_correct / total_days * 100 if total_days > 0 else 0
    result.baseline_avg_return = np.mean(baseline_returns) * 100 if baseline_returns else 0

    return result


def calibrate_thresholds(tickers: list[str]) -> dict:
    """用歷史數據找出各類別的最佳 Gate 1 閾值。"""
    from signal_gate import classify_ticker

    category_scores = {}  # {category: [(threshold, accuracy), ...]}

    for ticker in tickers:
        category = classify_ticker(ticker)
        featured = _download_data(ticker, "2y")
        if featured.empty:
            continue

        close = featured["Close"].squeeze()
        n = len(featured)
        forward_days = 5

        if n < 150:
            continue

        # 測試不同閾值
        for threshold in range(45, 75, 5):
            correct = 0
            total = 0
            for i in range(60, n - forward_days):
                row = featured.iloc[i]
                today_close = float(close.iloc[i])
                future_close = float(close.iloc[i + forward_days])
                score = _compute_simple_score(row, today_close)

                if score >= threshold:
                    total += 1
                    if future_close > today_close:
                        correct += 1

            if total >= 10:
                acc = correct / total * 100
                category_scores.setdefault(category, []).append((threshold, acc, total))

    # 選每個 category 準確率最高且訊號數 >= 10 的閾值
    best = {}
    for cat, entries in category_scores.items():
        # 用準確率 * log(訊號數) 排序，平衡準確率和訊號數
        scored = [(t, a, c, a * np.log(max(c, 1))) for t, a, c in entries]
        scored.sort(key=lambda x: x[3], reverse=True)
        if scored:
            best[cat] = scored[0][0]
            print(f"  {cat}: best threshold = {scored[0][0]} "
                  f"(accuracy={scored[0][1]:.1f}%, signals={scored[0][2]})")

    return best


def run_full_validation(tickers: list[str] | None = None) -> None:
    """對所有監控股票跑回測，輸出報告到 stdout。"""
    if tickers is None:
        from watchlist import get_watchlist
        wl = get_watchlist()
        tickers = [item["ticker"] for item in wl]

    print("=" * 85)
    print("三驗證回測結果")
    print("=" * 85)
    header = (
        f"{'股票':<14} | {'Gate1':>6} | {'Gate2':>6} | {'Gate3':>6} | "
        f"{'三驗證':>6} | {'訊號數':>5} | {'平均報酬':>7} | {'隨機勝率':>7}"
    )
    print(header)
    print("-" * 85)

    all_results = []
    for ticker in tickers:
        print(f"  正在回測 {ticker}...", end="", flush=True)
        result = backtest_gates(ticker)
        if result is None:
            print(" 跳過")
            continue
        print(f" 完成 ({result.combined_signals} 訊號)")
        all_results.append(result)

    print("-" * 85)
    for r in all_results:
        line = (
            f"{r.ticker:<14} | {r.gate1_accuracy:>5.1f}% | {r.gate2_accuracy:>5.1f}% | "
            f"{r.gate3_accuracy:>5.1f}% | {r.combined_accuracy:>5.1f}% | "
            f"{r.combined_signals:>5} | {r.avg_return_on_signal:>+6.1f}% | "
            f"{r.baseline_accuracy:>5.1f}%"
        )
        print(line)

    if all_results:
        print("-" * 85)
        avg_combined = np.mean([r.combined_accuracy for r in all_results if r.combined_signals > 0])
        avg_baseline = np.mean([r.baseline_accuracy for r in all_results])
        total_signals = sum(r.combined_signals for r in all_results)
        avg_return = np.mean([r.avg_return_on_signal for r in all_results if r.combined_signals > 0])
        print(f"{'平均':<14} | {'':>6} | {'':>6} | {'':>6} | "
              f"{avg_combined:>5.1f}% | {total_signals:>5} | {avg_return:>+6.1f}% | "
              f"{avg_baseline:>5.1f}%")
        print("=" * 85)

        # 判定
        if avg_combined > 70:
            print("\n✅ 三驗證準確率 > 70%，建議部署")
        elif avg_combined > 60:
            print("\n⚠️ 三驗證準確率 60-70%，可考慮調整閾值後再測")
        else:
            print("\n❌ 三驗證準確率 < 60%，不建議部署，需調整策略")


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--calibrate" in args:
        print("=== 校準閾值 ===")
        from watchlist import get_watchlist
        wl = get_watchlist()
        tickers = [item["ticker"] for item in wl]
        calibrate_thresholds(tickers)
    elif args:
        run_full_validation(args)
    else:
        run_full_validation()
