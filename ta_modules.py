"""
花城 — 八大進階技術分析模組
=============================================
模組一：KLineSentiment       — K 線力道判斷
模組二：MACrossFilter        — 均線優化過濾
模組三：SupportResistance    — 支撐壓力轉換
模組四：ChartPatterns        — 價格形態辨識
模組五：RetestConfirmation   — 回測確認
模組六：SignalConfidence      — 訊號權重與機率分級
模組七：NoTradeZone          — 等待與避險過濾機制
模組八：ProfitTarget         — 獲利強弱勢排序

所有模組使用純 numpy/pandas 實作，零新增依賴。
"""

import logging
import numpy as np
import pandas as pd

_log = logging.getLogger("huacheng.ta_modules")


# ============================================================================
# 共用工具
# ============================================================================

def _find_pivots(highs: np.ndarray, lows: np.ndarray, left: int = 5, right: int = 5):
    """
    找出 Pivot High / Pivot Low。
    回傳 (pivot_highs, pivot_lows)，各為 list of (index, price)。
    """
    n = len(highs)
    pivot_highs = []
    pivot_lows = []

    for i in range(left, n - right):
        # Pivot High: highs[i] >= 左右各 left/right 根的 high
        is_ph = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if highs[j] > highs[i]:
                is_ph = False
                break
        if is_ph:
            pivot_highs.append((i, float(highs[i])))

        # Pivot Low: lows[i] <= 左右各 left/right 根的 low
        is_pl = True
        for j in range(i - left, i + right + 1):
            if j == i:
                continue
            if lows[j] < lows[i]:
                is_pl = False
                break
        if is_pl:
            pivot_lows.append((i, float(lows[i])))

    return pivot_highs, pivot_lows


def _safe_result(signal: str = "neutral", strength: int = 0, max_strength: int = 5,
                 detail: str = "", **extra):
    """建構標準化結果 dict。"""
    result = {
        "signal": signal,
        "strength": strength,
        "max": max_strength,
        "detail": detail,
    }
    result.update(extra)
    return result


# ============================================================================
# 模組一：K 線力道判斷 (KLineSentiment)
# ============================================================================

class KLineSentiment:
    """分析最近 K 線的力道形態（單根 / 雙根 / 三根）。"""

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        try:
            return KLineSentiment._analyze(df)
        except Exception as e:
            _log.debug(f"KLineSentiment error: {e}")
            return _safe_result(detail="K 線分析不可用")

    @staticmethod
    def _analyze(df: pd.DataFrame) -> dict:
        if len(df) < 5:
            return _safe_result(detail="資料不足")

        o = df["Open"].values.flatten().astype(float)
        h = df["High"].values.flatten().astype(float)
        lo = df["Low"].values.flatten().astype(float)
        c = df["Close"].values.flatten().astype(float)

        # 取最後 5 根
        o5, h5, lo5, c5 = o[-5:], h[-5:], lo[-5:], c[-5:]

        # --- 三根形態（優先偵測）---

        # 晨星 (Morning Star): 大陰 + 小實體 + 大陽(收過第一根中線)
        morning_star = KLineSentiment._check_morning_star(o5, h5, lo5, c5)
        if morning_star:
            return _safe_result("bullish", 4, detail="晨星 (Morning Star) — 底部反轉訊號",
                                bias="bullish")

        # 暮星 (Evening Star): 大陽 + 小實體 + 大陰
        evening_star = KLineSentiment._check_evening_star(o5, h5, lo5, c5)
        if evening_star:
            return _safe_result("bearish", 4, detail="暮星 (Evening Star) — 頂部反轉訊號",
                                bias="bearish")

        # --- 單根/雙根形態（取最後 2 根）---

        # Marubozu (光頭光腳)
        marubozu = KLineSentiment._check_marubozu(o5[-1], h5[-1], lo5[-1], c5[-1])
        if marubozu:
            return marubozu

        # Engulfing (吞噬)
        engulfing = KLineSentiment._check_engulfing(o5[-2], c5[-2], o5[-1], c5[-1])
        if engulfing:
            return engulfing

        # Hammer (錘子)
        hammer = KLineSentiment._check_hammer(o5[-1], h5[-1], lo5[-1], c5[-1])
        if hammer:
            return hammer

        # Shooting Star (射擊之星)
        shooting = KLineSentiment._check_shooting_star(o5[-1], h5[-1], lo5[-1], c5[-1])
        if shooting:
            return shooting

        # Doji (十字線)
        doji = KLineSentiment._check_doji(o5[-1], h5[-1], lo5[-1], c5[-1])
        if doji:
            return doji

        return _safe_result("neutral", 0, detail="無明顯 K 線形態", bias="neutral")

    @staticmethod
    def _check_morning_star(o, h, lo, c):
        """檢查最後 3~5 根中是否有晨星形態。"""
        for i in range(len(o) - 3, -1, -1):
            # 第一根：大陰線
            body1 = abs(c[i] - o[i])
            range1 = h[i] - lo[i]
            if range1 == 0 or body1 / range1 < 0.5 or c[i] >= o[i]:
                continue
            # 第二根：小實體
            body2 = abs(c[i+1] - o[i+1])
            range2 = h[i+1] - lo[i+1]
            if range2 > 0 and body2 / range2 > 0.3:
                continue
            # 第三根：大陽線，收盤過第一根中線
            body3 = abs(c[i+2] - o[i+2])
            range3 = h[i+2] - lo[i+2]
            mid1 = (o[i] + c[i]) / 2
            if range3 > 0 and body3 / range3 >= 0.5 and c[i+2] > o[i+2] and c[i+2] > mid1:
                return True
        return False

    @staticmethod
    def _check_evening_star(o, h, lo, c):
        """檢查最後 3~5 根中是否有暮星形態。"""
        for i in range(len(o) - 3, -1, -1):
            # 第一根：大陽線
            body1 = abs(c[i] - o[i])
            range1 = h[i] - lo[i]
            if range1 == 0 or body1 / range1 < 0.5 or c[i] <= o[i]:
                continue
            # 第二根：小實體
            body2 = abs(c[i+1] - o[i+1])
            range2 = h[i+1] - lo[i+1]
            if range2 > 0 and body2 / range2 > 0.3:
                continue
            # 第三根：大陰線，收盤低於第一根中線
            body3 = abs(c[i+2] - o[i+2])
            range3 = h[i+2] - lo[i+2]
            mid1 = (o[i] + c[i]) / 2
            if range3 > 0 and body3 / range3 >= 0.5 and c[i+2] < o[i+2] and c[i+2] < mid1:
                return True
        return False

    @staticmethod
    def _check_marubozu(o, h, lo, c):
        """光頭光腳大陽/陰線：影線 < 實體 5%，實體為全距 95%+。"""
        body = abs(c - o)
        full_range = h - lo
        if full_range == 0:
            return None
        if body / full_range < 0.95:
            return None
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - lo
        if body > 0 and (upper_shadow / body > 0.05 or lower_shadow / body > 0.05):
            return None
        if c > o:
            return _safe_result("bullish", 5, detail="光頭光腳大陽線 (Marubozu) — 極強多方",
                                bias="bullish")
        return _safe_result("bearish", 5, detail="光頭光腳大陰線 (Marubozu) — 極強空方",
                            bias="bearish")

    @staticmethod
    def _check_engulfing(o_prev, c_prev, o_curr, c_curr):
        """吞噬形態：第二根實體完全包覆前一根。"""
        body_prev_hi = max(o_prev, c_prev)
        body_prev_lo = min(o_prev, c_prev)
        body_curr_hi = max(o_curr, c_curr)
        body_curr_lo = min(o_curr, c_curr)

        if body_curr_hi > body_prev_hi and body_curr_lo < body_prev_lo:
            if c_curr > o_curr and c_prev < o_prev:
                return _safe_result("bullish", 3, detail="看漲吞噬 (Bullish Engulfing)",
                                    bias="bullish")
            elif c_curr < o_curr and c_prev > o_prev:
                return _safe_result("bearish", 3, detail="看跌吞噬 (Bearish Engulfing)",
                                    bias="bearish")
        return None

    @staticmethod
    def _check_hammer(o, h, lo, c):
        """錘子線：下影線 >= 2x 實體，上影線 < 0.3x 實體。"""
        body = abs(c - o)
        if body == 0:
            return None
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - lo

        if lower_shadow >= 2 * body and upper_shadow < 0.3 * body:
            return _safe_result("bullish", 2, detail="錘子線 (Hammer) — 底部支撐訊號",
                                bias="bullish")
        return None

    @staticmethod
    def _check_shooting_star(o, h, lo, c):
        """射擊之星：上影線 >= 2x 實體，下影線 < 0.3x 實體。"""
        body = abs(c - o)
        if body == 0:
            return None
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - lo

        if upper_shadow >= 2 * body and lower_shadow < 0.3 * body:
            return _safe_result("bearish", 2, detail="射擊之星 (Shooting Star) — 頂部壓力訊號",
                                bias="bearish")
        return None

    @staticmethod
    def _check_doji(o, h, lo, c):
        """十字線：實體 < 全距 10%。"""
        body = abs(c - o)
        full_range = h - lo
        if full_range > 0 and body / full_range < 0.10:
            return _safe_result("neutral", 1, detail="十字線 (Doji) — 多空拉鋸",
                                bias="neutral")
        return None


# ============================================================================
# 模組二：均線優化過濾 (MACrossFilter)
# ============================================================================

class MACrossFilter:
    """MA50/MA100/MA200 交叉訊號 + 趨勢過濾。"""

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        try:
            return MACrossFilter._analyze(df)
        except Exception as e:
            _log.debug(f"MACrossFilter error: {e}")
            return _safe_result(detail="均線分析不可用")

    @staticmethod
    def _analyze(df: pd.DataFrame) -> dict:
        close = df["Close"].values.flatten().astype(float)
        s_close = pd.Series(close)
        n = len(close)

        if n < 200:
            return _safe_result(detail=f"資料不足 ({n} 天，需 200 天)")

        ma50 = s_close.rolling(50).mean()
        ma100 = s_close.rolling(100).mean()
        ma200 = s_close.rolling(200).mean()

        current = close[-1]
        ma50_now = float(ma50.iloc[-1])
        ma100_now = float(ma100.iloc[-1])
        ma200_now = float(ma200.iloc[-1])

        # MA200 趨勢：近 20 日斜率
        ma200_20ago = float(ma200.iloc[-20]) if n >= 220 else float(ma200.iloc[-min(n, 201)])
        ma200_rising = ma200_now > ma200_20ago
        ma200_flat_or_rising = ma200_now >= ma200_20ago * 0.998  # 容許 0.2% 下降

        # MA50 交叉 MA200 判斷
        ma50_prev = float(ma50.iloc[-2])
        ma200_prev = float(ma200.iloc[-2])
        golden_cross = ma50_now > ma200_now and ma50_prev <= ma200_prev
        death_cross = ma50_now < ma200_now and ma50_prev >= ma200_prev

        # 近期交叉（20 日內）
        recent_golden = False
        recent_death = False
        lookback = min(20, n - 201)
        if lookback > 1:
            for i in range(-lookback, 0):
                m50_a = float(ma50.iloc[i - 1])
                m50_b = float(ma50.iloc[i])
                m200_a = float(ma200.iloc[i - 1])
                m200_b = float(ma200.iloc[i])
                if m50_b > m200_b and m50_a <= m200_a:
                    recent_golden = True
                if m50_b < m200_b and m50_a >= m200_a:
                    recent_death = True

        golden_cross = golden_cross or recent_golden
        death_cross = death_cross or recent_death

        # 多頭排列
        if current > ma50_now > ma100_now > ma200_now and ma200_rising:
            return _safe_result(
                "bullish", 5,
                detail="完美多頭排列 (Price>MA50>MA100>MA200)，MA200 上升中",
                bias="bullish",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )

        # 有效金叉
        if golden_cross and ma200_flat_or_rising:
            return _safe_result(
                "bullish", 5,
                detail="MA50/MA200 有效金叉 ✅ MA200 持平或上升",
                bias="bullish",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )

        # 弱金叉
        if golden_cross and not ma200_flat_or_rising:
            return _safe_result(
                "neutral", 2,
                detail="MA50/MA200 弱金叉 ⚠️ MA200 仍在下降，留意假訊號",
                bias="neutral",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )

        # 弱死叉（回調）
        if death_cross and ma200_rising:
            return _safe_result(
                "neutral", 3,
                detail="MA50/MA200 弱死叉（回調）— MA200 仍上升，視為正常回調",
                bias="neutral",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )

        # 有效死叉
        if death_cross and not ma200_rising:
            return _safe_result(
                "bearish", 0,
                detail="MA50/MA200 有效死叉 ❌ MA200 下降中",
                bias="bearish",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )

        # 空頭排列
        if current < ma50_now < ma100_now < ma200_now:
            return _safe_result(
                "bearish", 0,
                detail="空頭排列 (Price<MA50<MA100<MA200)",
                bias="bearish",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )

        # 中間狀態：價格在 MA200 上方偏多，下方偏空
        if current > ma200_now:
            strength = 4 if current > ma50_now else 3
            return _safe_result(
                "bullish", strength,
                detail=f"價格在 MA200 上方 (偏多)，MA50={'上穿' if ma50_now > ma200_now else '仍在 MA200 下'}",
                bias="bullish",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )
        else:
            strength = 1
            return _safe_result(
                "bearish", strength,
                detail="價格在 MA200 下方 (偏空)",
                bias="bearish",
                ma50=ma50_now, ma100=ma100_now, ma200=ma200_now,
            )


# ============================================================================
# 模組三：支撐壓力轉換 (SupportResistance)
# ============================================================================

class SupportResistance:
    """偵測水平支撐壓力線、極性翻轉、回測訊號。"""

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        try:
            return SupportResistance._analyze(df)
        except Exception as e:
            _log.debug(f"SupportResistance error: {e}")
            return _safe_result(detail="支撐壓力分析不可用")

    @staticmethod
    def _analyze(df: pd.DataFrame) -> dict:
        highs = df["High"].values.flatten().astype(float)
        lows = df["Low"].values.flatten().astype(float)
        close = df["Close"].values.flatten().astype(float)
        current = close[-1]
        n = len(close)

        if n < 30:
            return _safe_result(detail="資料不足")

        pivot_highs, pivot_lows = _find_pivots(highs, lows, left=5, right=5)

        # 合併所有 pivot 價位
        all_pivots = [(idx, price, "high") for idx, price in pivot_highs] + \
                     [(idx, price, "low") for idx, price in pivot_lows]

        if not all_pivots:
            return _safe_result(detail="未找到有效 Pivot 點")

        # 水平線聚類：價差 <= 1.5% 歸同一組
        levels = SupportResistance._cluster_levels(all_pivots, current, threshold_pct=1.5)

        # 至少 2 次觸碰才是有效水平線
        valid_levels = [lv for lv in levels if lv["touches"] >= 2]

        if not valid_levels:
            return _safe_result(detail="無有效支撐壓力線 (需至少 2 次觸碰)",
                                levels=[])

        # 極性判斷 + 翻轉偵測
        for lv in valid_levels:
            lv["type"] = "support" if current > lv["price"] else "resistance"
            lv["flipped"] = SupportResistance._check_flip(close, lv, n)

        # 找最近的支撐和壓力
        supports = [lv for lv in valid_levels if lv["type"] == "support"]
        resistances = [lv for lv in valid_levels if lv["type"] == "resistance"]

        nearest_support = max(supports, key=lambda x: x["price"]) if supports else None
        nearest_resistance = min(resistances, key=lambda x: x["price"]) if resistances else None

        # 訊號判斷
        signal = "neutral"
        strength = 2
        detail_parts = []

        # 翻轉線計數
        flipped_supports = [lv for lv in supports if lv["flipped"]]
        if flipped_supports:
            signal = "bullish"
            strength = 4
            prices_str = ", ".join(f"${lv['price']:,.2f}" for lv in flipped_supports[:3])
            detail_parts.append(f"壓力變支撐 (R→S flip): {prices_str}")

        # 價格接近支撐
        if nearest_support:
            dist_pct = abs(current - nearest_support["price"]) / nearest_support["price"] * 100
            if dist_pct < 2.0:
                signal = "bullish"
                strength = max(strength, 3)
                detail_parts.append(f"接近支撐 ${nearest_support['price']:,.2f} (距離 {dist_pct:.1f}%)")

        # 價格接近壓力
        if nearest_resistance:
            dist_pct = abs(nearest_resistance["price"] - current) / current * 100
            if dist_pct < 2.0:
                detail_parts.append(f"接近壓力 ${nearest_resistance['price']:,.2f} (距離 {dist_pct:.1f}%)")
                if signal != "bullish":
                    signal = "bearish"
                    strength = 1

        if not detail_parts:
            sup_str = f"${nearest_support['price']:,.2f}" if nearest_support else "N/A"
            res_str = f"${nearest_resistance['price']:,.2f}" if nearest_resistance else "N/A"
            detail_parts.append(f"最近支撐 {sup_str} | 壓力 {res_str}")

        return _safe_result(
            signal, strength,
            detail=" | ".join(detail_parts),
            levels=valid_levels,
            nearest_support=nearest_support,
            nearest_resistance=nearest_resistance,
        )

    @staticmethod
    def _cluster_levels(all_pivots, current_price, threshold_pct=1.5):
        """將 pivot 價位聚類成水平線。"""
        if not all_pivots:
            return []

        # 按價格排序
        sorted_pivots = sorted(all_pivots, key=lambda x: x[1])
        clusters = []
        used = set()

        for i, (idx_i, price_i, type_i) in enumerate(sorted_pivots):
            if i in used:
                continue
            cluster_prices = [price_i]
            cluster_indices = [idx_i]
            used.add(i)

            for j in range(i + 1, len(sorted_pivots)):
                if j in used:
                    continue
                idx_j, price_j, type_j = sorted_pivots[j]
                if abs(price_j - price_i) / price_i * 100 <= threshold_pct:
                    cluster_prices.append(price_j)
                    cluster_indices.append(idx_j)
                    used.add(j)

            avg_price = float(np.mean(cluster_prices))
            clusters.append({
                "price": avg_price,
                "touches": len(cluster_prices),
                "indices": cluster_indices,
                "type": "unknown",
                "flipped": False,
            })

        return clusters

    @staticmethod
    def _check_flip(close, level, n):
        """檢查近 20 日是否有壓力變支撐的翻轉。"""
        price = level["price"]
        lookback = min(20, n)
        recent_close = close[-lookback:]

        # 曾在線下方
        was_below = any(c < price * 0.985 for c in recent_close[:-5])
        # 現在在線上方
        is_above = close[-1] > price

        return was_below and is_above


# ============================================================================
# 模組四：價格形態辨識 (ChartPatterns)
# ============================================================================

class ChartPatterns:
    """辨識雙底/雙頂/頭肩/楔形/三角/旗形等技術形態。"""

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        try:
            return ChartPatterns._analyze(df)
        except Exception as e:
            _log.debug(f"ChartPatterns error: {e}")
            return _safe_result(detail="形態辨識不可用")

    @staticmethod
    def _analyze(df: pd.DataFrame) -> dict:
        highs = df["High"].values.flatten().astype(float)
        lows = df["Low"].values.flatten().astype(float)
        close = df["Close"].values.flatten().astype(float)
        volume = df["Volume"].values.flatten().astype(float)
        n = len(close)

        if n < 30:
            return _safe_result(detail="資料不足")

        pivot_highs, pivot_lows = _find_pivots(highs, lows, left=5, right=5)

        # 依優先權偵測（複雜形態優先）
        patterns_found = []

        # Head & Shoulders
        hs = ChartPatterns._check_head_shoulders(pivot_highs, pivot_lows, close, n)
        if hs:
            patterns_found.append(hs)

        # Inverse Head & Shoulders
        ihs = ChartPatterns._check_inverse_head_shoulders(pivot_highs, pivot_lows, close, n)
        if ihs:
            patterns_found.append(ihs)

        # Double Bottom
        db = ChartPatterns._check_double_bottom(pivot_highs, pivot_lows, close, n)
        if db:
            patterns_found.append(db)

        # Double Top
        dt = ChartPatterns._check_double_top(pivot_highs, pivot_lows, close, n)
        if dt:
            patterns_found.append(dt)

        # Wedge patterns
        wedge = ChartPatterns._check_wedge(pivot_highs, pivot_lows, close, volume, n)
        if wedge:
            patterns_found.append(wedge)

        # Triangle patterns
        tri = ChartPatterns._check_triangle(pivot_highs, pivot_lows, close, volume, n)
        if tri:
            patterns_found.append(tri)

        # Flag patterns
        flag = ChartPatterns._check_flag(close, volume, n)
        if flag:
            patterns_found.append(flag)

        # Diamond Top
        diamond = ChartPatterns._check_diamond_top(pivot_highs, pivot_lows, close, n)
        if diamond:
            patterns_found.append(diamond)

        if not patterns_found:
            return _safe_result(detail="未偵測到明顯形態")

        # 回傳所有偵測到的形態（最高力道在首位）
        patterns_found.sort(key=lambda x: x.get("strength", 0), reverse=True)
        best = patterns_found[0]
        best["all_patterns"] = patterns_found
        return best

    @staticmethod
    def _check_double_bottom(pivot_highs, pivot_lows, close, n):
        """雙底 (W)：兩個 Pivot Low 價差 <= 2%，中間有 Pivot High。"""
        if len(pivot_lows) < 2:
            return None

        # 取最近的 pivot lows
        for i in range(len(pivot_lows) - 1, 0, -1):
            idx2, price2 = pivot_lows[i]
            for j in range(i - 1, max(i - 4, -1), -1):
                idx1, price1 = pivot_lows[j]
                if abs(price2 - price1) / price1 * 100 > 2.0:
                    continue
                # 中間需有 Pivot High (頸線)
                neckline_pivots = [
                    (idx_h, ph) for idx_h, ph in pivot_highs
                    if idx1 < idx_h < idx2
                ]
                if not neckline_pivots:
                    continue
                neckline = max(neckline_pivots, key=lambda x: x[1])
                # 確認現價在頸線附近或突破
                if close[-1] > neckline[1] * 0.98:
                    breakout = close[-1] > neckline[1]
                    strength = 5 if breakout else 3
                    return _safe_result(
                        "bullish", strength,
                        detail=f"雙底 (W) — {'已突破頸線' if breakout else '接近頸線'}，底部 ${price1:,.2f}",
                        bias="bullish",
                        pattern="double_bottom",
                        neckline=neckline[1],
                    )
        return None

    @staticmethod
    def _check_double_top(pivot_highs, pivot_lows, close, n):
        """雙頂 (M)：兩個 Pivot High 價差 <= 2%，中間有 Pivot Low。"""
        if len(pivot_highs) < 2:
            return None

        for i in range(len(pivot_highs) - 1, 0, -1):
            idx2, price2 = pivot_highs[i]
            for j in range(i - 1, max(i - 4, -1), -1):
                idx1, price1 = pivot_highs[j]
                if abs(price2 - price1) / price1 * 100 > 2.0:
                    continue
                neckline_pivots = [
                    (idx_l, pl) for idx_l, pl in pivot_lows
                    if idx1 < idx_l < idx2
                ]
                if not neckline_pivots:
                    continue
                neckline = min(neckline_pivots, key=lambda x: x[1])
                if close[-1] < neckline[1] * 1.02:
                    breakdown = close[-1] < neckline[1]
                    strength = 5 if breakdown else 3
                    return _safe_result(
                        "bearish", strength,
                        detail=f"雙頂 (M) — {'已跌破頸線' if breakdown else '接近頸線'}，頂部 ${price1:,.2f}",
                        bias="bearish",
                        pattern="double_top",
                        neckline=neckline[1],
                    )
        return None

    @staticmethod
    def _check_head_shoulders(pivot_highs, pivot_lows, close, n):
        """頭肩頂：3 個 Pivot High，中間最高，兩肩相近。"""
        if len(pivot_highs) < 3:
            return None

        for i in range(len(pivot_highs) - 1, 1, -1):
            for j in range(i - 1, 0, -1):
                for k in range(j - 1, max(j - 3, -1), -1):
                    idx_r, pr = pivot_highs[i]
                    idx_h, ph = pivot_highs[j]
                    idx_l, pl = pivot_highs[k]

                    # 中間最高
                    if ph <= pr or ph <= pl:
                        continue
                    # 兩肩價差 <= 3%
                    if abs(pr - pl) / pl * 100 > 3.0:
                        continue
                    # 頭比肩高至少 1%
                    if (ph - max(pr, pl)) / max(pr, pl) * 100 < 1.0:
                        continue

                    # 頸線：連接肩部之間的低點
                    neck_lows = [
                        (idx_nl, pnl) for idx_nl, pnl in pivot_lows
                        if idx_l < idx_nl < idx_r
                    ]
                    if len(neck_lows) < 1:
                        continue
                    neckline = np.mean([pnl for _, pnl in neck_lows])

                    if close[-1] < neckline * 1.02:
                        breakdown = close[-1] < neckline
                        return _safe_result(
                            "bearish", 5 if breakdown else 3,
                            detail=f"頭肩頂 — {'已跌破頸線' if breakdown else '接近頸線'}",
                            bias="bearish",
                            pattern="head_shoulders_top",
                            neckline=float(neckline),
                        )
        return None

    @staticmethod
    def _check_inverse_head_shoulders(pivot_highs, pivot_lows, close, n):
        """頭肩底：3 個 Pivot Low，中間最低，兩肩相近。"""
        if len(pivot_lows) < 3:
            return None

        for i in range(len(pivot_lows) - 1, 1, -1):
            for j in range(i - 1, 0, -1):
                for k in range(j - 1, max(j - 3, -1), -1):
                    idx_r, pr = pivot_lows[i]
                    idx_h, ph = pivot_lows[j]
                    idx_l, pl = pivot_lows[k]

                    # 中間最低
                    if ph >= pr or ph >= pl:
                        continue
                    # 兩肩價差 <= 3%
                    if abs(pr - pl) / pl * 100 > 3.0:
                        continue
                    # 頭比肩低至少 1%
                    if (min(pr, pl) - ph) / min(pr, pl) * 100 < 1.0:
                        continue

                    neck_highs = [
                        (idx_nh, pnh) for idx_nh, pnh in pivot_highs
                        if idx_l < idx_nh < idx_r
                    ]
                    if len(neck_highs) < 1:
                        continue
                    neckline = np.mean([pnh for _, pnh in neck_highs])

                    if close[-1] > neckline * 0.98:
                        breakout = close[-1] > neckline
                        return _safe_result(
                            "bullish", 5 if breakout else 3,
                            detail=f"頭肩底 — {'已突破頸線' if breakout else '接近頸線'}",
                            bias="bullish",
                            pattern="head_shoulders_bottom",
                            neckline=float(neckline),
                        )
        return None

    @staticmethod
    def _check_wedge(pivot_highs, pivot_lows, close, volume, n):
        """楔形：高點連線和低點連線收斂。"""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None

        # 取最近的 pivot points
        recent_ph = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs[-2:]
        recent_pl = pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows[-2:]

        if len(recent_ph) < 2 or len(recent_pl) < 2:
            return None

        # 高點趨勢
        ph_indices = [p[0] for p in recent_ph]
        ph_prices = [p[1] for p in recent_ph]
        high_slope = (ph_prices[-1] - ph_prices[0]) / max(ph_indices[-1] - ph_indices[0], 1)

        # 低點趨勢
        pl_indices = [p[0] for p in recent_pl]
        pl_prices = [p[1] for p in recent_pl]
        low_slope = (pl_prices[-1] - pl_prices[0]) / max(pl_indices[-1] - pl_indices[0], 1)

        # 收斂判斷
        converging = abs(high_slope - low_slope) < abs(high_slope) + abs(low_slope)

        if not converging:
            return None

        # 量能過濾
        vol_confirmed = ChartPatterns._volume_filter(volume, n)

        # 下降楔形（看漲）
        if high_slope < 0 and low_slope < 0 and abs(low_slope) > abs(high_slope):
            strength = 4 if vol_confirmed else 2
            return _safe_result(
                "bullish", strength,
                detail=f"下降楔形 (Falling Wedge) — {'量價確認' if vol_confirmed else '待確認'}",
                bias="bullish",
                pattern="falling_wedge",
            )

        # 上升楔形（看跌）
        if high_slope > 0 and low_slope > 0 and high_slope > low_slope:
            strength = 4 if vol_confirmed else 2
            return _safe_result(
                "bearish", strength,
                detail=f"上升楔形 (Rising Wedge) — {'量價確認' if vol_confirmed else '待確認'}",
                bias="bearish",
                pattern="rising_wedge",
            )

        return None

    @staticmethod
    def _check_triangle(pivot_highs, pivot_lows, close, volume, n):
        """三角形態：上升三角 / 下降三角。"""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None

        recent_ph = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs[-2:]
        recent_pl = pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows[-2:]

        ph_prices = [p[1] for p in recent_ph]
        pl_prices = [p[1] for p in recent_pl]

        # 高點水平判斷（價差 <= 1.5%）
        high_flat = (max(ph_prices) - min(ph_prices)) / min(ph_prices) * 100 <= 1.5
        # 低點水平判斷
        low_flat = (max(pl_prices) - min(pl_prices)) / min(pl_prices) * 100 <= 1.5

        low_rising = pl_prices[-1] > pl_prices[0]
        high_falling = ph_prices[-1] < ph_prices[0]

        vol_confirmed = ChartPatterns._volume_filter(volume, n)

        # 上升三角（看漲）
        if high_flat and low_rising:
            strength = 4 if vol_confirmed else 2
            return _safe_result(
                "bullish", strength,
                detail=f"上升三角 (Ascending Triangle) — {'量價確認' if vol_confirmed else '待確認'}",
                bias="bullish",
                pattern="ascending_triangle",
            )

        # 下降三角（看跌）
        if low_flat and high_falling:
            strength = 4 if vol_confirmed else 2
            return _safe_result(
                "bearish", strength,
                detail=f"下降三角 (Descending Triangle) — {'量價確認' if vol_confirmed else '待確認'}",
                bias="bearish",
                pattern="descending_triangle",
            )

        return None

    @staticmethod
    def _check_flag(close, volume, n):
        """旗形：急漲/急跌後微幅反向通道。"""
        if n < 20:
            return None

        # 檢查前段急漲/急跌（前 5~15 根）
        pole_start = max(0, n - 20)
        pole_end = n - 5
        flag_section = close[pole_end:]

        if pole_end <= pole_start:
            return None

        pole_change_pct = (close[pole_end] - close[pole_start]) / close[pole_start] * 100
        flag_change_pct = (flag_section[-1] - flag_section[0]) / flag_section[0] * 100

        # 上升旗形：急漲 > 5% 後微幅下降 (-3% ~ 0%)
        if pole_change_pct > 5 and -3 < flag_change_pct < 0:
            return _safe_result(
                "bullish", 3,
                detail=f"上升旗形 (Bull Flag) — 急漲 {pole_change_pct:.1f}% 後整理",
                bias="bullish",
                pattern="bull_flag",
            )

        # 下降旗形：急跌 > 5% 後微幅上升 (0% ~ 3%)
        if pole_change_pct < -5 and 0 < flag_change_pct < 3:
            return _safe_result(
                "bearish", 3,
                detail=f"下降旗形 (Bear Flag) — 急跌 {pole_change_pct:.1f}% 後反彈",
                bias="bearish",
                pattern="bear_flag",
            )

        return None

    @staticmethod
    def _check_diamond_top(pivot_highs, pivot_lows, close, n):
        """菱形頂 (Diamond Top)：先擴散後收斂的頂部反轉。"""
        if len(pivot_highs) < 4 or len(pivot_lows) < 4:
            return None

        # 取最近 4 個 high pivots 和 4 個 low pivots
        recent_ph = pivot_highs[-4:]
        recent_pl = pivot_lows[-4:]

        ph_prices = [p[1] for p in recent_ph]
        pl_prices = [p[1] for p in recent_pl]

        # 前半擴散：高點漸高且低點漸低
        if len(ph_prices) >= 3 and len(pl_prices) >= 3:
            mid = len(ph_prices) // 2
            # 前半
            front_h_rising = ph_prices[mid - 1] > ph_prices[0]
            front_l_falling = pl_prices[mid - 1] < pl_prices[0]
            # 後半收斂
            back_h_falling = ph_prices[-1] < ph_prices[mid]
            back_l_rising = pl_prices[-1] > pl_prices[mid]

            if front_h_rising and front_l_falling and back_h_falling and back_l_rising:
                # 價格跌破菱形下緣
                lower_bound = min(pl_prices[-2:])
                if close[-1] < lower_bound:
                    return _safe_result(
                        "bearish", 3,
                        detail=f"菱形頂 (Diamond Top) — 已跌破下緣 ${lower_bound:,.2f}",
                        bias="bearish",
                        pattern="diamond_top",
                    )
                elif close[-1] < lower_bound * 1.02:
                    return _safe_result(
                        "bearish", 2,
                        detail=f"菱形頂 (Diamond Top) — 接近下緣 ${lower_bound:,.2f}",
                        bias="bearish",
                        pattern="diamond_top",
                    )
        return None

    @staticmethod
    def _volume_filter(volume, n):
        """量價過濾器：收斂期量縮 + 突破放量。"""
        if n < 10:
            return False
        # 收斂期均量（前 10 根）
        convergence_vol = np.mean(volume[-10:-1])
        # 最後一根成交量
        last_vol = volume[-1]
        # 突破放量：>= 收斂期均量 1.3 倍
        return last_vol >= convergence_vol * 1.3


# ============================================================================
# 模組五：回測確認 (RetestConfirmation)
# ============================================================================

class RetestConfirmation:
    """整合模組 1+3+4，確認回測是否成功。"""

    @staticmethod
    def analyze(df: pd.DataFrame, kline_result: dict,
                sr_result: dict, pattern_result: dict) -> dict:
        try:
            return RetestConfirmation._analyze(df, kline_result, sr_result, pattern_result)
        except Exception as e:
            _log.debug(f"RetestConfirmation error: {e}")
            return _safe_result(detail="回測確認不可用")

    @staticmethod
    def _analyze(df: pd.DataFrame, kline_result: dict,
                 sr_result: dict, pattern_result: dict) -> dict:
        close = df["Close"].values.flatten().astype(float)
        n = len(close)
        current = close[-1]

        if n < 10:
            return _safe_result(detail="資料不足")

        # 收集需要觀察回測的價位
        retest_levels = []

        # 從 S/R 翻轉線
        levels = sr_result.get("levels", [])
        for lv in levels:
            if lv.get("flipped") and lv.get("type") == "support":
                retest_levels.append(("S/R翻轉", lv["price"]))

        # 從形態突破頸線
        neckline = pattern_result.get("neckline")
        if neckline and pattern_result.get("signal") == "bullish":
            retest_levels.append(("形態頸線", neckline))

        if not retest_levels:
            return _safe_result(detail="無回測觀察目標")

        # K 線力道
        kline_strength = kline_result.get("strength", 0)
        kline_bullish = kline_result.get("bias") == "bullish"

        # 檢查回測狀態
        best_status = None
        best_strength = 0

        for label, level_price in retest_levels:
            status = RetestConfirmation._check_retest(
                close, current, level_price, kline_strength, kline_bullish
            )
            if status["strength"] > best_strength:
                best_strength = status["strength"]
                best_status = status
                best_status["level_label"] = label
                best_status["level_price"] = level_price

        if not best_status:
            return _safe_result(detail="無回測觀察目標")

        return _safe_result(
            best_status.get("signal", "neutral"),
            best_status["strength"],
            detail=best_status["detail"],
        )

    @staticmethod
    def _check_retest(close, current, level_price, kline_strength, kline_bullish):
        """檢查單一價位的回測狀態。"""
        n = len(close)
        lookback = min(10, n)
        recent = close[-lookback:]

        # 是否有拉回到該價位 +/- 1.5%
        threshold = level_price * 0.015
        touched = any(abs(c - level_price) <= threshold for c in recent)

        # 是否守住（收盤仍在線上方）
        held = current > level_price

        # 正在接近
        approaching = abs(current - level_price) / level_price * 100 < 3.0 and current > level_price

        if touched and held and kline_bullish and kline_strength >= 2:
            return {
                "signal": "bullish",
                "strength": 5,
                "detail": f"回測確認 ✅ 拉回 ${level_price:,.2f} 守住 + 反轉 K 線",
            }
        elif touched and held:
            return {
                "signal": "bullish",
                "strength": 3,
                "detail": f"回測持守 — 拉回 ${level_price:,.2f} 守住，尚無反轉訊號",
            }
        elif approaching:
            return {
                "signal": "neutral",
                "strength": 2,
                "detail": f"回測進行中 — 正在接近支撐 ${level_price:,.2f}",
            }
        elif touched and not held:
            return {
                "signal": "bearish",
                "strength": 0,
                "detail": f"回測失敗 ❌ 跌破支撐 ${level_price:,.2f}",
            }
        else:
            return {
                "signal": "neutral",
                "strength": 1,
                "detail": f"觀察中 — 回測目標 ${level_price:,.2f}",
            }


# ============================================================================
# 模組六：訊號權重與機率分級 (SignalConfidence)
# ============================================================================

# 形態→等級對照表
_PATTERN_GRADES = {
    # --- S 級 (100%) 強烈反轉 ---
    "double_top":            {"grade": "S", "confidence": 100, "action": "strong_sell",
                              "label": "🔥🔥 S級 強烈賣出 (M頂跌破頸線)"},
    "head_shoulders_top":    {"grade": "S", "confidence": 100, "action": "strong_sell",
                              "label": "🔥🔥 S級 強烈賣出 (頭肩頂)"},
    "falling_wedge":         {"grade": "S", "confidence": 100, "action": "strong_buy",
                              "label": "🔥🔥 S級 強烈買入 (下降楔形突破)"},
    "head_shoulders_bottom": {"grade": "S", "confidence": 100, "action": "strong_buy",
                              "label": "🔥🔥 S級 強烈買入 (頭肩底突破)"},
    "double_bottom":         {"grade": "S", "confidence": 100, "action": "strong_buy",
                              "label": "🔥🔥 S級 強烈買入 (W底突破)"},
    # --- A 級 (80%) 趨勢中繼 ---
    "bear_flag":             {"grade": "A", "confidence": 80, "action": "sell",
                              "label": "🔥 A級 急賣 (下跌旗形)"},
    "bull_flag":             {"grade": "A", "confidence": 80, "action": "buy",
                              "label": "🔥 A級 急買 (上升旗形)"},
    "ascending_triangle":    {"grade": "A", "confidence": 80, "action": "buy",
                              "label": "🔥 A級 買入 (上升三角突破)"},
    "descending_triangle":   {"grade": "A", "confidence": 80, "action": "sell",
                              "label": "🔥 A級 賣出 (下降三角跌破)"},
    # --- B 級 (65%) 緩慢表態 / 試單 ---
    "diamond_top":           {"grade": "B", "confidence": 65, "action": "sell",
                              "label": "⚠️ B級 緩慢下跌 (菱形頂)"},
    "rising_wedge":          {"grade": "B", "confidence": 65, "action": "sell",
                              "label": "⚠️ B級 賣出 (上升楔形)"},
}


class SignalConfidence:
    """
    將模組四辨識出的形態分級為 S / A / B，並依此建議資金部位。
    S 級 (100%): 重倉 — M頂/頭肩頂/下降楔形/頭肩底/W底
    A 級 (80%):  標準倉 — 旗形/三角形
    B 級 (65%):  輕倉試單 — 菱形頂/上升楔形
    """

    @staticmethod
    def analyze(pattern_result: dict, retest_result: dict) -> dict:
        try:
            return SignalConfidence._analyze(pattern_result, retest_result)
        except Exception as e:
            _log.debug(f"SignalConfidence error: {e}")
            return _safe_result(detail="訊號分級不可用",
                                grade="none", confidence_pct=0,
                                position_pct=0, action="hold")

    @staticmethod
    def _analyze(pattern_result: dict, retest_result: dict) -> dict:
        pattern_name = pattern_result.get("pattern")
        if not pattern_name:
            return _safe_result(
                detail="無形態訊號，維持觀望",
                grade="none", confidence_pct=0,
                position_pct=0, action="hold",
            )

        grade_info = _PATTERN_GRADES.get(pattern_name)
        if not grade_info:
            return _safe_result(
                detail=f"形態 {pattern_name} 未分級",
                grade="none", confidence_pct=0,
                position_pct=0, action="hold",
            )

        grade = grade_info["grade"]
        confidence = grade_info["confidence"]
        action = grade_info["action"]
        label = grade_info["label"]

        # S 級買入形態：若回測確認成功，信心提升至最高
        retest_strength = retest_result.get("strength", 0)
        if grade == "S" and action in ("strong_buy",) and retest_strength >= 4:
            label += " + 回測確認 ✅"
            confidence = 100

        # 根據信心度決定部位比例
        position_pct = confidence  # 100% / 80% / 65%

        # 力道映射：S=5, A=4, B=3
        strength_map = {"S": 5, "A": 4, "B": 3}
        strength = strength_map.get(grade, 0)

        signal = "bullish" if action in ("strong_buy", "buy") else "bearish"

        return _safe_result(
            signal, strength,
            detail=f"{label} — 信心 {confidence}%，建議部位 {position_pct}%",
            grade=grade,
            confidence_pct=confidence,
            position_pct=position_pct,
            action=action,
        )


# ============================================================================
# 模組七：等待與避險過濾機制 (NoTradeZone)
# ============================================================================

class NoTradeZone:
    """
    偵測盤整/無方向狀態，強制程式空手等待。
    禁止交易形態：箱型盤整、對稱三角、擴散三角、上升通道內部。
    解除條件：實體 K 線 + 大成交量突破邊界。
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> dict:
        try:
            return NoTradeZone._analyze(df)
        except Exception as e:
            _log.debug(f"NoTradeZone error: {e}")
            return _safe_result(detail="避險過濾不可用",
                                trade_allowed=True, zone_type=None)

    @staticmethod
    def _analyze(df: pd.DataFrame) -> dict:
        highs = df["High"].values.flatten().astype(float)
        lows = df["Low"].values.flatten().astype(float)
        close = df["Close"].values.flatten().astype(float)
        volume = df["Volume"].values.flatten().astype(float)
        n = len(close)

        if n < 30:
            return _safe_result(
                detail="資料不足，允許交易", trade_allowed=True, zone_type=None,
            )

        pivot_highs, pivot_lows = _find_pivots(highs, lows, left=5, right=5)

        # 依序偵測禁止交易形態
        zones = []

        rect = NoTradeZone._check_rectangle(pivot_highs, pivot_lows, close, volume, n)
        if rect:
            zones.append(rect)

        sym_tri = NoTradeZone._check_symmetrical_triangle(pivot_highs, pivot_lows, close, volume, n)
        if sym_tri:
            zones.append(sym_tri)

        expanding = NoTradeZone._check_expanding_triangle(pivot_highs, pivot_lows, close, volume, n)
        if expanding:
            zones.append(expanding)

        channel = NoTradeZone._check_ascending_channel(pivot_highs, pivot_lows, close, volume, n)
        if channel:
            zones.append(channel)

        if not zones:
            return _safe_result(
                "neutral", 5,
                detail="未偵測到盤整區間，允許交易",
                trade_allowed=True, zone_type=None,
            )

        # 取最危險的（最低 strength）
        worst = min(zones, key=lambda x: x.get("strength", 5))
        return worst

    @staticmethod
    def _check_rectangle(pivot_highs, pivot_lows, close, volume, n):
        """箱型盤整：高點水平 + 低點水平。"""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None

        recent_ph = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs[-2:]
        recent_pl = pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows[-2:]

        ph_prices = [p[1] for p in recent_ph]
        pl_prices = [p[1] for p in recent_pl]

        high_flat = (max(ph_prices) - min(ph_prices)) / min(ph_prices) * 100 <= 2.0
        low_flat = (max(pl_prices) - min(pl_prices)) / min(pl_prices) * 100 <= 2.0

        if not (high_flat and low_flat):
            return None

        upper = np.mean(ph_prices)
        lower = np.mean(pl_prices)

        # 檢查是否已突破
        breakout = NoTradeZone._check_volume_breakout(close, volume, upper, lower, n)
        if breakout:
            signal = "bullish" if breakout == "up" else "bearish"
            return _safe_result(
                signal, 4,
                detail=f"箱型盤整已突破{'上緣' if breakout == 'up' else '下緣'} — 可交易",
                trade_allowed=True,
                zone_type="rectangle",
                breakout_direction=breakout,
                upper=float(upper), lower=float(lower),
            )

        return _safe_result(
            "neutral", 0,
            detail=f"⚠️ 箱型盤整 ${lower:,.2f}~${upper:,.2f} — 危險別碰，等待突破",
            trade_allowed=False,
            zone_type="rectangle",
            upper=float(upper), lower=float(lower),
        )

    @staticmethod
    def _check_symmetrical_triangle(pivot_highs, pivot_lows, close, volume, n):
        """對稱三角：高點漸低 + 低點漸高（收斂）。"""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None

        recent_ph = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs[-2:]
        recent_pl = pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows[-2:]

        ph_prices = [p[1] for p in recent_ph]
        pl_prices = [p[1] for p in recent_pl]

        high_falling = ph_prices[-1] < ph_prices[0]
        low_rising = pl_prices[-1] > pl_prices[0]

        if not (high_falling and low_rising):
            return None

        upper = ph_prices[-1]
        lower = pl_prices[-1]

        breakout = NoTradeZone._check_volume_breakout(close, volume, upper, lower, n)
        if breakout:
            signal = "bullish" if breakout == "up" else "bearish"
            return _safe_result(
                signal, 4,
                detail=f"對稱三角已突破{'上緣' if breakout == 'up' else '下緣'} — 可交易",
                trade_allowed=True,
                zone_type="symmetrical_triangle",
                breakout_direction=breakout,
            )

        return _safe_result(
            "neutral", 0,
            detail="⚠️ 對稱三角收斂中 — 方向未明，等待突破",
            trade_allowed=False,
            zone_type="symmetrical_triangle",
        )

    @staticmethod
    def _check_expanding_triangle(pivot_highs, pivot_lows, close, volume, n):
        """擴散三角：高點漸高 + 低點漸低（發散）。"""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None

        recent_ph = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs[-2:]
        recent_pl = pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows[-2:]

        ph_prices = [p[1] for p in recent_ph]
        pl_prices = [p[1] for p in recent_pl]

        high_rising = ph_prices[-1] > ph_prices[0]
        low_falling = pl_prices[-1] < pl_prices[0]

        if not (high_rising and low_falling):
            return None

        return _safe_result(
            "neutral", 0,
            detail="⚠️ 擴散三角 — 市場情緒失控，高風險勿碰",
            trade_allowed=False,
            zone_type="expanding_triangle",
        )

    @staticmethod
    def _check_ascending_channel(pivot_highs, pivot_lows, close, volume, n):
        """上升通道內部：高低點平行上升，價格在中間。"""
        if len(pivot_highs) < 2 or len(pivot_lows) < 2:
            return None

        recent_ph = pivot_highs[-3:] if len(pivot_highs) >= 3 else pivot_highs[-2:]
        recent_pl = pivot_lows[-3:] if len(pivot_lows) >= 3 else pivot_lows[-2:]

        ph_prices = [p[1] for p in recent_ph]
        ph_indices = [p[0] for p in recent_ph]
        pl_prices = [p[1] for p in recent_pl]
        pl_indices = [p[0] for p in recent_pl]

        # 兩組都上升
        high_rising = ph_prices[-1] > ph_prices[0]
        low_rising = pl_prices[-1] > pl_prices[0]

        if not (high_rising and low_rising):
            return None

        # 計算斜率，確認近似平行（斜率差異 < 50%）
        h_span = max(ph_indices[-1] - ph_indices[0], 1)
        l_span = max(pl_indices[-1] - pl_indices[0], 1)
        h_slope = (ph_prices[-1] - ph_prices[0]) / h_span
        l_slope = (pl_prices[-1] - pl_prices[0]) / l_span

        if h_slope <= 0 or l_slope <= 0:
            return None
        slope_ratio = min(h_slope, l_slope) / max(h_slope, l_slope)
        if slope_ratio < 0.5:
            return None  # 不夠平行

        # 估算當前通道上下緣
        upper_now = ph_prices[-1]
        lower_now = pl_prices[-1]
        channel_width = upper_now - lower_now
        current = close[-1]

        # 在上下緣 10% 內可交易（觸碰邊界）
        near_upper = current > upper_now - channel_width * 0.1
        near_lower = current < lower_now + channel_width * 0.1

        if near_upper or near_lower:
            return None  # 在邊緣，允許交易

        # 在通道中間，禁止交易
        return _safe_result(
            "neutral", 1,
            detail=f"⚠️ 上升通道內部 ${lower_now:,.2f}~${upper_now:,.2f} — 等待觸碰邊界再操作",
            trade_allowed=False,
            zone_type="ascending_channel",
            upper=float(upper_now), lower=float(lower_now),
        )

    @staticmethod
    def _check_volume_breakout(close, volume, upper, lower, n):
        """檢查最近 3 根是否有帶量突破。"""
        if n < 3:
            return None
        for i in range(-3, 0):
            c = close[i]
            v = volume[i]
            # 突破需要放量（>= 前 10 日均量 1.3 倍）
            avg_vol = np.mean(volume[max(0, n + i - 10):n + i]) if n + i > 10 else np.mean(volume[:n + i])
            if avg_vol <= 0:
                continue
            vol_ratio = v / avg_vol
            body = abs(c - close[i - 1]) if i > -n else 0
            if vol_ratio >= 1.3 and body > 0:
                if c > upper:
                    return "up"
                if c < lower:
                    return "down"
        return None


# ============================================================================
# 模組八：獲利強弱勢排序 (ProfitTarget)
# ============================================================================

# 形態→出場策略對照
_STRONG_PATTERNS = {"head_shoulders_bottom", "double_bottom", "bull_flag", "falling_wedge"}
_WEAK_PATTERNS = {"diamond_top", "rising_wedge", "descending_triangle"}


class ProfitTarget:
    """
    依據形態強度決定出場策略：
    - 強勢形態 → 移動停利 (Trailing Stop at MA10)，吃足波段
    - 弱勢形態 → 固定停利 (1:1.5 ~ 1:2 R/R)，快速獲利了結
    """

    @staticmethod
    def analyze(pattern_result: dict, sr_result: dict) -> dict:
        try:
            return ProfitTarget._analyze(pattern_result, sr_result)
        except Exception as e:
            _log.debug(f"ProfitTarget error: {e}")
            return _safe_result(detail="獲利策略不可用",
                                exit_strategy="none", trailing_ma=None,
                                target_rr=None)

    @staticmethod
    def _analyze(pattern_result: dict, sr_result: dict) -> dict:
        pattern_name = pattern_result.get("pattern")
        if not pattern_name:
            return _safe_result(
                detail="無形態，使用預設出場",
                exit_strategy="default", trailing_ma=None,
                target_rr=1.5,
            )

        # 強勢形態：使用移動停利
        if pattern_name in _STRONG_PATTERNS:
            detail_map = {
                "head_shoulders_bottom": "頭肩底突破走勢深遠",
                "double_bottom": "W底突破後動能強勁",
                "bull_flag": "旗形突破趨勢延續",
                "falling_wedge": "下降楔形反轉力道強",
            }
            detail = detail_map.get(pattern_name, "強勢形態")
            return _safe_result(
                "bullish", 5,
                detail=f"🚀 {detail} — 移動停利 (跌破 MA10 才出場)，吃足波段",
                exit_strategy="trailing_stop",
                trailing_ma=10,
                target_rr=None,
                pattern_type="strong",
            )

        # 弱勢形態：固定停利
        if pattern_name in _WEAK_PATTERNS:
            return _safe_result(
                "neutral", 2,
                detail="📊 弱勢形態 — 固定停利 (R/R 1:2)，達標後獲利了結一半",
                exit_strategy="fixed_target",
                trailing_ma=None,
                target_rr=2.0,
                pattern_type="weak",
            )

        # 中等形態（A 級但非強/弱）
        if pattern_name in ("ascending_triangle", "bear_flag"):
            rr = 2.0 if pattern_name == "ascending_triangle" else 1.5
            return _safe_result(
                "neutral", 3,
                detail=f"📊 中等形態 — 固定停利 (R/R 1:{rr:.1f})，搭配移動停利保護",
                exit_strategy="hybrid",
                trailing_ma=10,
                target_rr=rr,
                pattern_type="medium",
            )

        # 其他：中性預設
        return _safe_result(
            "neutral", 3,
            detail="📊 一般形態 — 固定停利 (R/R 1:1.5)",
            exit_strategy="fixed_target",
            trailing_ma=None,
            target_rr=1.5,
            pattern_type="default",
        )


# ============================================================================
# 統一入口
# ============================================================================

def compute_ta_overlay(df: pd.DataFrame, full: bool = False) -> dict:
    """
    計算進階技術分析疊加層。

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV 資料。
    full : bool
        False = quick mode (模組 1+2)，True = full mode (全部 8 模組)。

    Returns
    -------
    dict with keys: ta_overlay, ta_confidence, ta_confidence_max
    """
    _skip = _safe_result(detail="quick mode 略過")
    overlay = {}

    # 模組一：K 線力道（always）
    kline = KLineSentiment.analyze(df)
    overlay["kline"] = kline

    # 模組二：均線過濾（always）
    ma_cross = MACrossFilter.analyze(df)
    overlay["ma_cross"] = ma_cross

    if full:
        # 模組三：支撐壓力
        sr = SupportResistance.analyze(df)
        overlay["support_resistance"] = sr

        # 模組四：形態辨識
        pattern = ChartPatterns.analyze(df)
        overlay["chart_pattern"] = pattern

        # 模組五：回測確認（依賴 1+3+4）
        retest = RetestConfirmation.analyze(df, kline, sr, pattern)
        overlay["retest"] = retest

        # 模組六：訊號權重分級（依賴 4+5）
        confidence = SignalConfidence.analyze(pattern, retest)
        overlay["signal_confidence"] = confidence

        # 模組七：等待與避險過濾
        notrade = NoTradeZone.analyze(df)
        overlay["no_trade_zone"] = notrade

        # 模組八：獲利強弱排序（依賴 4+3）
        profit = ProfitTarget.analyze(pattern, sr)
        overlay["profit_target"] = profit
    else:
        overlay["support_resistance"] = _skip
        overlay["chart_pattern"] = _skip
        overlay["retest"] = _skip
        overlay["signal_confidence"] = _skip
        overlay["no_trade_zone"] = _skip
        overlay["profit_target"] = _skip

    # 計算總力道（8 模組 x 5 = 40 滿分）
    ta_confidence = sum(overlay[k].get("strength", 0) for k in overlay)
    ta_confidence_max = 40  # 8 modules x 5

    return {
        "ta_overlay": overlay,
        "ta_confidence": ta_confidence,
        "ta_confidence_max": ta_confidence_max,
    }
