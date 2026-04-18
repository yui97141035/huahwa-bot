"""
wp_content.py — Gemini AI 改寫引擎
將 PTT 真實故事改寫為虛構小說連載。
"""

import re
import logging
from google import genai
from google.genai import types

log = logging.getLogger("wp-poster.content")

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]

SYSTEM_PROMPT = (
    "你是一位全能型網路小說家，精通情感糾葛、職場暗鬥、家庭倫理、禁忌慾望等多種題材。\n"
    "你的文字節奏明快、場景寫實、衝突密集，讓讀者欲罷不能。\n"
    "你的任務是將真實故事改寫成虛構的連載小說。\n\n"
    "【核心原則 — 故事必須有結局】\n"
    "這是最重要的規則：每個系列必須有完整的故事弧線（開場→衝突→高潮→結局）。\n"
    "最後一集必須給出明確的結局，不可以懸在空中。\n"
    "結局可以是：圓滿收場、悲劇收場、反轉結局、開放式結局（但必須交代主線結果）。\n"
    "絕對禁止以「我不知道答案是什麼」「我們還能回到從前嗎？」這種空洞的自問收尾。\n\n"
    "【改寫規則】\n"
    "1. 所有真實姓名、暱稱、ID 替換為虛構名稱\n"
    "2. 所有真實地名、公司名、學校名替換為虛構名稱\n"
    "3. 保留故事的核心情節，但可以加強衝突和戲劇性\n"
    "4. 以第一人稱為主，讓讀者代入主角\n"
    "5. 每集 600～800 個中文字\n"
    "6. 根據故事長度拆分為 2～5 集，最後一集是結局\n\n"
    "【類型判斷 — 根據素材自動調整風格】\n"
    "・情感類（男女交往、曖昧、劈腿、追求）→ 情感爆點劇：高衝突、有渣有甜、結局收線\n"
    "・婚姻類（婆媳、外遇、離婚、冷暴力）→ 家庭倫理劇：文鬥為主，唇槍舌劍的對質場面\n"
    "・情慾類（性關係、禁忌、身體吸引）→ 感官小說：氛圍撩人，用留白和暗示，張力拉滿\n"
    "・職場/社會類（霸凌、不公、荒謬遭遇）→ 暗黑寫實：諷刺人性，結局帶有餘韻\n\n"
    "【寫作技法】\n"
    "1. 文鬥：安排至少一場角色之間的正面對質、攤牌、談判或言語交鋒。\n"
    "   對話要有殺傷力，每句話都藏著目的和立場，不是客套寒暄。\n"
    "2. 武鬥/行動：安排具體的行動場景——摔門而去、當眾撕破臉、攔截對方、\n"
    "   搶奪物品、肢體衝突等。不要只有心理活動，要有人「做了什麼」。\n"
    "3. 情慾場景：善用感官描寫（體溫、氣息、觸感、心跳），點到為止但張力拉滿。\n"
    "   曖昧和禁忌的吸引力是加分項，但必須服務於故事推進，不能原地打轉。\n"
    "4. 節奏控制：\n"
    "   - 每集前 100 字必須有 hook（衝突、反轉、意外發現）\n"
    "   - 對白佔比至少 35%，減少大段內心獨白\n"
    "   - 每 200 字推進一個情節節點，不要停在同一個情緒上反覆描寫\n"
    "5. 中間集結尾：可以用懸念或反轉吊胃口\n"
    "6. 最後一集結尾：必須給出結局。可以是：\n"
    "   - 爽快收場（打臉渣男/逆襲成功/浪子回頭）\n"
    "   - 虐心結局（錯過/分離/代價）\n"
    "   - 反轉結局（揭露真相/身份逆轉/意想不到的選擇）\n"
    "   - 餘韻式結局（主線已了結，用一個意象或場景收束，讓讀者回味）\n\n"
    "【輸出格式】\n"
    "第一行輸出系列標題，格式：《系列標題》\n"
    "每集以 ===第N集=== 分隔\n"
    "最後一集以 ===第N集（完結）=== 標記\n\n"
    "範例輸出：\n"
    "《深夜來電》\n"
    "===第1集===\n"
    "（600~800 字，以衝突或懸念開場）\n"
    "===第2集===\n"
    "（600~800 字，衝突升級，文鬥或武鬥場景）\n"
    "===第3集（完結）===\n"
    "（600~800 字，高潮 + 明確結局）\n"
)

# 為已有草稿續寫結局的 prompt
ENDING_PROMPT = (
    "你是一位全能型網路小說家。以下是一個連載小說系列的全部已寫內容，但這個故事沒有結局。\n"
    "請你為這個故事寫一集「完結篇」，給出明確的結局。\n\n"
    "【結局要求】\n"
    "1. 字數：600～800 個中文字\n"
    "2. 延續前面的人物、語氣、情節線\n"
    "3. 必須給出明確的結局 — 主線衝突必須有結果\n"
    "4. 結局類型（選最適合這個故事的）：\n"
    "   - 爽快收場：打臉/逆襲/真相大白\n"
    "   - 虐心結局：錯過/分離/無法回頭\n"
    "   - 反轉結局：意想不到的真相或選擇\n"
    "   - 溫暖收場：和解/成長/釋懷\n"
    "5. 安排至少一場正面對質或關鍵行動場景（文鬥或武鬥）\n"
    "6. 對白佔比至少 35%，不要只有內心獨白\n"
    "7. 最後一段要有收束感，讓讀者知道「故事到這裡結束了」\n\n"
    "【輸出格式】\n"
    "直接輸出完結篇的小說內容，不需要標題或分隔符號。\n"
)


class ContentGenerator:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        log.info(f"Gemini 改寫引擎已初始化 (主模型: {GEMINI_MODELS[0]})")

    def rewrite_story(self, ptt_title: str, ptt_content: str,
                      target_chars: int = 800) -> dict:
        """
        改寫 PTT 文章為小說連載。
        回傳: {"series_title": str, "episodes": list[str]}
        """
        prompt = (
            f"以下是一篇 PTT 文章，請將它改寫為虛構連載小說。\n"
            f"每集目標字數：約 {target_chars} 個中文字。\n\n"
            f"原文標題：{ptt_title}\n"
            f"原文內容：\n{ptt_content}\n"
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.9,
            max_output_tokens=8192,
        )

        response_text = self._call_gemini(config, prompt)
        return self._parse_response(response_text)

    def _call_gemini(self, config, contents: str) -> str:
        """嘗試所有 Gemini 模型，配額滿自動降級。"""
        last_err = None
        for model_name in GEMINI_MODELS:
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                result = resp.text
                if not result:
                    raise ValueError(f"Gemini {model_name} 回傳空內容（可能被 safety filter 攔截）")
                return result.strip()
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower():
                    log.warning(f"Gemini {model_name} 配額超限，嘗試下一個模型")
                    continue
                raise
        raise last_err

    @staticmethod
    def _parse_response(text: str) -> dict:
        """
        解析 AI 輸出，提取系列標題和各集內容。
        """
        # 提取系列標題
        title_match = re.search(r"《(.+?)》", text)
        series_title = title_match.group(0) if title_match else "《未命名故事》"

        # 依 ===第N集=== 或 ===第N集（完結）=== 分割
        parts = re.split(r"===\s*第\d+集(?:（完結）)?\s*===", text)

        episodes = []
        for part in parts:
            cleaned = part.strip()
            # 跳過標題區（第一段通常只有標題）
            if not cleaned or cleaned == series_title:
                continue
            # 移除可能殘留的標題行
            cleaned = re.sub(r"^《.+?》\s*", "", cleaned).strip()
            if len(cleaned) > 100:  # 至少 100 字才算有效集數
                episodes.append(cleaned)

        if not episodes:
            # fallback: 沒有正確分隔，把整段當一集
            full = re.sub(r"^《.+?》\s*", "", text).strip()
            full = re.sub(r"===\s*第\d+集\s*===", "", full).strip()
            if full:
                episodes = [full]

        log.info(f"改寫完成: {series_title} — {len(episodes)} 集")
        return {"series_title": series_title, "episodes": episodes}

    def generate_ending(self, series_title: str,
                        all_episodes: list[str]) -> str:
        """
        為已有的連載小說系列產生完結篇。
        回傳: 完結篇的文字內容
        """
        # 組合所有已有集數的內容（太長時只取每集的頭尾各 300 字）
        episodes_text = ""
        for i, ep in enumerate(all_episodes, 1):
            if len(ep) > 700:
                episodes_text += f"\n--- 第{i}集 ---\n{ep[:300]}\n...(中略)...\n{ep[-300:]}\n"
            else:
                episodes_text += f"\n--- 第{i}集 ---\n{ep}\n"

        prompt = (
            f"系列標題：{series_title}\n"
            f"目前共 {len(all_episodes)} 集，以下是全部已寫內容：\n"
            f"{episodes_text}\n\n"
            f"請為這個故事寫完結篇。"
        )

        config = types.GenerateContentConfig(
            system_instruction=ENDING_PROMPT,
            temperature=0.85,
            max_output_tokens=4096,
        )

        text = self._call_gemini(config, prompt)
        # 清理可能的格式殘留
        text = re.sub(r"^===.*?===\s*", "", text).strip()
        text = re.sub(r"^《.+?》\s*", "", text).strip()
        log.info(f"完結篇已產生: {series_title} — {len(text)} 字")
        return text

    def generate_fb_teaser(self, ep_content: str) -> str:
        """用 Gemini 產生 Facebook 分享摘要（1~2 句吸引人的懸念文字）。"""
        prompt = (
            "你是一位社群小編。以下是一篇連載小說的內容。\n"
            "請寫一段適合在 Facebook 分享的摘要文字，讓人想點進去看全文。\n\n"
            "【規則】\n"
            "1. 寫 1~2 個完整的句子\n"
            "2. 總長度約 30~60 個中文字\n"
            "3. 語氣：引人好奇、製造懸念，像書的封底文案\n"
            "4. 不要劇透結局，只暗示故事的張力和衝突\n"
            "5. 直接輸出摘要文字，不要加引號或其他格式標記\n"
            "6. 結尾用「⋯⋯」留下想像空間\n\n"
            f"【小說內容】\n{ep_content[:800]}\n"
        )
        config = types.GenerateContentConfig(
            temperature=0.8, max_output_tokens=200,
        )
        text = self._call_gemini(config, prompt)
        text = text.strip().strip('"「」""')
        if text and not text.endswith("⋯⋯"):
            # 找最後一個句末標點，截到那裡
            for i in range(len(text) - 1, -1, -1):
                if text[i] in "。！？⋯…，":
                    text = text[:i + 1]
                    break
            if not text.endswith("⋯⋯"):
                text += "⋯⋯"
        return text

    def generate_teaser(self, current_ep_content: str,
                        next_ep_content: str) -> str:
        """用 Gemini 產生下集預告（2~3 句懸念式摘要）。"""
        prompt = (
            "你是一位連載小說的編輯。以下是目前這一集的結尾內容，以及下一集的開頭內容。\n"
            "請根據這些內容，寫一段「下集預告」，要能吊起讀者的胃口，製造懸念。\n\n"
            "【規則】\n"
            "1. 寫 2~3 個完整的句子，每句必須以句號、問號或驚嘆號結尾\n"
            "2. 總長度約 60~120 個中文字\n"
            "3. 語氣：神祕、引人入勝，像電視劇的下集預告旁白\n"
            "4. 不要劇透太多，但要讓讀者忍不住想看下一集\n"
            "5. 直接輸出預告文字，不要加引號或其他格式標記\n\n"
            f"【本集結尾】\n{current_ep_content[-400:]}\n\n"
            f"【下集開頭】\n{next_ep_content[:400]}\n"
        )
        config = types.GenerateContentConfig(
            temperature=0.8, max_output_tokens=500,
        )
        text = self._call_gemini(config, prompt)
        text = text.strip().strip('"「」""')
        # 確保 teaser 以完整標點結尾，避免斷句
        if text and text[-1] not in "。！？⋯…":
            # 找最後一個句末標點，截到那裡
            for i in range(len(text) - 1, -1, -1):
                if text[i] in "。！？⋯…":
                    text = text[:i + 1]
                    break
            else:
                text += "⋯⋯"
        return text
