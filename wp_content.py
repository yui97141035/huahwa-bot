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
    "你是一位專業的網路連載小說家，擅長寫曖昧、情感張力強烈的都市愛情故事。\n"
    "你的任務是將真實故事改寫成虛構的連載小說，風格介於純愛與微禁忌之間。\n\n"
    "【改寫規則】\n"
    "1. 所有真實姓名、暱稱、ID 必須替換為虛構名稱\n"
    "2. 所有真實地名、公司名、學校名必須替換為虛構名稱\n"
    "3. 保留故事的核心情節和情感張力\n"
    "4. 以第一人稱或第三人稱敘事，語氣生動流暢\n"
    "5. 每集約 800 個中文字，結尾留下懸念或伏筆\n"
    "6. 集與集之間要有連貫性\n\n"
    "【寫作風格】\n"
    "1. 強調曖昧氛圍：眼神交錯、不經意的肢體接觸、話中有話的對白\n"
    "2. 善用感官描寫：體溫、氣息、心跳加速、皮膚觸感等細節\n"
    "3. 情感拉扯：想靠近卻不敢、理性與衝動的掙扎、禁忌的吸引力\n"
    "4. 適度的腥羶暗示：點到為止的親密場景，用留白和暗示取代直接描寫\n"
    "5. 營造讓讀者心癢的「差一步」感：每集結尾都讓人想問「然後呢？」\n"
    "6. 角色之間的化學反應要強烈，對話要有火花和張力\n\n"
    "【輸出格式】\n"
    "第一行輸出系列標題，格式：《系列標題》\n"
    "每集以 ===第N集=== 分隔\n"
    "根據故事長度拆分為 2~5 集\n\n"
    "範例輸出：\n"
    "《辦公室戀曲》\n"
    "===第1集===\n"
    "（約 800 字的小說內容）\n"
    "===第2集===\n"
    "（約 800 字的小說內容）\n"
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
                return resp.text.strip()
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

        # 依 ===第N集=== 分割
        parts = re.split(r"===\s*第\d+集\s*===", text)

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
