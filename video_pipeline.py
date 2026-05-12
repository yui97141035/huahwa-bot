"""
video_pipeline.py — 全自動影片產線
讀取已發布文章 → Gemini 生成腳本 → Edge-TTS 語音 → AI 生圖 → 合成影片 → 上傳 YouTube
"""

import os
import re
import json
import time
import asyncio
import logging
import argparse
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
from PIL import Image
import requests as http_requests
import edge_tts
from google import genai
from google.genai import types
from moviepy import (
    VideoClip, ImageClip, AudioFileClip, TextClip,
    CompositeVideoClip, concatenate_videoclips,
)
from dotenv import load_dotenv

from wp_database import Database

# YouTube
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("video_pipeline.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("video-pipeline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
TTS_VOICE = "zh-CN-XiaoxiaoNeural"
VIDEO_SIZE = (1920, 1080)
VIDEO_FPS = 24
CHECK_INTERVAL = 4 * 3600  # 4 hours

WORK_DIR = Path("video_workspace")
IMAGE_CACHE_DIR = WORK_DIR / "image_cache"
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_PATH = Path("youtube_token.json")

# Subtitle font candidates (first existing path wins)
FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Supplemental/Songti.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msjh.ttc",
]

SCRIPT_SYSTEM_PROMPT = (
    "你是一位 YouTube 影片腳本撰稿人。你的任務是將小說文章改寫成適合旁白朗讀的影片腳本。\n\n"
    "【規則】\n"
    "1. 將文章改寫為口語化的旁白腳本，適合朗讀\n"
    "2. 把腳本分成 4~6 段，每段約 30~45 秒的朗讀量（約 80~120 個中文字）\n"
    "3. 每段需要附帶一個英文圖片描述（image_prompt），用於 AI 生圖\n"
    "4. 圖片描述要具體、視覺化，適合生成唯美風格的插圖\n"
    "5. 產生適合的 YouTube 標題、描述和標籤\n"
    "6. 【重要】在 characters 欄位定義每個角色的固定外貌特徵（髮型、髮色、眼睛、膚色、服裝），"
    "所有段落的 image_prompt 必須引用相同的角色外貌描述，確保視覺一致\n"
    "7. 在 style 欄位定義統一的藝術風格，所有 image_prompt 都要使用這個風格\n"
    "8. image_prompt 中必須包含角色的完整外貌描述（不要只寫名字），讓 AI 生圖時能畫出一致的角色\n\n"
    "【輸出格式】嚴格使用以下 JSON 格式，不要加 markdown 標記：\n"
    "{\n"
    '  "title": "吸引人的影片標題",\n'
    '  "description": "YouTube 描述文字（2~3 句）",\n'
    '  "tags": ["標籤1", "標籤2", "標籤3"],\n'
    '  "style": "統一的藝術風格描述，例如: cinematic digital illustration, soft warm lighting, anime-inspired",\n'
    '  "characters": {\n'
    '    "角色名": "detailed English appearance: hair, eyes, skin, clothing, age, build",\n'
    '    "角色名2": "detailed English appearance..."\n'
    "  },\n"
    '  "segments": [\n'
    '    {"narration": "旁白文字...", "image_prompt": "A cinematic scene of [完整角色外貌描述] doing ..."},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
)


def _find_font() -> str:
    """Find a suitable CJK font on the system."""
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    # Fallback: try fc-list on Linux
    try:
        import subprocess
        result = subprocess.run(
            ["fc-list", ":lang=zh", "-f", "%{file}\n"],
            capture_output=True, text=True, timeout=5,
        )
        if result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    log.warning("找不到中文字體，字幕可能無法正確顯示")
    return "Arial"


# ---------------------------------------------------------------------------
# VideoMaker
# ---------------------------------------------------------------------------
class VideoMaker:
    def __init__(self):
        self.db = Database()
        self.gemini = genai.Client(api_key=self._get_gemini_key())
        self.font_path = _find_font()
        self.youtube_service = None

        WORK_DIR.mkdir(exist_ok=True)
        IMAGE_CACHE_DIR.mkdir(exist_ok=True)

        log.info(f"VideoMaker 已初始化 (字體: {self.font_path})")

    @staticmethod
    def _get_gemini_key() -> str:
        key = os.getenv("GEMINI_API_KEY_WP") or os.getenv("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("缺少 GEMINI_API_KEY 環境變數")
        return key

    # ── Step 1: Fetch article ──────────────────────────────────

    def fetch_article(self) -> dict | None:
        """從 DB 取得下一篇已發布但未製片的文章。"""
        return self.db.get_next_video_pending()

    # ── Step 2: Gemini script generation ───────────────────────

    def generate_script(self, article: dict) -> dict:
        """用 Gemini 將文章改寫為影片腳本。"""
        series = article["series_title"].strip("《》")
        ep_num = article["episode_num"]
        content = article["content"]

        prompt = (
            f"以下是連載小說《{series}》第 {ep_num} 集的內容，"
            f"請將它改寫為 YouTube 影片旁白腳本。\n\n"
            f"【文章內容】\n{content}\n"
        )

        config = types.GenerateContentConfig(
            system_instruction=SCRIPT_SYSTEM_PROMPT,
            temperature=0.8,
            max_output_tokens=8192,
            response_mime_type="application/json",
        )

        response_text = self._call_gemini(config, prompt)
        return self._parse_script(response_text, series, ep_num)

    def _call_gemini(self, config, contents: str) -> str:
        """嘗試所有 Gemini 模型，配額滿自動降級。"""
        last_err = None
        for model_name in GEMINI_MODELS:
            try:
                resp = self.gemini.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
                result = resp.text
                if not result:
                    raise ValueError(f"Gemini {model_name} 回傳空內容")
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
    def _repair_json(text: str) -> str:
        """嘗試修復常見的 JSON 問題（未跳脫的換行、控制字元等）。"""
        # Fix unescaped newlines inside string values
        fixed = re.sub(
            r'(?<=": ")(.*?)(?="[,\s*}])',
            lambda m: m.group(0).replace("\n", "\\n").replace("\t", "\\t"),
            text,
            flags=re.DOTALL,
        )
        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        return fixed

    @staticmethod
    def _parse_script(text: str, series: str, ep_num: int) -> dict:
        """解析 Gemini 回傳的 JSON 腳本。"""
        # Strip markdown code fence if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())

        # Try parsing, with progressively more aggressive repair
        data = None
        for attempt_text in [text, VideoMaker._repair_json(text)]:
            try:
                data = json.loads(attempt_text)
                break
            except json.JSONDecodeError:
                # Try extracting just the JSON object
                match = re.search(r"\{[\s\S]*\}", attempt_text)
                if match:
                    try:
                        data = json.loads(match.group())
                        break
                    except json.JSONDecodeError:
                        continue

        if data is None:
            raise ValueError(f"無法解析 Gemini 回傳的 JSON:\n{text[:500]}")

        if "segments" not in data or not data["segments"]:
            raise ValueError("腳本缺少 segments")

        data.setdefault("title", f"{series}｜EP.{ep_num}")
        data.setdefault("description", f"《{series}》第 {ep_num} 集")
        data.setdefault("tags", [series, "連載小說", "故事"])
        data.setdefault("style", "cinematic digital illustration, soft warm lighting, anime-inspired aesthetic")
        data.setdefault("characters", {})

        # Build character appearance prefix for image prompts
        char_desc = "; ".join(
            f"{name}: {desc}" for name, desc in data["characters"].items()
        )

        for seg in data["segments"]:
            if "narration" not in seg or not seg["narration"]:
                raise ValueError("segment 缺少 narration")
            seg.setdefault(
                "image_prompt",
                "A beautiful cinematic landscape, soft lighting, dreamy atmosphere",
            )
            # Prepend style + character descriptions to ensure consistency
            prompt = seg["image_prompt"]
            if data["style"] and data["style"].lower() not in prompt.lower():
                prompt = f"{prompt}, {data['style']}"
            if char_desc and char_desc[:30].lower() not in prompt.lower():
                prompt = f"{prompt}. Characters: {char_desc}"
            seg["image_prompt"] = prompt

        log.info(f"腳本生成完成: {data['title']} ({len(data['segments'])} 段)")
        return data

    # ── Step 3: Edge-TTS ───────────────────────────────────────

    async def _generate_segment_tts(self, text: str, output_path: Path) -> list[tuple]:
        """生成單段 TTS 音訊，回傳 word boundaries [(start_s, end_s, text), ...]。"""
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        word_boundaries = []

        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 10_000_000  # 100-ns ticks → seconds
                    duration = chunk["duration"] / 10_000_000
                    word_boundaries.append((start, start + duration, chunk["text"]))

        return word_boundaries

    async def _generate_all_tts(self, script: dict, work_dir: Path) -> list[dict]:
        """依序生成所有段落 TTS。"""
        segments_info = []

        for i, seg in enumerate(script["segments"]):
            audio_path = work_dir / f"segment_{i:02d}.mp3"
            narration = seg["narration"]

            log.info(f"TTS 生成中: 段落 {i+1}/{len(script['segments'])} ({len(narration)} 字)")
            word_boundaries = await self._generate_segment_tts(narration, audio_path)

            audio_clip = AudioFileClip(str(audio_path))
            duration = audio_clip.duration
            audio_clip.close()

            # Fallback: edge-tts 中文不回傳 WordBoundary，改用比例時間軸
            if not word_boundaries:
                word_boundaries = self._proportional_boundaries(narration, duration)
                log.info(f"  使用比例時間軸字幕 ({len(word_boundaries)} 字元)")

            segments_info.append({
                "index": i,
                "narration": narration,
                "audio_path": str(audio_path),
                "duration": duration,
                "word_boundaries": word_boundaries,
                "image_prompt": seg["image_prompt"],
            })

            log.info(f"段落 {i+1} TTS 完成: {duration:.1f}s")

        total = sum(s["duration"] for s in segments_info)
        log.info(f"TTS 全部完成: 共 {len(segments_info)} 段, {total:.1f}s")
        return segments_info

    def generate_tts(self, script: dict, work_dir: Path) -> list[dict]:
        """為每段生成 TTS 音訊 + 字幕資料。"""
        return asyncio.run(self._generate_all_tts(script, work_dir))

    @staticmethod
    def _proportional_boundaries(text: str, duration: float) -> list[tuple]:
        """逐字按比例分配時間軸（edge-tts 無 WordBoundary 時的備案）。"""
        if not text:
            return []
        char_dur = duration / len(text)
        return [(i * char_dur, (i + 1) * char_dur, ch) for i, ch in enumerate(text)]

    # ── Step 4: AI image generation (FLUX via HuggingFace) ────

    # HuggingFace Spaces fallback chain (FLUX models, free with HF_TOKEN)
    HF_SPACES = [
        ("black-forest-labs/FLUX.1-schnell", "/infer", {
            "randomize_seed": False, "width": 1920, "height": 1080,
            "num_inference_steps": 4,
        }),
        ("ByteDance/Hyper-FLUX-8Steps-LoRA", "/process_image", {
            "height": 1080, "width": 1920, "steps": 8, "scales": 3.5,
        }),
    ]

    def _generate_ai_image(self, prompt: str, output_path: Path,
                           seed: int = 42) -> bool:
        """用 HuggingFace Space (FLUX) 生成 AI 圖片。"""
        try:
            from gradio_client import Client
        except ImportError:
            log.warning("gradio_client 未安裝，跳過 AI 生圖")
            return False

        hf_token = os.getenv("HF_TOKEN", "")

        for space_name, api_name, params in self.HF_SPACES:
            try:
                client = Client(
                    space_name, verbose=False,
                    token=hf_token or None,
                )
                result = client.predict(
                    prompt=prompt,
                    seed=float(seed) if "seed" in str(params) or api_name == "/infer" else seed,
                    api_name=api_name,
                    **params,
                )
                # Result may be a tuple (image_path, seed) or just a path
                img_path = result[0] if isinstance(result, (list, tuple)) else result
                if img_path and os.path.exists(str(img_path)):
                    img = Image.open(str(img_path)).convert("RGB")
                    if img.size != VIDEO_SIZE:
                        img = img.resize(VIDEO_SIZE, Image.LANCZOS)
                    img.save(output_path)
                    log.info(f"AI 圖片生成完成: {output_path.name} (via {space_name})")
                    return True
            except Exception as e:
                err = str(e)
                if "quota" in err.lower() or "GPU" in err:
                    log.warning(f"HF Space {space_name} GPU 配額不足，嘗試下一個: {err[:80]}")
                else:
                    log.warning(f"HF Space {space_name} 失敗: {err[:120]}")

        return False

    def _download_picsum(self, seed: str, output_path: Path) -> bool:
        """從 Picsum Photos 下載風景照（備用方案）。"""
        for attempt in range(5):
            current_seed = f"{seed}-v{attempt}" if attempt > 0 else seed
            url = f"https://picsum.photos/seed/{current_seed}/1920/1080"
            try:
                resp = http_requests.get(url, timeout=20, allow_redirects=True)
                if resp.status_code != 200:
                    continue
                with open(output_path, "wb") as f:
                    f.write(resp.content)
                log.info(f"Picsum 備用圖片: {output_path.name} (seed: {current_seed})")
                return True
            except Exception:
                continue
        return False

    def generate_images(self, segments_info: list, work_dir: Path, episode_id: int = 0):
        """為每段生成 AI 圖片，失敗降級 Picsum，最後用漸層色塊。"""
        log.info(f"開始生成 {len(segments_info)} 張圖片...")
        ai_ok = 0
        for i, seg in enumerate(segments_info):
            output_path = work_dir / f"scene_{seg['index']:02d}.png"
            seg["image_path"] = str(output_path)
            prompt = seg.get("image_prompt", "A cinematic landscape scene")

            log.info(f"生成圖片 {i+1}/{len(segments_info)}: {prompt[:60]}...")

            # Try AI → Picsum → gradient fallback
            success = self._generate_ai_image(
                prompt, output_path, seed=episode_id * 100 + i
            )
            if success:
                ai_ok += 1
                # Brief delay between AI requests to respect rate limits
                if i + 1 < len(segments_info):
                    time.sleep(5)
            else:
                log.info("降級使用 Picsum 圖片...")
                success = self._download_picsum(
                    f"video-ep{episode_id}-seg{i}", output_path
                )
            if not success:
                log.warning(f"段落 {i} 所有圖片來源失敗，使用漸層色塊")
                self._create_fallback_image(output_path)

        log.info(f"圖片生成完成: AI={ai_ok}, 備用={len(segments_info)-ai_ok}")

    @staticmethod
    def _create_fallback_image(output_path: Path):
        """建立漸層色塊作為備用圖片。"""
        arr = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for y in range(1080):
            ratio = y / 1080
            arr[y, :] = [
                int(25 + 30 * ratio),
                int(25 + 20 * ratio),
                int(50 + 40 * ratio),
            ]
        Image.fromarray(arr).save(output_path)

    # ── Step 5: Video assembly ─────────────────────────────────

    def _make_ken_burns_clip(self, image_path: str, duration: float) -> VideoClip:
        """建立 Ken Burns 慢速縮放特效的影片片段。"""
        img = Image.open(image_path).convert("RGB")
        scale = 1.3
        base_w = int(VIDEO_SIZE[0] * scale)
        base_h = int(VIDEO_SIZE[1] * scale)
        img = img.resize((base_w, base_h), Image.LANCZOS)
        img_array = np.array(img)

        tw, th = VIDEO_SIZE

        def make_frame(t):
            progress = t / max(duration, 0.001)
            zoom = 1.0 + 0.15 * progress
            crop_w = min(int(base_w / zoom), base_w)
            crop_h = min(int(base_h / zoom), base_h)
            x = (base_w - crop_w) // 2
            y = (base_h - crop_h) // 2
            cropped = img_array[y : y + crop_h, x : x + crop_w]
            resized = np.array(
                Image.fromarray(cropped).resize((tw, th), Image.LANCZOS)
            )
            return resized

        return VideoClip(make_frame, duration=duration).with_fps(VIDEO_FPS)

    def _group_subtitles(self, word_boundaries: list, max_chars: int = 20) -> list[tuple]:
        """將 word boundaries 分組為字幕行。回傳 [(start, end, text), ...]。"""
        if not word_boundaries:
            return []

        # Sentence-ending punctuation — hard break
        HARD_BREAK = set("。！？")
        # Clause punctuation — soft break (only if current line is >= 8 chars)
        SOFT_BREAK = set("，；：、）」】》…—")

        lines = []
        current_text = ""
        current_start = None
        current_end = None

        for start, end, text in word_boundaries:
            if current_start is None:
                current_start = start
            current_text += text
            current_end = end

            should_break = False
            if text in HARD_BREAK:
                should_break = True
            elif text in SOFT_BREAK and len(current_text) >= 8:
                should_break = True
            elif len(current_text) >= max_chars:
                should_break = True

            if should_break:
                lines.append((current_start, current_end, current_text))
                current_text = ""
                current_start = None
                current_end = None

        # Capture any remaining text
        if current_text:
            lines.append((current_start, current_end, current_text))

        return lines

    def _make_subtitle_clips(self, word_boundaries: list, duration: float) -> list:
        """從 word boundaries 建立字幕 TextClip 列表。"""
        lines = self._group_subtitles(word_boundaries)
        if not lines:
            return []

        clips = []
        for i, (start, end, text) in enumerate(lines):
            text = text.strip()
            if not text:
                continue
            # End at next subtitle's start to prevent overlap; last one runs to segment end
            display_end = lines[i + 1][0] if i + 1 < len(lines) else duration
            # Ensure minimum display time of 0.3s
            display_end = max(display_end, start + 0.3)
            try:
                txt_clip = (
                    TextClip(
                        text=text,
                        font=self.font_path,
                        font_size=52,
                        color="white",
                        stroke_color="black",
                        stroke_width=2,
                        size=(VIDEO_SIZE[0] - 200, None),
                        method="caption",
                    )
                    .with_start(start)
                    .with_end(min(display_end, duration))
                    .with_position(("center", VIDEO_SIZE[1] - 140))
                )
                clips.append(txt_clip)
            except Exception as e:
                log.warning(f"字幕建立失敗: {e}")

        return clips

    def _make_title_card(self, title: str, duration: float = 3.0) -> VideoClip:
        """建立片頭標題卡（深色背景 + 大標題）。"""
        bg_array = np.full((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), (20, 20, 40), dtype=np.uint8)
        bg = ImageClip(bg_array).with_duration(duration).with_fps(VIDEO_FPS)
        try:
            txt = (
                TextClip(
                    text=title,
                    font=self.font_path,
                    font_size=72,
                    color="white",
                    size=(VIDEO_SIZE[0] - 300, None),
                    method="caption",
                )
                .with_duration(duration)
                .with_position("center")
            )
            return CompositeVideoClip([bg, txt], size=VIDEO_SIZE).with_fps(VIDEO_FPS)
        except Exception:
            return bg

    def _make_outro(self, duration: float = 3.0) -> VideoClip:
        """建立片尾訂閱提醒。"""
        bg_array = np.full((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), (20, 20, 40), dtype=np.uint8)
        bg = ImageClip(bg_array).with_duration(duration).with_fps(VIDEO_FPS)
        try:
            txt = (
                TextClip(
                    text="喜歡的話請訂閱頻道\n追蹤更多精彩故事",
                    font=self.font_path,
                    font_size=60,
                    color="white",
                    size=(VIDEO_SIZE[0] - 300, None),
                    method="caption",
                )
                .with_duration(duration)
                .with_position("center")
            )
            return CompositeVideoClip([bg, txt], size=VIDEO_SIZE).with_fps(VIDEO_FPS)
        except Exception:
            return bg

    def assemble_video(self, script: dict, segments_info: list, output_path: Path) -> Path:
        """合成最終影片：片頭 + 各段(圖片+音訊+字幕) + 片尾。"""
        log.info("開始合成影片...")

        clips = []

        # Title card (3s, silent)
        title_clip = self._make_title_card(script["title"])
        clips.append(title_clip)

        # Each segment: Ken Burns image + audio + subtitles
        for seg in segments_info:
            duration = seg["duration"]

            video_clip = self._make_ken_burns_clip(seg["image_path"], duration)
            audio_clip = AudioFileClip(seg["audio_path"])
            sub_clips = self._make_subtitle_clips(seg["word_boundaries"], duration)

            if sub_clips:
                segment_clip = CompositeVideoClip(
                    [video_clip] + sub_clips, size=VIDEO_SIZE
                )
            else:
                segment_clip = video_clip

            segment_clip = segment_clip.with_audio(audio_clip)
            clips.append(segment_clip)

        # Outro (3s, silent)
        outro_clip = self._make_outro()
        clips.append(outro_clip)

        # Concatenate all clips
        final = concatenate_videoclips(clips, method="compose")

        log.info(f"輸出影片: {output_path} ({final.duration:.1f}s)")
        final.write_videofile(
            str(output_path),
            fps=VIDEO_FPS,
            codec="libx264",
            audio_codec="aac",
            bitrate="5000k",
            preset="medium",
            threads=4,
            logger="bar",
        )

        final.close()
        for clip in clips:
            try:
                clip.close()
            except Exception:
                pass

        log.info(f"影片合成完成: {output_path}")
        return output_path

    # ── Step 6: YouTube upload ─────────────────────────────────

    def auth_youtube(self):
        """執行 YouTube OAuth2 授權流程（首次使用時）。"""
        client_id = os.getenv("YOUTUBE_CLIENT_ID", "")
        client_secret = os.getenv("YOUTUBE_CLIENT_SECRET", "")

        if not client_id or not client_secret:
            raise RuntimeError(
                "缺少 YOUTUBE_CLIENT_ID / YOUTUBE_CLIENT_SECRET 環境變數\n"
                "請到 Google Cloud Console 建立 OAuth2 憑證"
            )

        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost"],
                }
            },
            YOUTUBE_SCOPES,
        )
        creds = flow.run_local_server(port=0)

        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

        log.info(f"YouTube 授權成功，token 已存至 {TOKEN_PATH}")
        return creds

    def _get_youtube_service(self):
        """取得 YouTube API 服務（自動刷新 token）。"""
        if self.youtube_service:
            return self.youtube_service

        creds = None
        if TOKEN_PATH.exists():
            creds = Credentials.from_authorized_user_file(
                str(TOKEN_PATH), YOUTUBE_SCOPES
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(GoogleAuthRequest())
                with open(TOKEN_PATH, "w") as f:
                    f.write(creds.to_json())
            else:
                creds = self.auth_youtube()

        self.youtube_service = build("youtube", "v3", credentials=creds)
        return self.youtube_service

    def upload_youtube(self, video_path: Path, script: dict) -> str:
        """上傳影片到 YouTube，回傳 video ID。"""
        youtube = self._get_youtube_service()

        title = script["title"][:100]
        description = script.get("description", "")
        tags = script.get("tags", [])

        body = {
            "snippet": {
                "title": title,
                "description": (
                    f"{description}\n\n"
                    f"{'  '.join('#' + t for t in tags[:5])}\n\n"
                    f"---\n"
                    f"本影片由 AI 自動生成"
                ),
                "tags": tags[:30],
                "categoryId": "24",  # Entertainment
                "defaultLanguage": "zh-Hant",
            },
            "status": {
                "privacyStatus": "public",
                "selfDeclaredMadeForKids": False,
            },
        }

        media = MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=10 * 1024 * 1024,
        )

        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media,
        )

        log.info(f"開始上傳 YouTube: {title}")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                log.info(f"上傳進度: {int(status.progress() * 100)}%")

        video_id = response["id"]
        log.info(f"YouTube 上傳完成: https://youtube.com/watch?v={video_id}")
        return video_id

    def _upload_thumbnail(self, video_id: str, image_path: str):
        """上傳自訂縮圖到 YouTube。"""
        try:
            youtube = self._get_youtube_service()
            media = MediaFileUpload(image_path, mimetype="image/png")
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=media,
            ).execute()
            log.info(f"縮圖已上傳: {video_id}")
        except Exception as e:
            log.warning(f"縮圖上傳失敗（可能需要驗證頻道）: {e}")

    # ── Main pipeline ──────────────────────────────────────────

    def process_one(self, skip_upload: bool = False) -> bool:
        """處理一篇文章。回傳是否有處理。"""
        article = self.fetch_article()
        if not article:
            log.info("沒有待製片的文章")
            return False

        ep_id = article["id"]
        series = article["series_title"].strip("《》")
        ep_num = article["episode_num"]
        log.info(f"開始製片: {series} EP.{ep_num} (ID={ep_id})")

        self.db.mark_video_processing(ep_id)

        work_dir = WORK_DIR / f"ep_{ep_id}"
        work_dir.mkdir(exist_ok=True)
        output_path = work_dir / "output.mp4"

        try:
            # Step 2: Generate script
            script = self.generate_script(article)
            with open(work_dir / "script.json", "w", encoding="utf-8") as f:
                json.dump(script, f, ensure_ascii=False, indent=2)

            # Step 3: TTS
            segments_info = self.generate_tts(script, work_dir)

            # Step 4: Generate images
            self.generate_images(segments_info, work_dir, episode_id=ep_id)

            # Step 5: Assemble video
            self.assemble_video(script, segments_info, output_path)

            # Step 6: Upload to YouTube (optional)
            if skip_upload:
                log.info(f"跳過 YouTube 上傳（影片路徑: {output_path}）")
                self.db.mark_video_uploaded(ep_id, str(output_path), "")
            else:
                video_id = self.upload_youtube(output_path, script)
                if segments_info:
                    self._upload_thumbnail(video_id, segments_info[0]["image_path"])
                self.db.mark_video_uploaded(ep_id, str(output_path), video_id)

            log.info(f"製片完成: {series} EP.{ep_num}")
            return True

        except Exception as e:
            log.error(f"製片失敗: {series} EP.{ep_num} — {e}\n{traceback.format_exc()}")
            self.db.mark_video_failed(ep_id)
            return False

    def run(self):
        """主循環：每 4 小時檢查新文章。"""
        log.info("=== Video Pipeline 啟動 ===")

        while True:
            try:
                processed = self.process_one()
                if processed:
                    time.sleep(60)  # Brief cooldown before next
                    continue
            except Exception:
                log.error(f"主循環異常:\n{traceback.format_exc()}")

            log.info(f"下次檢查: {CHECK_INTERVAL // 3600} 小時後")
            time.sleep(CHECK_INTERVAL)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="全自動影片產線")
    parser.add_argument("--auth", action="store_true", help="執行 YouTube OAuth2 授權")
    parser.add_argument("--once", action="store_true", help="只處理一篇文章後退出")
    parser.add_argument("--no-upload", action="store_true", help="只產生影片，不上傳 YouTube")
    args = parser.parse_args()

    maker = VideoMaker()

    if args.auth:
        maker.auth_youtube()
        print("YouTube 授權完成！")
        return

    if args.once:
        success = maker.process_one(skip_upload=args.no_upload)
        if success:
            print("影片製作完成！")
        else:
            print("沒有待處理的文章，或處理失敗。")
        return

    maker.run()


if __name__ == "__main__":
    main()
