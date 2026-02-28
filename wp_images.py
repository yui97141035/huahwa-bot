"""
wp_images.py — 文章配圖模組
根據文章內容自動取得配圖，上傳並嵌入文章開頭。
"""

import re
import logging
import requests

log = logging.getLogger("wp-poster.images")

# 中文關鍵字 → 英文搜尋詞（避免正面人像，偏風景/剪影/背影/唯美）
KEYWORD_MAP = {
    "辦公室": "window,cityscape,dusk",
    "咖啡": "coffee,window,rain",
    "雨": "rain,window,bokeh",
    "夜": "night,citylight,bokeh",
    "戀": "sunset,silhouette,two",
    "愛": "sunset,landscape,warmth",
    "婚": "wedding,rings,flowers",
    "離婚": "empty,bench,autumn",
    "分手": "lonely,road,fog",
    "告白": "letter,desk,candlelight",
    "曖昧": "twilight,path,lantern",
    "吵架": "storm,clouds,dramatic",
    "思念": "autumn,leaves,bench",
    "異鄉": "train,station,journey",
    "距離": "road,horizon,sunset",
    "秘密": "shadow,alley,mystery",
    "回憶": "vintage,polaroid,sepia",
    "街": "street,lantern,evening",
    "海": "ocean,wave,horizon",
    "酒": "wine,glass,candlelight",
    "學校": "campus,corridor,autumn",
    "公司": "building,skyline,dusk",
    "家": "window,curtain,sunlight",
    "夢": "sky,clouds,dreamy",
    "淚": "rain,glass,droplets",
    "笑": "sunshine,field,flowers",
    "孤獨": "silhouette,fog,solitude",
    "擁抱": "sunset,silhouette,embrace",
}

DEFAULT_QUERIES = [
    "silhouette,sunset,horizon",
    "rain,window,bokeh",
    "street,lantern,evening",
    "autumn,leaves,path",
    "twilight,sky,clouds",
    "ocean,horizon,calm",
]

_query_index = 0


def extract_search_query(title: str, content: str) -> str:
    """從標題和內容提取英文搜尋關鍵字。"""
    global _query_index
    text = title + " " + content[:500]

    matched = []
    for zh_key, en_query in KEYWORD_MAP.items():
        if zh_key in text:
            matched.append(en_query)
        if len(matched) >= 3:
            break

    if matched:
        return ",".join(matched)

    query = DEFAULT_QUERIES[_query_index % len(DEFAULT_QUERIES)]
    _query_index += 1
    return query


def _has_red_border(image_data: bytes) -> bool:
    """檢查圖片左右邊緣是否有 Picsum 的紅色填充邊框。"""
    try:
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(image_data))
        w, h = img.size
        for box in [(0, 0, 5, h), (w - 5, 0, w, h)]:
            pixels = list(img.crop(box).getdata())
            avg_r = sum(p[0] for p in pixels) / len(pixels)
            avg_g = sum(p[1] for p in pixels) / len(pixels)
            avg_b = sum(p[2] for p in pixels) / len(pixels)
            if avg_r > 150 and avg_r > avg_g * 2 and avg_r > avg_b * 2:
                return True
        return False
    except Exception:
        return False


def download_image(seed: str, max_retries: int = 5) -> bytes | None:
    """從 Picsum Photos 下載唯美風景照，seed 保證每篇不同。
    自動檢測並跳過有紅色邊框的圖片，最多重試 max_retries 次。"""
    for attempt in range(max_retries):
        current_seed = f"{seed}-v{attempt}" if attempt > 0 else seed
        url = f"https://picsum.photos/seed/{current_seed}/1200/630"
        try:
            resp = requests.get(url, timeout=20, allow_redirects=True)
            if resp.status_code != 200 or "image" not in resp.headers.get("content-type", ""):
                log.warning(f"圖片下載失敗: HTTP {resp.status_code} (seed: {current_seed})")
                continue
            if _has_red_border(resp.content):
                log.warning(f"圖片有紅色邊框，換一張 (seed: {current_seed})")
                continue
            log.info(f"圖片下載成功: {len(resp.content)} bytes (seed: {current_seed})")
            return resp.content
        except requests.RequestException as e:
            log.error(f"圖片下載錯誤: {e}")
            continue
    log.error(f"嘗試 {max_retries} 次仍無法取得無紅框圖片 (seed: {seed})")
    return None


def upload_to_wordpress(session: requests.Session, api_base: str,
                        image_data: bytes, filename: str) -> tuple[str, int] | None:
    """上傳圖片到 WordPress，回傳 (圖片 URL, media_id)。"""
    upload_headers = dict(session.headers)
    upload_headers["Content-Type"] = "image/jpeg"
    upload_headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    try:
        resp = requests.post(
            f"{api_base}/media",
            headers=upload_headers,
            data=image_data,
            timeout=30,
        )
        if resp.status_code in (200, 201):
            media = resp.json()
            img_url = media.get("source_url", "")
            media_id = media.get("id", 0)
            log.info(f"圖片已上傳: {img_url} (media_id={media_id})")
            return img_url, media_id
        log.error(f"圖片上傳失敗: HTTP {resp.status_code} — {resp.text[:200]}")
        return None
    except requests.RequestException as e:
        log.error(f"圖片上傳錯誤: {e}")
        return None


def get_image_html(title: str, content: str, session: requests.Session,
                   api_base: str, post_id: int) -> tuple[str, int]:
    """取得配圖並回傳 (HTML img tag, media_id)，供嵌入文章開頭。失敗時回傳 ("", 0)。"""
    image_data = download_image(seed=f"post{post_id}")
    if not image_data:
        return "", 0

    filename = re.sub(r"[^a-zA-Z0-9_]", "", re.sub(r"\s+", "_", title))[:30] or "cover"
    filename = f"{filename}_{post_id}.jpg"
    result = upload_to_wordpress(session, api_base, image_data, filename)
    if not result:
        return "", 0

    img_url, media_id = result
    html = (
        f'<figure class="wp-block-image size-large">'
        f'<img src="{img_url}" alt="{title}" style="width:100%;height:auto;" />'
        f'</figure>'
    )
    return html, media_id
