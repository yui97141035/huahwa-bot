"""
wp_add_endings.py — 為所有未完結系列補上完結篇（草稿 + 已發布）
一次跑一批（預設 5 個），避免 Gemini 429。可重複執行，已補過的會跳過。

用法:
  python3 wp_add_endings.py [batch_size] [--published]
    batch_size: 每次處理幾個系列（預設 5）
    --published: 也處理已發布但沒結局的系列（產生完結篇並直接發布）
"""

import os
import sys
import time
import logging
import sqlite3
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wp_poster.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("wp-endings")

from wp_content import ContentGenerator
from wp_publisher import WordPressPublisher

DB_PATH = "wp_poster.db"
ENDING_MARKERS = ["（完）", "（全文完）", "—完—", "＊完結＊"]
DELAY_BETWEEN = 20  # 每個系列之間等待秒數


def get_unfinished_series(conn, include_published=False) -> list[dict]:
    """找出所有沒有完結篇的系列"""
    if include_published:
        status_filter = "('wp_draft', 'draft', 'published')"
    else:
        status_filter = "('wp_draft', 'draft')"

    rows = conn.execute(f"""
        SELECT DISTINCT series_title
        FROM episodes
        WHERE status IN {status_filter}
        AND series_title NOT IN (
            SELECT DISTINCT series_title FROM episodes
            WHERE content LIKE '%（完）%' OR content LIKE '%—完—%'
               OR content LIKE '%（全文完）%' OR content LIKE '%＊完結＊%'
        )
        ORDER BY id ASC
    """).fetchall()

    series_list = []
    for row in rows:
        title = row["series_title"]
        eps = conn.execute("""
            SELECT id, episode_num, content, status, source_id
            FROM episodes WHERE series_title = ?
            ORDER BY episode_num ASC
        """, (title,)).fetchall()
        max_ep = max(e["episode_num"] for e in eps)
        # 判斷這個系列是否全部已發布（決定完結篇要直接發布還是存草稿）
        all_published = all(e["status"] == "published" for e in eps)
        series_list.append({
            "series_title": title,
            "episodes": eps,
            "max_ep": max_ep,
            "source_id": eps[0]["source_id"],
            "all_published": all_published,
        })
    return series_list


def main():
    # 解析參數
    args = sys.argv[1:]
    include_published = "--published" in args
    args = [a for a in args if a != "--published"]
    batch_size = int(args[0]) if args else 5

    api_key = os.getenv("GEMINI_API_KEY_WP") or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        log.error("GEMINI_API_KEY 未設定")
        return

    generator = ContentGenerator(api_key=api_key)

    # WordPress publisher (access_token 或 OAuth2 credentials 皆可)
    wp = None
    wp_site = os.getenv("WP_SITE_URL", "")
    wp_token = os.getenv("WP_ACCESS_TOKEN", "")
    wp_client_id = os.getenv("WP_CLIENT_ID", "")
    wp_has_oauth = all([wp_client_id, os.getenv("WP_CLIENT_SECRET", ""),
                        os.getenv("WP_USERNAME", ""), os.getenv("WP_PASSWORD", "")])
    if wp_site and (wp_token or wp_has_oauth):
        wp = WordPressPublisher(
            site_url=wp_site, access_token=wp_token,
            client_id=wp_client_id,
            client_secret=os.getenv("WP_CLIENT_SECRET", ""),
            username=os.getenv("WP_USERNAME", ""),
            password=os.getenv("WP_PASSWORD", ""),
        )
        log.info("WordPress publisher 已初始化" + (" (OAuth2)" if not wp_token else ""))

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    series_list = get_unfinished_series(conn, include_published=include_published)
    mode = "草稿+已發布" if include_published else "僅草稿"
    log.info(f"模式: {mode} — 找到 {len(series_list)} 個未完結系列，本次處理 {min(batch_size, len(series_list))} 個")

    processed = 0
    for series in series_list[:batch_size]:
        title = series["series_title"]
        eps = series["episodes"]
        next_ep_num = series["max_ep"] + 1
        should_publish = series["all_published"]

        action = "產生+發布" if should_publish else "產生+存草稿"
        log.info(f"--- [{action}] {title} (目前 {len(eps)} 集，補第 {next_ep_num} 集完結篇) ---")

        # 收集所有集數內容
        all_content = [e["content"] for e in eps]

        try:
            ending = generator.generate_ending(title, all_content)
        except Exception as e:
            log.error(f"產生完結篇失敗: {title} — {e}")
            if "429" in str(e) or "quota" in str(e).lower():
                log.warning("配額用盡，停止本批次")
                break
            continue

        if not ending or len(ending) < 400:
            log.warning(f"完結篇內容過短（{len(ending) if ending else 0} 字），跳過: {title}")
            continue

        # 確保結尾有完結標記（供去重偵測用）
        if not any(m in ending[-50:] for m in ENDING_MARKERS):
            ending = ending.rstrip() + "\n\n（完）"

        # 寫入 DB
        source_id = series["source_id"]
        initial_status = "draft"
        conn.execute(
            "INSERT INTO episodes (source_id, series_title, episode_num, content, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (source_id, title, next_ep_num, ending, initial_status),
        )
        conn.commit()
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        log.info(f"完結篇已存入 DB: {title} EP.{next_ep_num} (id={new_id}, {len(ending)} 字)")

        # 同步到 WP
        if wp:
            try:
                series_short = title.strip("《》")
                wp_title = f"{series_short}｜EP.{next_ep_num}（完結）"

                # 產生 FB teaser
                fb_teaser = ""
                try:
                    fb_teaser = generator.generate_fb_teaser(ending)
                except Exception:
                    pass

                if should_publish:
                    # 已發布系列 → 直接發布完結篇
                    wp_post_id = wp.publish_post(
                        title=wp_title,
                        content=ending,
                        category_id=int(os.getenv("WP_CATEGORY_ID", "1")),
                        tags=[t.strip() for t in os.getenv(
                            "WP_DEFAULT_TAGS", "PTT故事,連載小說,完結篇"
                        ).split(",")],
                    )
                    conn.execute(
                        "UPDATE episodes SET status = 'published', wp_post_id = ?, "
                        "fb_teaser = ?, published_at = datetime('now', 'localtime') WHERE id = ?",
                        (wp_post_id, fb_teaser, new_id),
                    )
                    conn.commit()
                    log.info(f"完結篇已發布: {wp_title} (WP ID={wp_post_id})")
                else:
                    # 草稿系列 → 存為 WP 草稿
                    wp_post_id = wp.create_draft(
                        title=wp_title,
                        content=ending,
                        category_id=int(os.getenv("WP_CATEGORY_ID", "1")),
                        tags=[t.strip() for t in os.getenv(
                            "WP_DEFAULT_TAGS", "PTT故事,連載小說,完結篇"
                        ).split(",")],
                    )
                    conn.execute(
                        "UPDATE episodes SET status = 'wp_draft', wp_post_id = ?, fb_teaser = ? WHERE id = ?",
                        (wp_post_id, fb_teaser, new_id),
                    )
                    conn.commit()
                    log.info(f"完結篇已存為 WP 草稿: {wp_title} (WP ID={wp_post_id})")
            except Exception as e:
                log.error(f"同步 WP 失敗: {title} — {e}")

        processed += 1
        if processed < min(batch_size, len(series_list)):
            log.info(f"等待 {DELAY_BETWEEN} 秒後處理下一個...")
            time.sleep(DELAY_BETWEEN)

    conn.close()
    remaining = len(series_list) - processed
    log.info(f"=== 本批次完成: 處理 {processed} 個系列，剩餘 {remaining} 個未完結 ===")
    if remaining > 0:
        log.info(f"可再次執行: python3 wp_add_endings.py {batch_size} {'--published' if include_published else ''}")


if __name__ == "__main__":
    main()
