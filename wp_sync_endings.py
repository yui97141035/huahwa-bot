"""
wp_sync_endings.py — 將 DB 中已生成但未同步到 WP 的完結篇上傳
處理 wp_add_endings.py 在沒有 WP publisher 時產生的 draft 狀態結局

用法:
  python3 wp_sync_endings.py [--dry-run]
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
log = logging.getLogger("wp-sync-endings")

from wp_publisher import WordPressPublisher
from wp_content import ContentGenerator

DB_PATH = "wp_poster.db"
ENDING_MARKERS = ["（完）", "（全文完）", "—完—", "＊完結＊"]
DELAY_BETWEEN = 5  # WP API 間隔


def main():
    dry_run = "--dry-run" in sys.argv

    # 初始化 WP publisher
    wp_site = os.getenv("WP_SITE_URL", "")
    wp_token = os.getenv("WP_ACCESS_TOKEN", "")
    wp_client_id = os.getenv("WP_CLIENT_ID", "")
    wp_has_oauth = all([wp_client_id, os.getenv("WP_CLIENT_SECRET", ""),
                        os.getenv("WP_USERNAME", ""), os.getenv("WP_PASSWORD", "")])
    if not wp_site or not (wp_token or wp_has_oauth):
        log.error("WordPress credentials 不足，無法同步")
        return

    wp = WordPressPublisher(
        site_url=wp_site, access_token=wp_token,
        client_id=wp_client_id,
        client_secret=os.getenv("WP_CLIENT_SECRET", ""),
        username=os.getenv("WP_USERNAME", ""),
        password=os.getenv("WP_PASSWORD", ""),
    )

    # 初始化 content generator (for FB teaser)
    api_key = os.getenv("GEMINI_API_KEY_WP") or os.getenv("GEMINI_API_KEY", "")
    generator = ContentGenerator(api_key=api_key) if api_key else None

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 找出所有完結篇 + 沒有 wp_post_id 的記錄
    rows = conn.execute("""
        SELECT e.id, e.series_title, e.episode_num, e.content, e.status,
               e.wp_post_id
        FROM episodes e
        WHERE (e.content LIKE '%（完）%' OR e.content LIKE '%—完—%'
               OR e.content LIKE '%（全文完）%' OR e.content LIKE '%＊完結＊%')
          AND e.wp_post_id IS NULL
        ORDER BY e.id ASC
    """).fetchall()

    log.info(f"找到 {len(rows)} 個未同步完結篇" + (" (dry-run)" if dry_run else ""))

    # 判斷每個完結篇對應的系列是否全部已發布
    synced = 0
    for row in rows:
        series_title = row["series_title"]
        ep_num = row["episode_num"]

        # 查看同系列其他集數是否已發布
        other_eps = conn.execute("""
            SELECT status FROM episodes
            WHERE series_title = ? AND id != ?
        """, (series_title, row["id"])).fetchall()
        all_published = all(e["status"] == "published" for e in other_eps) if other_eps else False

        series_short = series_title.strip("《》")
        wp_title = f"{series_short}｜EP.{ep_num}（完結）"

        if all_published:
            action = "發布"
        else:
            action = "存草稿"

        log.info(f"  [{action}] {wp_title} (id={row['id']}, {len(row['content'])}字)")

        if dry_run:
            synced += 1
            continue

        try:
            # 產生 FB teaser
            fb_teaser = ""
            if generator:
                try:
                    fb_teaser = generator.generate_fb_teaser(row["content"])
                except Exception:
                    pass

            category_id = int(os.getenv("WP_CATEGORY_ID", "1"))
            tags = [t.strip() for t in os.getenv(
                "WP_DEFAULT_TAGS", "PTT故事,連載小說,完結篇"
            ).split(",")]

            if all_published:
                wp_post_id = wp.publish_post(
                    title=wp_title, content=row["content"],
                    category_id=category_id, tags=tags,
                )
                conn.execute(
                    "UPDATE episodes SET status = 'published', wp_post_id = ?, "
                    "fb_teaser = ?, published_at = datetime('now', 'localtime') WHERE id = ?",
                    (wp_post_id, fb_teaser, row["id"]),
                )
            else:
                wp_post_id = wp.create_draft(
                    title=wp_title, content=row["content"],
                    category_id=category_id, tags=tags,
                )
                conn.execute(
                    "UPDATE episodes SET status = 'wp_draft', wp_post_id = ? , fb_teaser = ? WHERE id = ?",
                    (wp_post_id, fb_teaser, row["id"]),
                )

            conn.commit()
            log.info(f"    ✓ WP ID={wp_post_id}")
            synced += 1

        except Exception as e:
            log.error(f"    ✗ 同步失敗: {e}")

        time.sleep(DELAY_BETWEEN)

    conn.close()
    log.info(f"=== 同步完成: {synced}/{len(rows)} 個完結篇 ===")


if __name__ == "__main__":
    main()
