"""
wp_poster.py — WordPress 自動發文機器人（主程式）
從 PTT 爬熱門文章 → Gemini AI 改寫成虛構小說連載 → 每 4 小時發布到 WordPress。

排程: 00:00 / 04:00 / 08:00 / 12:00 / 16:00 / 20:00（一天 6 篇）
"""

import os
import logging
import time
import traceback
from datetime import datetime, timedelta

from dotenv import load_dotenv

from wp_database import Database
from wp_scraper import PTTScraper
from wp_content import ContentGenerator
from wp_publisher import WordPressPublisher
from wp_facebook import FacebookPublisher

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("wp_poster.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("wp-poster")

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
load_dotenv()


class Config:
    # WordPress.com OAuth2
    WP_SITE_URL = os.getenv("WP_SITE_URL", "")
    WP_ACCESS_TOKEN = os.getenv("WP_ACCESS_TOKEN", "")
    WP_CLIENT_ID = os.getenv("WP_CLIENT_ID", "")
    WP_CLIENT_SECRET = os.getenv("WP_CLIENT_SECRET", "")
    WP_USERNAME = os.getenv("WP_USERNAME", "")
    WP_PASSWORD = os.getenv("WP_PASSWORD", "")
    WP_CATEGORY_ID = int(os.getenv("WP_CATEGORY_ID", "1"))
    WP_DEFAULT_TAGS = [
        t.strip() for t in os.getenv("WP_DEFAULT_TAGS", "PTT故事,連載小說,愛情故事").split(",")
    ]

    # PTT
    PTT_BOARDS = [
        b.strip() for b in os.getenv("PTT_BOARDS", "Boy-Girl,marriage,sex,WomenTalk,Hate").split(",")
    ]
    PTT_MIN_PUSH_COUNT = int(os.getenv("PTT_MIN_PUSH_COUNT", "10"))

    # 排程
    PUBLISH_HOURS = [0, 4, 8, 12, 16, 20]
    MIN_QUEUE_SIZE = int(os.getenv("MIN_QUEUE_SIZE", "10"))
    EPISODE_TARGET_CHARS = int(os.getenv("EPISODE_TARGET_CHARS", "800"))

    # Facebook（選填）
    FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN", "")
    FB_PAGE_ID = os.getenv("FB_PAGE_ID", "")
    FB_APP_ID = os.getenv("FB_APP_ID", "")
    FB_APP_SECRET = os.getenv("FB_APP_SECRET", "")

    # Discord 通知
    DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
    DISCORD_OWNER_ID = os.getenv("DISCORD_OWNER_ID", "")

    # Gemini（優先用 WP 專用 key，沒設就 fallback 共用 key）
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_WP") or os.getenv("GEMINI_API_KEY", "")

    @classmethod
    def validate(cls):
        missing = []
        if not cls.WP_SITE_URL:
            missing.append("WP_SITE_URL")
        # 需要 access_token 或 client_id+client_secret+username+password
        if not cls.WP_ACCESS_TOKEN:
            for var in ("WP_CLIENT_ID", "WP_CLIENT_SECRET", "WP_USERNAME", "WP_PASSWORD"):
                if not getattr(cls, var):
                    missing.append(var)
        if not cls.GEMINI_API_KEY:
            missing.append("GEMINI_API_KEY")
        if missing:
            raise RuntimeError(f"缺少必要環境變數: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
CHECK_INTERVAL = 600  # 主迴圈每 10 分鐘檢查一次


class WPPoster:
    def __init__(self):
        Config.validate()
        self.db = Database()
        self.scraper = PTTScraper(min_push=Config.PTT_MIN_PUSH_COUNT)
        self.generator = ContentGenerator(api_key=Config.GEMINI_API_KEY)
        self.publisher = WordPressPublisher(
            site_url=Config.WP_SITE_URL,
            access_token=Config.WP_ACCESS_TOKEN,
            client_id=Config.WP_CLIENT_ID,
            client_secret=Config.WP_CLIENT_SECRET,
            username=Config.WP_USERNAME,
            password=Config.WP_PASSWORD,
        )
        self.fb: FacebookPublisher | None = None
        if Config.FB_PAGE_ACCESS_TOKEN and Config.FB_PAGE_ID:
            self.fb = FacebookPublisher(
                page_access_token=Config.FB_PAGE_ACCESS_TOKEN,
                page_id=Config.FB_PAGE_ID,
                app_id=Config.FB_APP_ID,
                app_secret=Config.FB_APP_SECRET,
            )
        self._last_scrape_date: str | None = None
        self._last_token_check_date: str | None = None

    def _get_last_publish_time(self) -> datetime | None:
        """從資料庫查詢最後一次發布時間，避免重啟時重複發文。"""
        row = self.db._conn.execute(
            "SELECT published_at FROM episodes WHERE status = 'published' "
            "ORDER BY published_at DESC LIMIT 1"
        ).fetchone()
        if row and row["published_at"]:
            try:
                return datetime.strptime(row["published_at"], "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return None
        return None

    def run(self):
        """主迴圈：每 10 分鐘檢查是否該發文或補充草稿。"""
        log.info("=== WP Poster 啟動 ===")
        log.info(f"看板: {Config.PTT_BOARDS}")
        log.info(f"發文時段: {Config.PUBLISH_HOURS}")
        log.info(f"草稿佇列下限: {Config.MIN_QUEUE_SIZE}")
        log.info(f"目標字數/集: {Config.EPISODE_TARGET_CHARS}")

        if not self.publisher.test_connection():
            log.error("WordPress 連線失敗，請檢查設定")
            return

        if self.fb:
            if self.fb.test_connection():
                log.info("Facebook 同步已啟用")
                self._check_fb_token()
            else:
                log.warning("Facebook 連線失敗，將不會同步到 FB")
                self.fb = None
        else:
            log.info("未設定 Facebook 憑證，FB 同步停用")

        draft_count = self.db.count_drafts()
        wp_draft_count = self.db.count_wp_drafts()
        log.info(f"目前佇列: {draft_count} 篇待同步草稿, {wp_draft_count} 篇 WP 草稿")

        # 啟動時同步既有草稿到 WP
        self.sync_drafts_to_wp()

        while True:
            try:
                now = datetime.now()

                # 檢查是否該發文
                if self._should_publish(now):
                    self.publish_next_episode()

                # 每日檢查 FB token 過期
                if self.fb and self._should_check_token(now):
                    self._check_fb_token()
                    self._last_token_check_date = now.strftime("%Y-%m-%d")

                # 檢查是否該補充草稿
                if self._should_scrape(now):
                    self.scrape_and_generate()
                    self._last_scrape_date = now.strftime("%Y-%m-%d")
                    # 爬取後同步新草稿到 WP
                    self.sync_drafts_to_wp()

            except Exception:
                log.error(f"主迴圈異常:\n{traceback.format_exc()}")

            time.sleep(CHECK_INTERVAL)

    def _should_publish(self, now: datetime) -> bool:
        """判斷目前是否為發文時段（查資料庫防止重啟後重複發文）。"""
        if now.hour not in Config.PUBLISH_HOURS:
            return False
        last = self._get_last_publish_time()
        if last and last.date() == now.date() and last.hour == now.hour:
            return False
        return True

    def _should_scrape(self, now: datetime) -> bool:
        """草稿佇列不足且今日尚未爬取時觸發。"""
        today = now.strftime("%Y-%m-%d")
        if self._last_scrape_date == today:
            return False
        return self.db.count_drafts() < Config.MIN_QUEUE_SIZE

    def scrape_and_generate(self):
        """爬取 PTT → AI 改寫 → 存入草稿佇列。"""
        log.info("--- 開始爬取 + 產生新內容 ---")
        total_new = 0

        for board in Config.PTT_BOARDS:
            try:
                posts = self.scraper.scrape_board(board, pages=3)
            except Exception as e:
                log.error(f"爬取 {board} 失敗: {e}")
                continue

            for post in posts:
                # 嘗試加入資料庫（去重）
                source_id = self.db.add_source(
                    url=post["url"],
                    board=post["board"],
                    title=post["title"],
                    author=post["author"],
                    content="",  # 先佔位，稍後填入
                    push_count=post["push_count"],
                )
                if source_id is None:
                    continue  # 已存在

                total_new += 1
                log.info(f"新來源: [{board}] {post['title']} (推{post['push_count']})")

        log.info(f"本次新增 {total_new} 篇來源")

        # 從未處理的來源中取出，產生小說
        generated = 0
        while True:
            source = self.db.get_unused_source()
            if not source:
                break

            try:
                # 取得完整文章內容
                content = self.scraper.fetch_article(source["url"])
                if not content or len(content) < 200:
                    log.warning(f"文章內容過短，跳過: {source['title']}")
                    self.db.mark_source_used(source["id"])
                    continue

                # AI 改寫
                result = self.generator.rewrite_story(
                    ptt_title=source["title"],
                    ptt_content=content,
                    target_chars=Config.EPISODE_TARGET_CHARS,
                )

                if not result["episodes"]:
                    log.warning(f"AI 改寫無結果，跳過: {source['title']}")
                    self.db.mark_source_used(source["id"])
                    continue

                # 存入草稿
                self.db.add_episodes(
                    source_id=source["id"],
                    series_title=result["series_title"],
                    episodes=result["episodes"],
                )
                self.db.mark_source_used(source["id"])
                generated += len(result["episodes"])

                log.info(
                    f"已產生: {result['series_title']} — "
                    f"{len(result['episodes'])} 集"
                )

            except Exception as e:
                log.error(f"處理來源 {source['title']} 失敗: {e}")
                self.db.mark_source_used(source["id"])

            # 草稿夠了就停
            if self.db.count_drafts() >= Config.MIN_QUEUE_SIZE:
                break

        log.info(f"本次產生 {generated} 集草稿，佇列現有 {self.db.count_drafts()} 篇")

    def _should_check_token(self, now: datetime) -> bool:
        """每天檢查一次 FB token。"""
        today = now.strftime("%Y-%m-%d")
        return self._last_token_check_date != today

    def _check_fb_token(self):
        """檢查 FB token 過期時間，剩餘 14 天內發出通知。"""
        if not self.fb:
            return
        remaining = self.fb.check_token_expiry()
        if remaining is None:
            return
        if remaining <= 14:
            msg = (
                f"⚠️ Facebook Page Token 將在 {remaining} 天後過期！\n"
                f"請到 Graph API Explorer 重新產生 Page Token 並更新 .env"
            )
            log.warning(msg)
            self._send_discord_dm(msg)
        elif remaining <= 30:
            log.info(f"FB Token 剩餘 {remaining} 天")

    def _send_discord_dm(self, message: str):
        """透過 Discord Bot 發送私訊通知。"""
        if not Config.DISCORD_TOKEN or not Config.DISCORD_OWNER_ID:
            return
        try:
            headers = {"Authorization": f"Bot {Config.DISCORD_TOKEN}"}
            # 建立 DM channel
            resp = requests.post(
                "https://discord.com/api/v10/users/@me/channels",
                json={"recipient_id": Config.DISCORD_OWNER_ID},
                headers=headers,
                timeout=15,
            )
            resp.raise_for_status()
            channel_id = resp.json()["id"]
            # 發送訊息
            requests.post(
                f"https://discord.com/api/v10/channels/{channel_id}/messages",
                json={"content": message},
                headers=headers,
                timeout=15,
            )
            log.info("已透過 Discord 發送通知")
        except Exception as e:
            log.warning(f"Discord 通知發送失敗: {e}")

    def sync_drafts_to_wp(self):
        """將所有 SQLite draft 同步到 WordPress 草稿。"""
        pending = self.db.get_all_pending_drafts()
        if not pending:
            log.info("沒有待同步的草稿")
            return

        log.info(f"開始同步 {len(pending)} 篇草稿到 WordPress...")

        for i, draft in enumerate(pending):
            series = draft['series_title'].strip("《》")
            title = f"{series}｜EP.{draft['episode_num']}"

            # 取得下一集（用於預告）
            next_title = ""
            next_teaser = ""
            # 從 pending 清單中找下一集，或從 DB 撈
            next_draft = pending[i + 1] if i + 1 < len(pending) else None
            if next_draft:
                nxt_series = next_draft['series_title'].strip("《》")
                next_title = f"{nxt_series}｜EP.{next_draft['episode_num']}"
                try:
                    next_teaser = self.generator.generate_teaser(
                        current_ep_content=draft["content"],
                        next_ep_content=next_draft["content"],
                    )
                    log.info(f"下集預告已產生: {next_teaser[:50]}...")
                except Exception as e:
                    log.warning(f"產生預告失敗，使用摘要: {e}")
                    excerpt = next_draft["content"][:80].rsplit("，", 1)[0]
                    next_teaser = excerpt + "⋯⋯"

            # 產生 FB 摘要
            fb_teaser = ""
            if self.fb:
                try:
                    fb_teaser = self.generator.generate_fb_teaser(draft["content"])
                    log.info(f"FB 摘要已產生: {fb_teaser[:50]}...")
                except Exception as e:
                    log.warning(f"產生 FB 摘要失敗: {e}")

            try:
                wp_post_id = self.publisher.create_draft(
                    title=title,
                    content=draft["content"],
                    category_id=Config.WP_CATEGORY_ID,
                    tags=Config.WP_DEFAULT_TAGS,
                    next_episode_title=next_title,
                    next_episode_teaser=next_teaser,
                )
                self.db.mark_wp_draft(draft["id"], wp_post_id, fb_teaser)
                log.info(f"草稿已同步到 WP: {title} (WP ID={wp_post_id})")
            except Exception as e:
                log.error(f"同步草稿到 WP 失敗: {title} — {e}")

            # 每集之間延遲 15 秒（避免 Gemini rate limit）
            if i < len(pending) - 1:
                time.sleep(15)

        log.info(f"草稿同步完成，WP 草稿佇列: {self.db.count_wp_drafts()} 篇")

    def publish_next_episode(self):
        """從 WP 草稿中取出下一篇發布。"""
        draft = self.db.get_next_wp_draft()
        if not draft:
            log.warning("WP 草稿佇列為空，無法發文")
            return

        series = draft['series_title'].strip("《》")
        title = f"{series}｜EP.{draft['episode_num']}"
        wp_post_id = draft['wp_post_id']
        log.info(f"準備發布 WP 草稿: {title} (WP ID={wp_post_id})")

        try:
            _, permalink = self.publisher.publish_draft(wp_post_id)
            self.db.mark_published(draft["id"], wp_post_id)
            log.info(f"發布成功: {title} (WP ID={wp_post_id})")

            # Facebook 同步
            if self.fb:
                post_url = permalink or f"https://mnnuyenn.art.blog/?p={wp_post_id}"
                teaser = draft.get("fb_teaser") or ""
                if not teaser:
                    teaser = draft["content"][:100].rsplit("，", 1)[0] + "⋯⋯"
                fb_message = f"{title}\n{teaser}\n{post_url}"
                self.fb.share_post(post_url, fb_message)

        except Exception as e:
            error_msg = str(e)
            self.db.mark_failed(draft["id"], error_msg)
            log.error(f"發布失敗: {title} — {error_msg}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    poster = WPPoster()
    poster.run()
