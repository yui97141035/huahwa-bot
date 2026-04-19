"""
wp_database.py — SQLite 草稿管理
管理 PTT 來源文章與小說連載集數的佇列。
"""

import sqlite3
import logging
import time as _time

log = logging.getLogger("wp-poster.db")

DB_PATH = "wp_poster.db"


class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                url        TEXT UNIQUE NOT NULL,
                board      TEXT NOT NULL,
                title      TEXT NOT NULL,
                author     TEXT NOT NULL DEFAULT '',
                content    TEXT NOT NULL DEFAULT '',
                push_count INTEGER NOT NULL DEFAULT 0,
                used       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id     INTEGER NOT NULL REFERENCES sources(id),
                series_title  TEXT NOT NULL,
                episode_num   INTEGER NOT NULL,
                content       TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'draft',
                wp_post_id    INTEGER,
                error_msg     TEXT,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                published_at  TEXT
            )
        """)
        self._conn.commit()
        log.info(f"資料庫已初始化: {self.db_path}")

    # ── sources ──────────────────────────────────────────────

    def add_source(self, url: str, board: str, title: str,
                   author: str, content: str, push_count: int) -> int | None:
        """新增來源。若 URL 已存在回傳 None（去重）。"""
        try:
            cur = self._conn.execute(
                "INSERT INTO sources (url, board, title, author, content, push_count) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (url, board, title, author, content, push_count),
            )
            self._conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

    def get_unused_source(self) -> dict | None:
        """取得一篇尚未處理的來源文章。"""
        row = self._conn.execute(
            "SELECT * FROM sources WHERE used = 0 ORDER BY push_count DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def mark_source_used(self, source_id: int):
        self._conn.execute("UPDATE sources SET used = 1 WHERE id = ?", (source_id,))
        self._conn.commit()

    # ── episodes ─────────────────────────────────────────────

    def add_episodes(self, source_id: int, series_title: str, episodes: list[str]):
        """批次新增多集草稿。"""
        for i, ep_content in enumerate(episodes, start=1):
            self._conn.execute(
                "INSERT INTO episodes (source_id, series_title, episode_num, content) "
                "VALUES (?, ?, ?, ?)",
                (source_id, series_title, i, ep_content),
            )
        self._conn.commit()
        log.info(f"已新增 {len(episodes)} 集草稿 — {series_title}")

    def get_next_drafts(self, n: int = 2) -> list[dict]:
        """取得接下來 N 篇待發布的草稿（按建立順序）。"""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE status = 'draft' "
            "ORDER BY id ASC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_next_draft(self) -> dict | None:
        """取得下一篇待發布的草稿。"""
        drafts = self.get_next_drafts(1)
        return drafts[0] if drafts else None

    def count_drafts(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM episodes WHERE status = 'draft'"
        ).fetchone()
        return row["cnt"]

    def _retry_execute(self, sql: str, params: tuple = (), retries: int = 3):
        """執行 SQL 並在 database locked 時重試。"""
        for attempt in range(retries):
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e) and attempt < retries - 1:
                    log.warning(f"SQLite locked, 重試 {attempt+1}/{retries}...")
                    _time.sleep(1 * (attempt + 1))
                    continue
                raise

    def mark_published(self, episode_id: int, wp_post_id: int):
        self._retry_execute(
            "UPDATE episodes SET status = 'published', wp_post_id = ?, "
            "published_at = datetime('now', 'localtime') WHERE id = ?",
            (wp_post_id, episode_id),
        )

    def mark_wp_draft(self, episode_id: int, wp_post_id: int, fb_teaser: str = ""):
        """標記為已存到 WordPress 草稿，等待排程發布。"""
        self._ensure_fb_teaser_column()
        self._retry_execute(
            "UPDATE episodes SET status = 'wp_draft', wp_post_id = ?, fb_teaser = ? WHERE id = ?",
            (wp_post_id, fb_teaser, episode_id),
        )

    def _ensure_fb_teaser_column(self):
        """確保 fb_teaser 欄位存在（向後相容）。"""
        try:
            self._conn.execute("SELECT fb_teaser FROM episodes LIMIT 0")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE episodes ADD COLUMN fb_teaser TEXT DEFAULT ''")
            self._conn.commit()
            log.info("已新增 fb_teaser 欄位")

    def get_next_wp_draft(self) -> dict | None:
        """取得下一篇 wp_draft 狀態的集數。"""
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE status = 'wp_draft' "
            "ORDER BY id ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def get_next_wp_drafts(self, n: int = 2) -> list[dict]:
        """取得接下來 N 篇 wp_draft 狀態的集數。"""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE status = 'wp_draft' "
            "ORDER BY id ASC LIMIT ?", (n,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_pending_drafts(self) -> list[dict]:
        """取得所有 status='draft' 的集數（尚未存到 WP 草稿）。"""
        rows = self._conn.execute(
            "SELECT * FROM episodes WHERE status = 'draft' ORDER BY id ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_wp_drafts(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM episodes WHERE status = 'wp_draft'"
        ).fetchone()
        return row["cnt"]

    def mark_failed(self, episode_id: int, error_msg: str):
        self._retry_execute(
            "UPDATE episodes SET status = 'failed', error_msg = ? WHERE id = ?",
            (error_msg, episode_id),
        )

    # ── video pipeline ────────────────────────────────────────

    def _ensure_video_columns(self):
        """確保 video 相關欄位存在（向後相容）。"""
        for col, default in [
            ("video_status", "'pending'"),
            ("video_path", "''"),
            ("youtube_id", "''"),
        ]:
            try:
                self._conn.execute(f"SELECT {col} FROM episodes LIMIT 0")
            except sqlite3.OperationalError:
                self._conn.execute(
                    f"ALTER TABLE episodes ADD COLUMN {col} TEXT DEFAULT {default}"
                )
                self._conn.commit()
                log.info(f"已新增 {col} 欄位")

    def get_next_video_pending(self) -> dict | None:
        """取得下一篇已發布但未製片的文章。"""
        self._ensure_video_columns()
        row = self._conn.execute(
            "SELECT * FROM episodes WHERE status = 'published' "
            "AND video_status = 'pending' ORDER BY id ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def mark_video_processing(self, episode_id: int):
        self._ensure_video_columns()
        self._retry_execute(
            "UPDATE episodes SET video_status = 'processing' WHERE id = ?",
            (episode_id,),
        )

    def mark_video_uploaded(self, episode_id: int, video_path: str, youtube_id: str):
        self._ensure_video_columns()
        self._retry_execute(
            "UPDATE episodes SET video_status = 'uploaded', "
            "video_path = ?, youtube_id = ? WHERE id = ?",
            (video_path, youtube_id, episode_id),
        )

    def mark_video_failed(self, episode_id: int):
        self._ensure_video_columns()
        self._retry_execute(
            "UPDATE episodes SET video_status = 'failed' WHERE id = ?",
            (episode_id,),
        )
