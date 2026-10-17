"""
wp_threads.py — Threads Graph API 客戶端（文章樹洞專用帳號）
發布純文字貼文（標題＋摘要＋WP連結）到 Threads，與喵小仙的 Threads 帳號完全獨立。

環境變數：
    THREADS_USER_ID: 此帳號的 Threads 使用者 ID
    THREADS_ACCESS_TOKEN: 此帳號的 long-lived OAuth2 token（60 天有效）
"""

import logging
import time

import requests

log = logging.getLogger("wp-poster.threads")

GRAPH_API = "https://graph.threads.net/v1.0"


class ThreadsPublisher:
    def __init__(self, user_id: str, access_token: str):
        self.user_id = user_id
        self.access_token = access_token
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """驗證 access token 有效，並回報帳號名稱。"""
        try:
            resp = self.session.get(
                f"{GRAPH_API}/{self.user_id}",
                params={"fields": "username", "access_token": self.access_token},
                timeout=15,
            )
            if resp.status_code == 200:
                username = resp.json().get("username", "")
                log.info(f"Threads 帳號連線成功: @{username}")
                return True
            log.error(f"Threads 認證失敗: HTTP {resp.status_code} — {resp.text}")
            return False
        except requests.RequestException as e:
            log.error(f"Threads 連線失敗: {e}")
            return False

    def post_text(self, text: str) -> str | None:
        """
        發布純文字貼文到 Threads。
        流程：建立 media container → 等待處理 → 發布。
        回傳: Threads post ID（成功）或 None（失敗）
        """
        try:
            create_resp = self.session.post(
                f"{GRAPH_API}/{self.user_id}/threads",
                data={
                    "media_type": "TEXT",
                    "text": text,
                    "access_token": self.access_token,
                },
                timeout=30,
            )
            if create_resp.status_code != 200:
                log.warning(f"Threads container 建立失敗: {create_resp.status_code} {create_resp.text}")
                return None

            container_id = create_resp.json().get("id")
            if not container_id:
                log.warning("Threads container 回應中沒有 ID")
                return None

            time.sleep(10)  # 讓 Meta 處理 container

            publish_resp = self.session.post(
                f"{GRAPH_API}/{self.user_id}/threads_publish",
                data={
                    "creation_id": container_id,
                    "access_token": self.access_token,
                },
                timeout=30,
            )
            if publish_resp.status_code != 200:
                log.warning(f"Threads 發布失敗: {publish_resp.status_code} {publish_resp.text}")
                return None

            post_id = publish_resp.json().get("id")
            log.info(f"Threads 發文成功: {post_id}")
            return post_id
        except requests.RequestException as e:
            log.warning(f"Threads 發文失敗: {e}")
            return None

    def check_token_expiry(self) -> int | None:
        """檢查 long-lived token 剩餘有效天數（Threads token 固定 60 天效期）。"""
        try:
            resp = requests.get(
                f"{GRAPH_API}/refresh_access_token",
                params={"grant_type": "th_refresh_token", "access_token": self.access_token},
                timeout=15,
            )
            if resp.status_code != 200:
                return None
            return None  # Threads API 不直接回傳到期時間，僅用於 refresh 測試
        except Exception as e:
            log.warning(f"檢查 Threads token 失敗: {e}")
            return None

    def refresh_token(self) -> str | None:
        """刷新 long-lived token（需在 60 天到期前呼叫，建議每 45-50 天跑一次）。"""
        try:
            resp = requests.get(
                f"{GRAPH_API}/refresh_access_token",
                params={"grant_type": "th_refresh_token", "access_token": self.access_token},
                timeout=15,
            )
            if resp.status_code != 200:
                log.error(f"Threads token 刷新失敗: {resp.text}")
                return None
            new_token = resp.json().get("access_token")
            log.info(f"Threads token 已刷新（{resp.json().get('expires_in')} 秒後到期）")
            return new_token
        except Exception as e:
            log.error(f"Threads token 刷新錯誤: {e}")
            return None
