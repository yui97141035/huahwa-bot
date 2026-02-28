"""
wp_facebook.py — Facebook Graph API 客戶端
發布 WordPress 文章連結到 Facebook 粉專。
"""

import logging
import requests

log = logging.getLogger("wp-poster.facebook")

GRAPH_API = "https://graph.facebook.com/v21.0"


class FacebookPublisher:
    def __init__(self, page_access_token: str, page_id: str,
                 app_id: str = "", app_secret: str = ""):
        self.page_id = page_id
        self.app_id = app_id
        self.app_secret = app_secret
        self.session = requests.Session()
        self.session.params = {"access_token": page_access_token}

    def test_connection(self) -> bool:
        """驗證 Page Access Token 有效。"""
        try:
            resp = self.session.get(
                f"{GRAPH_API}/{self.page_id}",
                params={"fields": "name,id"},
                timeout=15,
            )
            if resp.status_code == 200:
                name = resp.json().get("name", "")
                log.info(f"Facebook 粉專連線成功: {name}")
                return True
            log.error(f"Facebook 認證失敗: HTTP {resp.status_code} — {resp.text}")
            return False
        except requests.RequestException as e:
            log.error(f"Facebook 連線失敗: {e}")
            return False

    def share_post(self, post_url: str, message: str) -> int | None:
        """
        發布連結到粉專。
        回傳: FB post ID (成功) 或 None (失敗)
        """
        try:
            resp = self.session.post(
                f"{GRAPH_API}/{self.page_id}/feed",
                data={"message": message, "link": post_url},
                timeout=30,
            )
            resp.raise_for_status()
            fb_post_id = resp.json().get("id")
            log.info(f"FB 發文成功: {fb_post_id}")
            return fb_post_id
        except requests.RequestException as e:
            log.warning(f"FB 發文失敗: {e}")
            return None

    def check_token_expiry(self) -> int | None:
        """
        檢查 token 剩餘有效天數。
        回傳: 剩餘天數（永久 token 回傳 None）
        """
        if not self.app_id or not self.app_secret:
            return None
        try:
            resp = requests.get(
                f"{GRAPH_API}/debug_token",
                params={
                    "input_token": self.session.params["access_token"],
                    "access_token": f"{self.app_id}|{self.app_secret}",
                },
                timeout=15,
            )
            data = resp.json().get("data", {})
            expires_at = data.get("expires_at", 0)
            if expires_at == 0:
                log.info("FB Token: 永不過期")
                return None
            import time
            remaining = int((expires_at - time.time()) / 86400)
            log.info(f"FB Token: 剩餘 {remaining} 天")
            return remaining
        except Exception as e:
            log.warning(f"檢查 FB Token 過期失敗: {e}")
            return None
