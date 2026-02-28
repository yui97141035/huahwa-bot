"""
wp_publisher.py — WordPress REST API 客戶端
支援 WordPress.com (OAuth2 Bearer Token) 發布文章。
圖片直接嵌入文章內容開頭，結尾附下集預告。
"""

import logging
import requests
from wp_images import get_image_html

log = logging.getLogger("wp-poster.publisher")

WPCOM_API = "https://public-api.wordpress.com"


class WordPressPublisher:
    def __init__(self, site_url: str, access_token: str,
                 client_id: str = "", client_secret: str = "",
                 username: str = "", password: str = ""):
        site_host = site_url.replace("https://", "").replace("http://", "").rstrip("/")
        self.api_base = f"{WPCOM_API}/wp/v2/sites/{site_host}"
        self.session = requests.Session()
        self._tag_cache: dict[str, int] = {}

        if access_token:
            token = access_token
        else:
            token = self._obtain_token(client_id, client_secret, username, password)

        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    @staticmethod
    def _obtain_token(client_id: str, client_secret: str,
                      username: str, password: str) -> str:
        resp = requests.post(f"{WPCOM_API}/oauth2/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "password",
            "username": username,
            "password": password,
        }, timeout=15)
        resp.raise_for_status()
        token = resp.json()["access_token"]
        log.info("已透過 OAuth2 取得 access token")
        return token

    def test_connection(self) -> bool:
        try:
            resp = self.session.get(f"{self.api_base}/posts?per_page=1", timeout=15)
            if resp.status_code == 200:
                log.info("WordPress.com 連線成功")
                return True
            log.error(f"WordPress.com 認證失敗: HTTP {resp.status_code}")
            return False
        except requests.RequestException as e:
            log.error(f"WordPress.com 連線失敗: {e}")
            return False

    def publish_post(self, title: str, content: str,
                     category_id: int | None = None,
                     tags: list[str] | None = None,
                     next_episode_title: str = "",
                     next_episode_teaser: str = "") -> int:
        """
        發布文章到 WordPress。
        圖片嵌入內容開頭，下集預告附在結尾。
        回傳: wp_post_id
        """
        html_content = self._text_to_html(content)

        # 先建立草稿取得 post_id（供圖片命名用）
        payload = {
            "title": title,
            "content": html_content,
            "status": "draft",
        }
        if category_id:
            payload["categories"] = [category_id]
        if tags:
            tag_ids = [self._get_or_create_tag(t) for t in tags]
            tag_ids = [tid for tid in tag_ids if tid]
            if tag_ids:
                payload["tags"] = tag_ids

        resp = self.session.post(
            f"{self.api_base}/posts", json=payload, timeout=30,
        )
        resp.raise_for_status()
        wp_id = resp.json()["id"]

        # 取得配圖 HTML + media_id（用於 featured_media）
        img_html = ""
        media_id = 0
        try:
            img_html, media_id = get_image_html(title, content, self.session, self.api_base, wp_id)
        except Exception as e:
            log.warning(f"配圖失敗（不影響發文）: {e}")

        # 組合最終內容：圖片 + 本文 + 下集預告
        final_content = ""
        if img_html:
            final_content += img_html + "\n\n"
        final_content += html_content
        if next_episode_teaser:
            final_content += self._build_teaser(next_episode_title, next_episode_teaser)

        # 更新內容並發布（含精選圖片）
        update_payload = {"content": final_content, "status": "publish"}
        if media_id:
            update_payload["featured_media"] = media_id
        self.session.post(
            f"{self.api_base}/posts/{wp_id}",
            json=update_payload,
            timeout=30,
        )

        log.info(f"文章已發布: ID={wp_id} — {title}")
        return wp_id

    def create_draft(self, title: str, content: str,
                     category_id: int | None = None,
                     tags: list[str] | None = None,
                     next_episode_title: str = "",
                     next_episode_teaser: str = "") -> int:
        """
        建立 WordPress 草稿（不發布）。
        上傳配圖、組合內容，但保持 status=draft。
        回傳: wp_post_id
        """
        html_content = self._text_to_html(content)

        payload = {
            "title": title,
            "content": html_content,
            "status": "draft",
        }
        if category_id:
            payload["categories"] = [category_id]
        if tags:
            tag_ids = [self._get_or_create_tag(t) for t in tags]
            tag_ids = [tid for tid in tag_ids if tid]
            if tag_ids:
                payload["tags"] = tag_ids

        resp = self.session.post(
            f"{self.api_base}/posts", json=payload, timeout=30,
        )
        resp.raise_for_status()
        wp_id = resp.json()["id"]

        # 取得配圖
        img_html = ""
        media_id = 0
        try:
            img_html, media_id = get_image_html(title, content, self.session, self.api_base, wp_id)
        except Exception as e:
            log.warning(f"配圖失敗（不影響草稿）: {e}")

        # 組合最終內容：圖片 + 本文 + 下集預告
        final_content = ""
        if img_html:
            final_content += img_html + "\n\n"
        final_content += html_content
        if next_episode_teaser:
            final_content += self._build_teaser(next_episode_title, next_episode_teaser)

        # 更新草稿內容（保持 draft 狀態）
        update_payload = {"content": final_content, "status": "draft"}
        if media_id:
            update_payload["featured_media"] = media_id
        self.session.post(
            f"{self.api_base}/posts/{wp_id}",
            json=update_payload,
            timeout=30,
        )

        log.info(f"草稿已建立: ID={wp_id} — {title}")
        return wp_id

    def publish_draft(self, wp_post_id: int) -> tuple[int, str]:
        """將已存在的 WP 草稿改為 publish。回傳 (wp_post_id, permalink)。"""
        resp = self.session.post(
            f"{self.api_base}/posts/{wp_post_id}",
            json={"status": "publish"},
            timeout=30,
        )
        resp.raise_for_status()
        permalink = resp.json().get("link", "")
        log.info(f"草稿已發布: ID={wp_post_id} — {permalink}")
        return wp_post_id, permalink

    @staticmethod
    def _build_teaser(next_title: str, teaser_text: str) -> str:
        """產生下集預告 HTML。"""
        return (
            f'\n\n<hr />\n'
            f'<p style="color:#888;font-size:0.9em;">── 待續 ──</p>\n'
            f'<p><strong>下一集 → {next_title}</strong></p>\n'
            f'<p style="color:#555;font-style:italic;">{teaser_text}</p>'
        )

    def _get_or_create_tag(self, tag_name: str) -> int | None:
        tag_name = tag_name.strip()
        if tag_name in self._tag_cache:
            return self._tag_cache[tag_name]
        try:
            resp = self.session.get(
                f"{self.api_base}/tags",
                params={"search": tag_name, "per_page": 5}, timeout=15,
            )
            resp.raise_for_status()
            for t in resp.json():
                if t["name"].lower() == tag_name.lower():
                    self._tag_cache[tag_name] = t["id"]
                    return t["id"]
            resp = self.session.post(
                f"{self.api_base}/tags",
                json={"name": tag_name}, timeout=15,
            )
            resp.raise_for_status()
            new_tag = resp.json()
            self._tag_cache[tag_name] = new_tag["id"]
            log.info(f"已建立 tag: {tag_name} (ID={new_tag['id']})")
            return new_tag["id"]
        except requests.RequestException as e:
            log.warning(f"處理 tag '{tag_name}' 失敗: {e}")
            return None

    @staticmethod
    def _text_to_html(text: str) -> str:
        # 如果 AI 輸出沒有雙換行，把單換行當段落分隔
        if "\n\n" not in text and "\n" in text:
            text = text.replace("\n", "\n\n")
        paragraphs = text.split("\n\n")
        html_parts = []
        for p in paragraphs:
            p = p.strip()
            if p:
                p = p.replace("\n", "<br>\n")
                html_parts.append(f"<p>{p}</p>")
        return "\n\n".join(html_parts)
