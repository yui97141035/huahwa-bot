"""
wp_scraper.py — PTT 爬蟲
爬取指定看板的熱門文章（推文數 ≥ 閾值），清理後回傳文章內容。
"""

import re
import time
import logging
import requests
from bs4 import BeautifulSoup

log = logging.getLogger("wp-poster.scraper")

PTT_BASE = "https://www.ptt.cc"
PTT_COOKIES = {"over18": "1"}
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
}
CRAWL_DELAY = 1.0  # 每頁間隔秒數（禮貌爬取）


class PTTScraper:
    def __init__(self, min_push: int = 10):
        self.min_push = min_push
        self.session = requests.Session()
        self.session.cookies.update(PTT_COOKIES)
        self.session.headers.update(REQUEST_HEADERS)

    # ── 公開介面 ──────────────────────────────────────────────

    def scrape_board(self, board: str, pages: int = 5) -> list[dict]:
        """
        爬取看板最近 N 頁，回傳符合推文門檻的文章摘要。
        回傳: [{"url", "title", "author", "push_count"}, ...]
        """
        results = []
        url = f"{PTT_BASE}/bbs/{board}/index.html"

        for page_i in range(pages):
            log.info(f"爬取 {board} 第 {page_i + 1}/{pages} 頁: {url}")
            try:
                resp = self.session.get(url, timeout=15)
                resp.raise_for_status()
            except requests.RequestException as e:
                log.error(f"爬取失敗: {e}")
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            entries = soup.select("div.r-ent")

            for entry in entries:
                meta = self._parse_entry(entry, board)
                if meta and meta["push_count"] >= self.min_push:
                    results.append(meta)

            # 取得上一頁連結
            prev_link = self._find_prev_page(soup)
            if not prev_link:
                break
            url = PTT_BASE + prev_link

            if page_i < pages - 1:
                time.sleep(CRAWL_DELAY)

        log.info(f"{board}: 找到 {len(results)} 篇熱門文章 (推文 ≥ {self.min_push})")
        return results

    def fetch_article(self, url: str) -> str:
        """取得單篇文章的純文字內容（已清理 metadata/推文/簽名檔）。"""
        try:
            resp = self.session.get(url, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            log.error(f"取得文章失敗 {url}: {e}")
            return ""

        time.sleep(CRAWL_DELAY)
        soup = BeautifulSoup(resp.text, "html.parser")
        main = soup.select_one("#main-content")
        if not main:
            return ""

        # 移除 metadata 區塊（作者/看板/標題/時間）
        for meta_div in main.select("div.article-metaline, div.article-metaline-right"):
            meta_div.decompose()

        # 移除推文區塊
        for push in main.select("div.push"):
            push.decompose()

        text = main.get_text()
        text = self._clean_content(text)
        return text

    # ── 內部方法 ──────────────────────────────────────────────

    def _parse_entry(self, entry, board: str) -> dict | None:
        """解析單一看板列表項目。"""
        title_tag = entry.select_one("div.title a")
        if not title_tag:
            return None

        title = title_tag.text.strip()
        href = title_tag.get("href", "")
        url = PTT_BASE + href

        # 跳過公告
        if title.startswith("[公告]"):
            return None

        author_tag = entry.select_one("div.author")
        author = author_tag.text.strip() if author_tag else ""

        push_tag = entry.select_one("div.nrec span")
        push_count = self._parse_push_count(push_tag)

        return {
            "url": url,
            "title": title,
            "author": author,
            "board": board,
            "push_count": push_count,
        }

    @staticmethod
    def _parse_push_count(push_tag) -> int:
        """解析推文數：數字、'爆'=100、'XX'=-100、空=0。"""
        if not push_tag:
            return 0
        text = push_tag.text.strip()
        if text == "爆":
            return 100
        if text.startswith("X"):
            return -100
        try:
            return int(text)
        except ValueError:
            return 0

    @staticmethod
    def _find_prev_page(soup) -> str | None:
        """從分頁按鈕取得上一頁連結。"""
        paging = soup.select("div.btn-group-paging a")
        for a in paging:
            if "上頁" in a.text:
                return a.get("href")
        return None

    @staticmethod
    def _clean_content(text: str) -> str:
        """清理文章內容：移除簽名檔、多餘空行。"""
        # 移除簽名檔（--\n 之後的內容）
        sig_patterns = ["\n--\n", "\n-- \n", "\n---\n"]
        for sig in sig_patterns:
            idx = text.find(sig)
            if idx != -1:
                text = text[:idx]

        # 移除 ※ 開頭的 PTT 系統行（發信站、文章網址等）
        lines = text.split("\n")
        lines = [ln for ln in lines if not ln.strip().startswith("※")]

        text = "\n".join(lines)
        # 壓縮連續空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
