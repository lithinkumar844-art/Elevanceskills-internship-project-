"""
sources/loaders.py — Fetch raw text from multiple source types.

Each loader returns a list of dicts:
    [{"text": "...", "source": "...", "metadata": {...}}, ...]
"""

from __future__ import annotations

import logging
import requests
import feedparser

from bs4 import BeautifulSoup
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Web / HTML loader ─────────────────────────────────────────────────────────

def load_web_page(url: str, timeout: int = 15) -> list[dict]:
    """Scrape visible text from a web page URL."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove boilerplate tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        if not text.strip():
            return []

        return [{"text": text, "source": url, "metadata": {"type": "web", "url": url}}]

    except Exception as exc:
        logger.warning("Web load failed for %s: %s", url, exc)
        return []


# ── RSS / Atom feed loader ────────────────────────────────────────────────────

def load_rss_feed(feed_url: str, max_entries: int = 10) -> list[dict]:
    """Parse an RSS/Atom feed and return the latest entries as text chunks."""
    try:
        feed = feedparser.parse(feed_url)
        results = []

        for entry in feed.entries[:max_entries]:
            title   = entry.get("title", "")
            summary = entry.get("summary", "")
            link    = entry.get("link", feed_url)

            # Strip HTML from summary if present
            if summary:
                summary = BeautifulSoup(summary, "html.parser").get_text(strip=True)

            text = f"{title}\n\n{summary}".strip()
            if text:
                results.append({
                    "text": text,
                    "source": link,
                    "metadata": {"type": "rss", "feed": feed_url, "title": title},
                })

        return results

    except Exception as exc:
        logger.warning("RSS load failed for %s: %s", feed_url, exc)
        return []


# ── Local file loader ─────────────────────────────────────────────────────────

def load_file(file_path: str) -> list[dict]:
    """Load text from a local file (.txt, .md, .pdf)."""
    path = Path(file_path)
    if not path.exists():
        logger.warning("File not found: %s", file_path)
        return []

    try:
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif suffix in {".txt", ".md", ".rst"}:
            text = path.read_text(encoding="utf-8", errors="ignore")

        else:
            logger.warning("Unsupported file type: %s", suffix)
            return []

        if not text.strip():
            return []

        return [{
            "text": text,
            "source": str(path),
            "metadata": {"type": "file", "filename": path.name},
        }]

    except Exception as exc:
        logger.warning("File load failed for %s: %s", file_path, exc)
        return []


# ── API / JSON loader ─────────────────────────────────────────────────────────

def load_api(
    url: str,
    headers: Optional[dict] = None,
    text_field: str = "content",
    timeout: int = 15,
) -> list[dict]:
    """
    Fetch JSON from a REST API endpoint.

    Supports two response shapes:
      - A list of objects: [{text_field: "...", ...}, ...]
      - A single object with a text_field key
    """
    try:
        resp = requests.get(url, headers=headers or {}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        items = data if isinstance(data, list) else [data]
        results = []

        for item in items:
            text = item.get(text_field, "")
            if isinstance(text, str) and text.strip():
                results.append({
                    "text": text,
                    "source": url,
                    "metadata": {"type": "api", "url": url},
                })

        return results

    except Exception as exc:
        logger.warning("API load failed for %s: %s", url, exc)
        return []
