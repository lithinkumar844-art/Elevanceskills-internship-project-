"""
ingestion/scheduler.py — Periodically fetches new data and updates ChromaDB.

Uses APScheduler (BackgroundScheduler) so the chatbot stays responsive
while updates happen in a background thread.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

import config
from sources.loaders import load_web_page, load_rss_feed, load_file, load_api
from ingestion.pipeline import ingest

if TYPE_CHECKING:
    from vectordb.store import VectorStore

logger = logging.getLogger(__name__)


def fetch_all_sources() -> list[dict]:
    """Fetch raw documents from every configured source."""
    raw: list[dict] = []

    # Web pages
    for url in config.WEB_SOURCES:
        logger.info("Fetching web page: %s", url)
        raw.extend(load_web_page(url))

    # RSS / Atom feeds
    for feed_url in config.RSS_FEEDS:
        logger.info("Fetching RSS feed: %s", feed_url)
        raw.extend(load_rss_feed(feed_url))

    # Local files
    for file_path in config.FILE_SOURCES:
        logger.info("Loading file: %s", file_path)
        raw.extend(load_file(file_path))

    # REST APIs
    for api_cfg in config.API_SOURCES:
        logger.info("Calling API: %s", api_cfg["url"])
        raw.extend(
            load_api(
                url=api_cfg["url"],
                headers=api_cfg.get("headers"),
                text_field=api_cfg.get("text_field", "content"),
            )
        )

    logger.info("Total raw documents fetched: %d", len(raw))
    return raw


def run_update(store: "VectorStore") -> None:
    """Single update cycle: fetch → ingest."""
    logger.info("⏱  Knowledge base update started...")
    raw_docs = fetch_all_sources()
    added    = ingest(raw_docs, store)
    logger.info("⏱  Update complete. %d new chunks added. Total: %d", added, store.count())


class KnowledgeBaseScheduler:
    """Wrapper around APScheduler that runs periodic KB updates."""

    def __init__(self, store: "VectorStore") -> None:
        self._store     = store
        self._scheduler = BackgroundScheduler(daemon=True)

    def start(self, run_now: bool = True) -> None:
        """
        Start the background scheduler.

        Args:
            run_now: if True, perform an immediate update before scheduling.
        """
        if run_now:
            logger.info("Running initial knowledge base update...")
            run_update(self._store)

        self._scheduler.add_job(
            func=run_update,
            trigger=IntervalTrigger(hours=config.UPDATE_INTERVAL_HOURS),
            args=[self._store],
            id="kb_update",
            name="Knowledge base periodic update",
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "Scheduler running — updates every %.1f hour(s).",
            config.UPDATE_INTERVAL_HOURS,
        )

    def stop(self) -> None:
        """Gracefully shut down the scheduler."""
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
            logger.info("Scheduler stopped.")
