"""
data_loader.py — Load arXiv papers into a pandas DataFrame.

Priority:
  1. Full Kaggle JSONL  (data/arxiv-metadata-oai-snapshot.json)
  2. Pre-built sample CSV  (data/cs_papers_sample.csv)
  3. Generate sample from arXiv public API  (fallback, ~500 papers)
"""

from __future__ import annotations
import json
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

CS_CATS = set(config.CS_CATEGORIES.keys())

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cat_label(cats: str) -> str:
    """Return the primary CS sub-category label."""
    for c in cats.split():
        if c in config.CS_CATEGORIES:
            return config.CS_CATEGORIES[c]
    return "Computer Science"


def _is_cs(cats: str) -> bool:
    return any(c in CS_CATS for c in cats.split())


# ── Source 1: Full Kaggle JSONL ───────────────────────────────────────────────

def _load_from_jsonl(path: Path) -> pd.DataFrame:
    """Stream the big JSONL file and keep only CS papers."""
    records, n_read = [], 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError:
                continue

            cats = p.get("categories", "")
            if not _is_cs(cats):
                continue

            abstract = (p.get("abstract") or "").replace("\n", " ").strip()
            if len(abstract) < 100:
                continue

            # Parse year from update_date or versions
            year = None
            ud = p.get("update_date", "")
            if ud:
                try:
                    year = int(ud[:4])
                except ValueError:
                    pass

            records.append({
                "id":        p.get("id", ""),
                "title":     (p.get("title") or "").replace("\n", " ").strip(),
                "abstract":  abstract,
                "authors":   p.get("authors", ""),
                "categories": cats,
                "category_label": _cat_label(cats),
                "year":      year,
                "doi":       p.get("doi", ""),
                "url":       f"https://arxiv.org/abs/{p.get('id','')}",
            })

            n_read += 1
            if n_read >= config.MAX_PAPERS_FROM_JSONL:
                break

    df = pd.DataFrame(records).drop_duplicates("id").reset_index(drop=True)
    logger.info("Loaded %d CS papers from JSONL.", len(df))
    return df


# ── Source 2: Cached sample CSV ───────────────────────────────────────────────

def _load_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d papers from sample CSV.", len(df))
    return df


# ── Source 3: arXiv public API (fallback) ────────────────────────────────────

def _fetch_from_api(total: int = 500) -> pd.DataFrame:
    """Fetch papers via the free arXiv Atom API (no key needed)."""
    NS = "http://www.w3.org/2005/Atom"
    records, start = [], 0
    batch = 100

    cats_query = " OR ".join(f"cat:{c}" for c in list(CS_CATS)[:5])

    logger.info("Fetching CS papers from arXiv API...")
    while len(records) < total:
        url = (
            f"http://export.arxiv.org/api/query"
            f"?search_query={cats_query}"
            f"&start={start}&max_results={batch}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
        except Exception as e:
            logger.warning("API fetch failed: %s", e)
            break

        root = ET.fromstring(r.text)
        entries = root.findall(f"{{{NS}}}entry")
        if not entries:
            break

        for entry in entries:
            arxiv_id = (entry.findtext(f"{{{NS}}}id") or "").split("/abs/")[-1].strip()
            title    = (entry.findtext(f"{{{NS}}}title") or "").replace("\n", " ").strip()
            abstract = (entry.findtext(f"{{{NS}}}summary") or "").replace("\n", " ").strip()
            authors  = ", ".join(
                (a.findtext(f"{{{NS}}}name") or "")
                for a in entry.findall(f"{{{NS}}}author")
            )
            cats = " ".join(
                t.attrib.get("term", "")
                for t in entry.findall(f"{{{NS}}}category")
            )
            published = entry.findtext(f"{{{NS}}}published") or ""
            year = int(published[:4]) if published else None

            if len(abstract) < 100:
                continue

            records.append({
                "id":             arxiv_id,
                "title":          title,
                "abstract":       abstract,
                "authors":        authors,
                "categories":     cats,
                "category_label": _cat_label(cats),
                "year":           year,
                "doi":            "",
                "url":            f"https://arxiv.org/abs/{arxiv_id}",
            })

        start += batch
        time.sleep(3)   # be polite to the API

    df = pd.DataFrame(records).drop_duplicates("id").reset_index(drop=True)
    # Cache it
    df.to_csv(config.SAMPLE_CSV, index=False)
    logger.info("Fetched and cached %d papers from arXiv API.", len(df))
    return df


# ── Public entry point ────────────────────────────────────────────────────────

def load_papers(force_api: bool = False) -> pd.DataFrame:
    """
    Load CS papers using the best available source.

    Returns a DataFrame with columns:
      id, title, abstract, authors, categories, category_label, year, doi, url
    """
    if not force_api and config.ARXIV_JSONL.exists():
        return _load_from_jsonl(config.ARXIV_JSONL)

    if not force_api and config.SAMPLE_CSV.exists():
        return _load_from_csv(config.SAMPLE_CSV)

    return _fetch_from_api(total=500)
