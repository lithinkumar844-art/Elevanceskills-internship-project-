"""
retrieval.py — Hybrid BM25 + TF-IDF retrieval engine for arXiv papers.

Pipeline:
  1. BM25  → fast candidate retrieval (top-K)
  2. TF-IDF cosine → re-ranking the candidates
  3. Filter by year / category if requested
"""

from __future__ import annotations
import logging
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

import config

logger = logging.getLogger(__name__)
STOP = set(stopwords.words("english"))
CACHE = config.CACHE_DIR / "retrieval_engine.pkl"


@dataclass
class SearchResult:
    idx:            int
    id:             str
    title:          str
    abstract:       str
    authors:        str
    categories:     str
    category_label: str
    year:           int | None
    url:            str
    bm25_score:     float
    tfidf_score:    float
    combined_score: float


def _tokenise(text: str) -> list[str]:
    return [
        w.lower() for w in re.findall(r"\b[a-z]{2,}\b", text.lower())
        if w not in STOP
    ]


class RetrievalEngine:
    """Hybrid BM25 + TF-IDF search over arXiv CS papers."""

    def __init__(self) -> None:
        self._df:        pd.DataFrame | None = None
        self._bm25:      BM25Okapi | None    = None
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_mat  = None

    # ── Build / cache ─────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, use_cache: bool = True) -> None:
        if use_cache and CACHE.exists():
            self._load()
            logger.info("Retrieval engine loaded from cache (%d papers).", len(self._df))
            return

        self._df = df.copy().reset_index(drop=True)
        corpus   = (df["title"] + " " + df["abstract"]).tolist()

        # BM25
        tokenised    = [_tokenise(t) for t in corpus]
        self._bm25   = BM25Okapi(tokenised)

        # TF-IDF re-ranker
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=60_000,
            min_df=2,
            sublinear_tf=True,
            stop_words="english",
        )
        self._tfidf_mat = self._vectorizer.fit_transform(corpus)

        if use_cache:
            self._save()
        logger.info("Retrieval engine built for %d papers.", len(self._df))

    def _save(self) -> None:
        with open(CACHE, "wb") as f:
            pickle.dump((self._df, self._bm25, self._vectorizer, self._tfidf_mat), f)

    def _load(self) -> None:
        with open(CACHE, "rb") as f:
            self._df, self._bm25, self._vectorizer, self._tfidf_mat = pickle.load(f)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:         str,
        top_k:         int  = 10,
        year_min:      int | None = None,
        year_max:      int | None = None,
        categories:    list[str] | None = None,
    ) -> list[SearchResult]:

        if self._bm25 is None:
            raise RuntimeError("Call build() before search().")

        tokens     = _tokenise(query)
        bm25_scores = self._bm25.get_scores(tokens)
        top_bm25   = np.argsort(bm25_scores)[::-1][:config.TOP_K_BM25]

        # TF-IDF re-rank on candidates only
        q_vec   = self._vectorizer.transform([query])
        cand_mat = self._tfidf_mat[top_bm25]
        cos_scores = cosine_similarity(q_vec, cand_mat).flatten()

        # Normalise BM25 scores to [0,1]
        bm25_subset = bm25_scores[top_bm25]
        bm25_max    = bm25_subset.max() or 1.0
        bm25_norm   = bm25_subset / bm25_max

        combined = 0.5 * bm25_norm + 0.5 * cos_scores
        order    = np.argsort(combined)[::-1]

        results: list[SearchResult] = []
        for rank_pos in order:
            idx = int(top_bm25[rank_pos])
            row = self._df.iloc[idx]

            # Apply filters
            if year_min and (row.get("year") or 0) < year_min:
                continue
            if year_max and (row.get("year") or 9999) > year_max:
                continue
            if categories:
                cats = str(row.get("categories", ""))
                if not any(c in cats for c in categories):
                    continue

            results.append(SearchResult(
                idx=idx,
                id=str(row.get("id", "")),
                title=str(row.get("title", "")),
                abstract=str(row.get("abstract", "")),
                authors=str(row.get("authors", "")),
                categories=str(row.get("categories", "")),
                category_label=str(row.get("category_label", "CS")),
                year=int(row["year"]) if pd.notna(row.get("year")) else None,
                url=str(row.get("url", "")),
                bm25_score=float(bm25_scores[idx]),
                tfidf_score=float(cos_scores[rank_pos]),
                combined_score=float(combined[rank_pos]),
            ))

            if len(results) >= top_k:
                break

        return results

    # ── Paper lookup ──────────────────────────────────────────────────────────

    def get_paper(self, arxiv_id: str) -> pd.Series | None:
        if self._df is None:
            return None
        match = self._df[self._df["id"] == arxiv_id]
        return match.iloc[0] if not match.empty else None

    @property
    def size(self) -> int:
        return len(self._df) if self._df is not None else 0

    @property
    def df(self) -> pd.DataFrame | None:
        return self._df
