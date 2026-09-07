"""
retrieval.py — TF-IDF retrieval engine over MedQuAD QA pairs.

Steps:
  1. Vectorise all questions with TF-IDF at build time.
  2. At query time, transform the user query and compute cosine similarity.
  3. Return the top-k most relevant QA pairs.
"""

from __future__ import annotations
import pickle
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download NLTK stopwords silently
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))
CACHE_PATH = Path("retrieval_cache.pkl")


@dataclass
class RetrievalResult:
    question:   str
    answer:     str
    focus:      str
    qtype:      str
    source:     str
    url:        str
    score:      float          # cosine similarity 0–1
    rank:       int


class RetrievalEngine:
    """TF-IDF based retrieval over MedQuAD questions."""

    def __init__(self) -> None:
        self._df:         pd.DataFrame | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix      = None    # sparse TF-IDF matrix

    # ── Build / Load ──────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame, use_cache: bool = True) -> None:
        """Fit the TF-IDF vectoriser on all questions.  Cache to disk."""
        if use_cache and CACHE_PATH.exists():
            self._load_cache()
            return

        self._df = df.copy().reset_index(drop=True)

        self._vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),          # unigrams + bigrams
            min_df=2,
            max_df=0.90,
            stop_words=list(STOP_WORDS),
            sublinear_tf=True,
        )

        # Combine question + focus for richer matching
        corpus = (df["question"] + " " + df["focus"]).tolist()
        self._matrix = self._vectorizer.fit_transform(corpus)

        if use_cache:
            self._save_cache()

    def _save_cache(self) -> None:
        with open(CACHE_PATH, "wb") as f:
            pickle.dump((self._df, self._vectorizer, self._matrix), f)

    def _load_cache(self) -> None:
        with open(CACHE_PATH, "rb") as f:
            self._df, self._vectorizer, self._matrix = pickle.load(f)

    # ── Query ─────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return the top_k most relevant QA pairs for a query."""
        if self._vectorizer is None or self._matrix is None:
            raise RuntimeError("Call build() before search().")

        q_vec  = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_idx, start=1):
            row = self._df.iloc[idx]
            results.append(RetrievalResult(
                question=row["question"],
                answer=row["answer"],
                focus=row["focus"],
                qtype=row["qtype"],
                source=row["source"],
                url=row["url"],
                score=float(scores[idx]),
                rank=rank,
            ))

        return results

    @property
    def size(self) -> int:
        return len(self._df) if self._df is not None else 0
