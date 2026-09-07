"""
vectordb/store.py — Persistent ChromaDB vector store using HuggingFace embeddings.

Wraps LangChain's Chroma integration and exposes a clean interface for:
  - adding new documents
  - similarity search
  - listing existing document IDs (for deduplication)
"""

from __future__ import annotations

import logging
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import config

logger = logging.getLogger(__name__)


class VectorStore:
    """Thin wrapper around LangChain's Chroma client."""

    def __init__(self) -> None:
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._db = Chroma(
            collection_name=config.COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        logger.info(
            "ChromaDB ready — collection '%s' at '%s'",
            config.COLLECTION_NAME,
            config.CHROMA_PERSIST_DIR,
        )

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_documents(self, docs: list[Document], ids: list[str]) -> None:
        """Add new documents with explicit IDs (used for deduplication)."""
        self._db.add_documents(documents=docs, ids=ids)
        logger.debug("Stored %d documents.", len(docs))

    # ── Read ───────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[Document]:
        """Return the k most relevant documents for a query."""
        return self._db.similarity_search(query, k=k)

    def get_existing_ids(self) -> set[str]:
        """Return the set of document IDs already in the collection."""
        try:
            result = self._db._collection.get(include=[])   # only IDs, no embeddings
            return set(result.get("ids", []))
        except Exception:
            return set()

    def count(self) -> int:
        """Return total number of chunks stored."""
        return self._db._collection.count()

    # ── Retriever (LangChain-compatible) ─────────────────────────────────────

    def as_retriever(self, k: int = 5):
        """Return a LangChain retriever for use in chains."""
        return self._db.as_retriever(search_kwargs={"k": k})
