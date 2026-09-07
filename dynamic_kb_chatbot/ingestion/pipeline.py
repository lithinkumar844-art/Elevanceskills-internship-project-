"""
ingestion/pipeline.py — Clean, chunk, deduplicate, and embed raw documents.

Flow:
  raw docs → split into chunks → hash-based dedup → embed → store in ChromaDB
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config

if TYPE_CHECKING:
    from vectordb.store import VectorStore

logger = logging.getLogger(__name__)


def _doc_hash(text: str) -> str:
    """Return a stable SHA-256 fingerprint for a text chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_documents(raw_docs: list[dict]) -> list[Document]:
    """Split raw source dicts into LangChain Document chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks: list[Document] = []

    for doc in raw_docs:
        text   = doc.get("text", "").strip()
        source = doc.get("source", "unknown")
        meta   = doc.get("metadata", {})

        if not text:
            continue

        for chunk_text in splitter.split_text(text):
            chunk_text = chunk_text.strip()
            if chunk_text:
                chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={**meta, "source": source, "doc_hash": _doc_hash(chunk_text)},
                    )
                )

    logger.info("Split %d raw docs → %d chunks", len(raw_docs), len(chunks))
    return chunks


def _deduplicate(
    chunks: list[Document],
    existing_ids: set[str],
) -> tuple[list[Document], list[str]]:
    """
    Remove chunks whose hash already exists in the vector store.

    Returns:
        new_chunks  — documents not yet in the store
        new_ids     — their corresponding IDs (used as ChromaDB doc IDs)
    """
    new_chunks: list[Document] = []
    new_ids:    list[str]      = []

    for chunk in chunks:
        doc_id = chunk.metadata["doc_hash"]
        if doc_id not in existing_ids:
            new_chunks.append(chunk)
            new_ids.append(doc_id)

    logger.info(
        "Dedup: %d total chunks, %d already stored, %d new",
        len(chunks),
        len(chunks) - len(new_chunks),
        len(new_chunks),
    )
    return new_chunks, new_ids


def ingest(raw_docs: list[dict], store: "VectorStore") -> int:
    """
    Full ingestion pipeline.

    Args:
        raw_docs: list of {"text": ..., "source": ..., "metadata": ...}
        store:    VectorStore instance (wraps ChromaDB)

    Returns:
        Number of NEW chunks added to the store.
    """
    if not raw_docs:
        logger.info("No documents to ingest.")
        return 0

    chunks = _split_documents(raw_docs)
    if not chunks:
        return 0

    existing_ids       = store.get_existing_ids()
    new_chunks, new_ids = _deduplicate(chunks, existing_ids)

    if not new_chunks:
        logger.info("All chunks already in store — nothing to add.")
        return 0

    store.add_documents(new_chunks, ids=new_ids)
    logger.info("✅ Added %d new chunks to the knowledge base.", len(new_chunks))
    return len(new_chunks)
