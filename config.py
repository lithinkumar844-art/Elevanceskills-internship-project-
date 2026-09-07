"""
config.py — Central configuration for the Dynamic Knowledge Base Chatbot
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL      = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# ── Embeddings ────────────────────────────────────────────────────────────────
# Use a local HuggingFace model so no API key is required for embeddings
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME    = os.getenv("COLLECTION_NAME",    "knowledge_base")

# ── Ingestion ────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE",    "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Scheduler ────────────────────────────────────────────────────────────────
UPDATE_INTERVAL_HOURS = float(os.getenv("UPDATE_INTERVAL_HOURS", "6"))

# ── Sources ───────────────────────────────────────────────────────────────────
# Add your own URLs, RSS feeds, file paths, or API configs here
WEB_SOURCES: list[str] = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
]

RSS_FEEDS: list[str] = [
    "https://feeds.feedburner.com/TechCrunch",           # Tech news
    "https://rss.arxiv.org/rss/cs.AI",                  # arXiv AI papers
]

FILE_SOURCES: list[str] = [
    # "./docs/my_document.pdf",
    # "./docs/notes.txt",
]

API_SOURCES: list[dict] = [
    # {
    #     "url": "https://api.example.com/articles",
    #     "headers": {"Authorization": "Bearer YOUR_TOKEN"},
    #     "text_field": "body",    # JSON key that contains the article text
    # }
]
