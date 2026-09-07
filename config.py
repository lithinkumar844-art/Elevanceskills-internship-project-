"""
config.py — Central configuration for the arXiv CS Expert Chatbot
"""
import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATA_DIR       = BASE_DIR / "data"
CACHE_DIR      = BASE_DIR / "cache"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# The Kaggle arXiv JSONL file (user places it here after downloading)
ARXIV_JSONL    = DATA_DIR / "arxiv-metadata-oai-snapshot.json"

# Pre-built sample file (generated if Kaggle file not present)
SAMPLE_CSV     = DATA_DIR / "cs_papers_sample.csv"

# ── Dataset settings ──────────────────────────────────────────────────────────
# CS sub-categories to include
CS_CATEGORIES = {
    "cs.AI":  "Artificial Intelligence",
    "cs.LG":  "Machine Learning",
    "cs.CL":  "Computation & Language (NLP)",
    "cs.CV":  "Computer Vision",
    "cs.NE":  "Neural & Evolutionary Computing",
    "cs.IR":  "Information Retrieval",
    "cs.RO":  "Robotics",
    "cs.CR":  "Cryptography & Security",
    "cs.DC":  "Distributed Computing",
    "cs.DS":  "Data Structures & Algorithms",
}

MAX_PAPERS_FROM_JSONL = 100_000   # how many CS papers to load from full dataset
SAMPLE_SIZE           = 5_000     # papers in the lightweight built-in sample

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_BM25    = 20   # BM25 candidates
TOP_K_TFIDF   = 5    # final TF-IDF re-rank
MIN_SCORE     = 0.01

# ── LLM (Ollama local, optional) ──────────────────────────────────────────────
OLLAMA_URL    = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL  = os.getenv("OLLAMA_MODEL", "mistral")
USE_LLM       = os.getenv("USE_LLM", "false").lower() == "true"

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE     = "arXiv CS Expert Chatbot"
APP_ICON      = "🔬"
