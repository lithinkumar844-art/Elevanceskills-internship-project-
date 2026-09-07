# Dynamic Knowledge Base Chatbot

A chatbot that **automatically expands its knowledge** over time by
periodically pulling from web pages, RSS feeds, local files, and REST APIs —
and storing everything in a persistent ChromaDB vector database.

---

## Architecture

```
Sources (Web / RSS / Files / APIs)
         │
         ▼
  Ingestion Pipeline
  (fetch → chunk → deduplicate → embed)
         │
         ▼
  ChromaDB (persistent vector store)
         │
         ▼
  LangChain RAG Chain  ←→  User Chat
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Configure your sources (optional)
Open `config.py` and add your URLs, RSS feeds, file paths, or API endpoints.

### 4. Run the chatbot
```bash
python main.py
```

The chatbot will:
1. Run an **immediate ingestion** on startup
2. Start a **background scheduler** that re-ingests every 6 hours (configurable)
3. Open an **interactive chat loop**

### One-shot ingestion only
```bash
python main.py --ingest
```

---

## Chat Commands

| Command  | Action                                |
|----------|---------------------------------------|
| `quit`   | Exit the chatbot                      |
| `clear`  | Reset conversation history            |
| `status` | Show how many chunks are in the store |

---

## Configuration (`config.py`)

| Variable               | Default                                   | Description                         |
|------------------------|-------------------------------------------|-------------------------------------|
| `LLM_MODEL`            | `gpt-3.5-turbo`                           | OpenAI chat model                   |
| `EMBEDDING_MODEL`      | `sentence-transformers/all-MiniLM-L6-v2` | Local HuggingFace embedding model   |
| `CHROMA_PERSIST_DIR`   | `./chroma_db`                             | Where ChromaDB stores its data      |
| `CHUNK_SIZE`           | `500`                                     | Max characters per text chunk       |
| `CHUNK_OVERLAP`        | `50`                                      | Characters shared between chunks    |
| `UPDATE_INTERVAL_HOURS`| `6`                                       | How often sources are re-fetched    |

---

## Adding Sources

**Web pages** — add URLs to `WEB_SOURCES` in `config.py`:
```python
WEB_SOURCES = ["https://example.com/article"]
```

**RSS feeds** — add feed URLs to `RSS_FEEDS`:
```python
RSS_FEEDS = ["https://rss.arxiv.org/rss/cs.AI"]
```

**Local files** — add paths to `FILE_SOURCES` (supports `.txt`, `.md`, `.pdf`):
```python
FILE_SOURCES = ["./docs/manual.pdf"]
```

**REST APIs** — add configs to `API_SOURCES`:
```python
API_SOURCES = [{
    "url": "https://api.example.com/articles",
    "headers": {"Authorization": "Bearer TOKEN"},
    "text_field": "body",
}]
```

---

## Project Structure

```
dynamic_kb_chatbot/
├── main.py                   # Entry point
├── config.py                 # All configuration
├── requirements.txt
├── .env.example
├── sources/
│   └── loaders.py            # Web, RSS, file, API loaders
├── ingestion/
│   ├── pipeline.py           # Chunk + dedup + embed pipeline
│   └── scheduler.py          # APScheduler background updater
├── vectordb/
│   └── store.py              # ChromaDB wrapper
└── chatbot/
    └── rag_chain.py          # LangChain RAG chain
```
