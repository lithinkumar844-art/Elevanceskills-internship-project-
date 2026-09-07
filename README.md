# arXiv CS Expert Chatbot

A Streamlit-based domain-expert chatbot for Computer Science research, powered by the arXiv dataset.

---

## Features

| Feature | Details |
|---------|---------|
| **Hybrid retrieval** | BM25 candidate selection + TF-IDF cosine re-ranking |
| **NLP pipeline** | Keyword extraction, extractive summarisation, concept detection |
| **CS concept glossary** | 35+ key concepts with definitions and paper context |
| **Concept maps** | Plotly network graphs linking paper topics |
| **Dataset analytics** | Category distribution, publication timeline, keyword bubbles |
| **Follow-up handling** | Detects follow-up questions and builds on conversation context |
| **Optional LLM** | Plug in local Ollama (Mistral, LLaMA3) for generative answers |
| **4-tab UI** | Chat · Paper Search · Analytics · Concept Explorer |

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get arXiv data (choose one option)

**Option A — Full Kaggle dataset (~4GB, 1.7M papers):**
```bash
# Download from https://www.kaggle.com/datasets/Cornell-University/arxiv
# Place the file at:
data/arxiv-metadata-oai-snapshot.json
```

**Option B — Auto-fetch from arXiv API (free, ~500 CS papers, no login):**
```bash
# Just run the app — it will fetch papers automatically on first launch
streamlit run app.py
```

### 3. Run the app
```bash
streamlit run app.py
```
Opens at **http://localhost:8501**

---

## Optional: Use a local LLM (Ollama)

For generative explanations instead of template-based answers:

```bash
# Install Ollama: https://ollama.com
ollama pull mistral        # or llama3, phi3, gemma, etc.
ollama serve               # starts the API at localhost:11434
```

Then toggle **"Use local LLM"** in the sidebar and set the model name.

---

## Project Structure

```
arxiv_chatbot/
├── app.py                  # Streamlit UI (4 tabs)
├── config.py               # All settings
├── data_loader.py          # JSONL / CSV / API paper loader
├── retrieval.py            # Hybrid BM25 + TF-IDF engine
├── nlp_pipeline.py         # Keywords, summarisation, NER, intent
├── llm_interface.py        # Ollama + template response generator
├── visualizations.py       # Plotly charts
├── requirements.txt
├── data/                   # Place arxiv JSONL here (or auto-generated CSV)
└── cache/                  # Index cache (auto-created)
```

---

## Architecture

```
User query
    │
    ▼
Intent classifier        ← explain / summarise / search / compare / recent
    │
    ▼
BM25 retrieval           ← fast candidate selection (top-20)
    │
    ▼
TF-IDF re-ranking        ← cosine similarity re-score (top-5)
    │
    ▼
NLP pipeline             ← summarise + keyword extract + concept detect
    │
    ▼
LLM / Template           ← Ollama (generative) or structured template
    │
    ▼
Streamlit UI             ← chat bubble + concept map + source links
```

---

## CS Sub-Categories Covered

| Code | Name |
|------|------|
| cs.AI | Artificial Intelligence |
| cs.LG | Machine Learning |
| cs.CL | Computation & Language (NLP) |
| cs.CV | Computer Vision |
| cs.NE | Neural & Evolutionary Computing |
| cs.IR | Information Retrieval |
| cs.RO | Robotics |
| cs.CR | Cryptography & Security |
| cs.DC | Distributed Computing |
| cs.DS | Data Structures & Algorithms |
