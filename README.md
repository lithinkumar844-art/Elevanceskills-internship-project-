# Elevance Skills Internship Project

This repository contains a collection of AI/NLP internship projects, each self-contained in its own folder with its own dependencies and documentation.

## Projects

### 🧠 [`dynamic_kb_chatbot`](./dynamic_kb_chatbot)
A retrieval-augmented chatbot built on a dynamic knowledge base. Includes an ingestion pipeline with a scheduler, document loaders, a vector database store, and a RAG chain for question answering.

- **Key modules:** `chatbot/`, `ingestion/`, `sources/`, `vectordb/`
- **Entry point:** `main.py`
- See [`dynamic_kb_chatbot/README.md`](./dynamic_kb_chatbot/README.md) for setup details.

### 🩺 [`Medical Q&A Chatbot`](./Medical%20Q%26A%20Chatbot)
A medical question-answering chatbot trained/evaluated on the MedQuAD dataset, with entity recognition and retrieval components.

- **Key modules:** `app.py`, `data_loader.py`, `entity_recognition.py`, `retrieval.py`
- **Dataset:** `medquad_raw/` (CancerGov, NIDDK, NINDS, and CDC Q&A sources)
- See [`Medical Q&A Chatbot/README.md`](./Medical%20Q%26A%20Chatbot/README.md) for setup details.

### 🌐 [`multilingual_chatbot`](./multilingual_chatbot)
A chatbot supporting multiple languages, paired with a sentiment-aware chatbot variant.

- **Key files:** `multilingual_chatbot.py`, `sentiment_chatbot.py`
- See [`multilingual_chatbot/README.md`](./multilingual_chatbot/README.md) for setup details.

### 🎛️ [`multimodal_assistant`](./multimodal_assistant)
A multimodal AI assistant with an orchestration layer covering reasoning, memory, vision, ambiguity resolution, and validation — plus a test suite.

- **Key modules:** `core/` (`orchestrator.py`, `reasoning.py`, `memory.py`, `vision.py`, `ambiguity.py`, `validator.py`, `config.py`)
- **Tests:** `tests/test_pipeline.py`
- **Entry point:** `app.py` (see `example_usage.py` for a usage sample)
- See [`multimodal_assistant/README.md`](./multimodal_assistant/README.md) for setup details, including environment variables (`.env.example`).

### 💬 [`sentiment_chatbot`](./sentiment_chatbot)
A standalone sentiment-analysis chatbot with a logged chat history sample.

- **Key files:** `sentiment_chatbot.py`, `chat_log.csv`
- See [`sentiment_chatbot/README.md`](./sentiment_chatbot/README.md) for setup details.

### 🔬 [`advanced NLP techniques`](./advanced%20NLP%20techniques)
An NLP pipeline combining retrieval, an LLM interface, and data visualization for advanced text-processing tasks.

- **Key files:** `app.py`, `config.py`, `data_loader.py`, `llm_interface.py`, `nlp_pipeline.py`, `retrieval.py`, `visualizations.py`
- See [`advanced NLP techniques/README.md`](./advanced%20NLP%20techniques/README.md) for setup details.

## Repository Structure

```
.
├── dynamic_kb_chatbot/
├── Medical Q&A Chatbot/
├── multilingual_chatbot/
├── multimodal_assistant/
├── sentiment_chatbot/
└── advanced NLP techniques/
```

Each project folder is self-contained with its own `requirements.txt` and `README.md`. Refer to the individual project READMEs for installation and usage instructions specific to that project.

## Getting Started

1. Clone the repository:
   ```powershell
   git clone https://github.com/lithinkumar844-art/Elevanceskills-internship-project-.git
   cd Elevanceskills-internship-project-
   ```
2. Navigate into the project you want to run, e.g.:
   ```powershell
   cd dynamic_kb_chatbot
   pip install -r requirements.txt
   ```
3. Follow that project's own README for further setup (API keys, environment variables, etc.).

## Notes

- Some projects use a `.env` file for configuration (see `.env.example` in `multimodal_assistant`) — never commit real secrets.
- Cached bytecode (`__pycache__/`) and dataset archives are excluded from version control via `.gitignore`.
