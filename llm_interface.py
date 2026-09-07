"""
llm_interface.py — LLM integration for explanation generation.

Two modes:
  1. Local Ollama (e.g. Mistral, LLaMA3) — when USE_LLM=true and Ollama is running
  2. Template-based fallback — works without any LLM; uses extractive NLP

The template mode produces high-quality structured answers without a GPU.
"""

from __future__ import annotations
import logging
import re
import textwrap

import requests

import config
from nlp_pipeline import summarise, extract_concepts, extract_keywords, classify_intent

logger = logging.getLogger(__name__)


# ── Ollama client ─────────────────────────────────────────────────────────────

def _ollama_generate(prompt: str, model: str = config.OLLAMA_MODEL) -> str | None:
    try:
        r = requests.post(
            f"{config.OLLAMA_URL}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Ollama unavailable: %s", e)
        return None


def _ollama_available() -> bool:
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        return r.ok
    except Exception:
        return False


# ── Template-based answer generation ─────────────────────────────────────────

def _format_authors(authors: str) -> str:
    parts = [a.strip() for a in authors.split(",")]
    if len(parts) > 3:
        return ", ".join(parts[:3]) + " et al."
    return ", ".join(parts)


def _concept_block(text: str) -> str:
    concepts = extract_concepts(text)[:4]
    if not concepts:
        return ""
    lines = "\n".join(f"  • **{c.term}** — {c.definition}" for c in concepts)
    return f"\n\n**Key concepts:**\n{lines}"


def generate_paper_explanation(paper: dict, query: str) -> str:
    """
    Generate a rich explanation for a single paper.
    Uses LLM if available, otherwise templates.
    """
    title    = paper.get("title", "")
    abstract = paper.get("abstract", "")
    authors  = _format_authors(paper.get("authors", "Unknown"))
    year     = paper.get("year", "")
    cat      = paper.get("category_label", "Computer Science")
    url      = paper.get("url", "")

    summary  = summarise(abstract, n_sentences=3)
    keywords = extract_keywords(abstract, top_n=6)
    intent   = classify_intent(query)

    # ── Try Ollama first ──────────────────────────────────────────────────────
    if config.USE_LLM and _ollama_available():
        prompt = f"""You are an expert computer science researcher. 
A user asked: "{query}"

Here is an arXiv paper that is relevant:
Title: {title}
Authors: {authors} ({year})
Abstract: {abstract}

Please:
1. Summarise the key contribution in 2-3 sentences.
2. Explain how it relates to the user's question.
3. Highlight the most important technical concepts.
4. State any limitations mentioned.

Be concise, clear, and accessible to a graduate CS student."""

        llm_response = _ollama_generate(prompt)
        if llm_response:
            return f"### 📄 {title}\n*{authors}, {year}*\n\n{llm_response}\n\n🔗 [{url}]({url})"

    # ── Template fallback ─────────────────────────────────────────────────────
    kw_str = ", ".join(f"`{k}`" for k in keywords)

    if intent == "explain":
        body = (
            f"This paper by {authors} ({year}) addresses the topic of **{cat}**.\n\n"
            f"**Summary:** {summary}\n"
            f"{_concept_block(abstract)}\n\n"
            f"**Keywords:** {kw_str}"
        )
    elif intent == "summarise":
        body = f"**TL;DR:** {summary}\n\n**Keywords:** {kw_str}"
    else:
        body = (
            f"{summary}\n"
            f"{_concept_block(abstract)}\n\n"
            f"**Keywords:** {kw_str}"
        )

    return (
        f"### 📄 {title}\n"
        f"*{authors} · {year} · {cat}*\n\n"
        f"{body}\n\n"
        f"🔗 [Read on arXiv]({url})"
    )


def generate_concept_explanation(concept: str, definition: str, context_papers: list[dict]) -> str:
    """Explain a CS concept with paper context."""
    if config.USE_LLM and _ollama_available():
        paper_ctx = "\n".join(
            f"- {p.get('title','')} ({p.get('year','')})"
            for p in context_papers[:3]
        )
        prompt = (
            f"Explain the concept of '{concept}' in computer science. "
            f"Definition: {definition}\n"
            f"Relevant recent papers:\n{paper_ctx}\n\n"
            "Give a clear, structured explanation suitable for a grad student. "
            "Include: what it is, why it matters, and how it's used in practice."
        )
        resp = _ollama_generate(prompt)
        if resp:
            return resp

    # Template fallback
    papers_md = "\n".join(
        f"  • *{p.get('title','')}* ({p.get('year','')})"
        for p in context_papers[:3]
    ) if context_papers else "  *(no papers retrieved)*"

    return textwrap.dedent(f"""
    **{concept.title()}**

    {definition}

    **Why it matters in CS research:**
    This concept appears frequently in papers on {', '.join(
        p.get('category_label','CS') for p in context_papers[:2]
    ) or 'machine learning and AI'}.

    **Related papers in the index:**
    {papers_md}
    """).strip()


def generate_chat_response(
    query:   str,
    results: list[dict],
    history: list[dict],
) -> str:
    """
    Generate a conversational response given retrieved papers and chat history.
    Handles follow-up questions using history context.
    """
    intent = classify_intent(query)

    # Detect follow-up patterns
    followup_signals = re.search(
        r"\b(more|elaborate|detail|explain further|what about|how does|why|example)\b",
        query.lower()
    )
    is_followup = bool(followup_signals) and len(history) > 0

    if not results:
        return (
            "I couldn't find papers closely matching your query in the indexed dataset. "
            "Try rephrasing, using different keywords, or broadening the category filter.\n\n"
            "**Tip:** Use terms like *neural network*, *transformer*, *reinforcement learning*, etc."
        )

    top = results[0]
    title    = top.get("title", "")
    abstract = top.get("abstract", "")
    authors  = _format_authors(top.get("authors", "Unknown"))
    year     = top.get("year", "")
    url      = top.get("url", "")

    # If LLM available, use it for follow-ups
    if config.USE_LLM and _ollama_available() and is_followup:
        history_txt = "\n".join(
            f"{m['role'].title()}: {m['content'][:200]}"
            for m in history[-4:]
        )
        prompt = (
            f"Conversation so far:\n{history_txt}\n\n"
            f"User follow-up: {query}\n\n"
            f"Most relevant paper: {title} — {abstract[:500]}\n\n"
            "Continue the conversation naturally, answering the follow-up "
            "using the paper context where relevant."
        )
        resp = _ollama_generate(prompt)
        if resp:
            return resp

    # Template response
    summary  = summarise(abstract, n_sentences=2)
    concepts = extract_concepts(abstract)[:3]

    if is_followup:
        intro = f"Building on our discussion — here's more detail from a related paper:"
    elif intent == "recent":
        intro = f"Here's the most recent relevant work I found ({year}):"
    elif intent == "explain":
        intro = f"Let me explain based on the research:"
    else:
        intro = f"Here's the most relevant paper for your query:"

    concept_lines = ""
    if concepts:
        concept_lines = "\n\n**Concepts in this paper:**\n" + "\n".join(
            f"  • **{c.term}** — {c.definition}" for c in concepts
        )

    other_titles = ""
    if len(results) > 1:
        others = "\n".join(
            f"  {i+2}. *{r.get('title','')}* ({r.get('year','')})"
            for i, r in enumerate(results[1:4])
        )
        other_titles = f"\n\n**Other relevant papers:**\n{others}"

    return (
        f"{intro}\n\n"
        f"**{title}**\n"
        f"*{authors} · {year}*\n\n"
        f"{summary}"
        f"{concept_lines}"
        f"{other_titles}\n\n"
        f"🔗 [Read on arXiv]({url})"
    )
