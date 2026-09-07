"""
nlp_pipeline.py — NLP utilities for the arXiv CS Expert Chatbot.

Features:
  - Keyword / keyphrase extraction  (TF-IDF + domain vocabulary)
  - Extractive summarisation        (sentence scoring)
  - Concept extraction              (CS domain terms)
  - Query intent classification
"""

from __future__ import annotations
import re
import math
import logging
from collections import Counter
from dataclasses import dataclass, field

import nltk
import pandas as pd

# Lazy NLTK downloads
for resource in ("stopwords", "punkt_tab", "averaged_perceptron_tagger_eng"):
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource
                       else f"taggers/{resource}" if "tagger" in resource
                       else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)
STOP = set(stopwords.words("english"))

# ── CS domain term vocabulary ─────────────────────────────────────────────────

CS_TERMS: dict[str, str] = {
    # ML / DL
    "neural network":        "A computing system loosely inspired by the brain's biological neural networks.",
    "deep learning":         "ML using neural networks with many layers to learn data representations.",
    "transformer":           "Attention-based architecture that revolutionised NLP (BERT, GPT, etc.).",
    "attention mechanism":   "Allows models to focus on relevant parts of the input dynamically.",
    "gradient descent":      "Optimisation algorithm that iteratively minimises a loss function.",
    "backpropagation":       "Algorithm to compute gradients for training neural networks.",
    "overfitting":           "When a model learns noise in training data and generalises poorly.",
    "regularisation":        "Techniques (dropout, L1/L2) to prevent overfitting.",
    "batch normalisation":   "Normalises layer inputs to stabilise and speed up training.",
    "convolutional network": "CNN — uses convolution filters for spatial feature extraction.",
    "recurrent network":     "RNN — processes sequential data with hidden state memory.",
    "lstm":                  "Long Short-Term Memory — RNN that handles long-range dependencies.",
    "generative adversarial network": "GAN — generator vs discriminator trained adversarially.",
    "variational autoencoder": "VAE — generative model that learns a latent space distribution.",
    "reinforcement learning": "Agent learns by interacting with an environment via rewards.",
    "transfer learning":     "Reusing a model trained on one task as a starting point for another.",
    "fine-tuning":           "Adapting a pre-trained model on a smaller domain-specific dataset.",
    "zero-shot learning":    "Model generalises to unseen classes without task-specific training.",
    "few-shot learning":     "Model learns from very few labelled examples.",
    "contrastive learning":  "Learns representations by contrasting similar and dissimilar pairs.",
    "self-supervised":       "Learning from unlabelled data using pretext tasks.",
    "knowledge distillation":"Transferring knowledge from a large model to a smaller one.",
    # NLP
    "tokenisation":          "Splitting text into tokens (words or sub-words) for model input.",
    "embedding":             "Dense vector representation of words or sentences.",
    "bert":                  "Bidirectional Encoder Representations from Transformers — pre-trained LM.",
    "gpt":                   "Generative Pre-trained Transformer — autoregressive language model.",
    "large language model":  "Massive neural LM trained on internet-scale text (GPT-4, LLaMA, etc.).",
    "named entity recognition": "NER — identifying named entities (people, places, orgs) in text.",
    "sentiment analysis":    "Classifying the emotional tone of text.",
    # CV
    "object detection":      "Locating and classifying objects within images.",
    "image segmentation":    "Assigning a label to every pixel of an image.",
    "feature extraction":    "Deriving informative representations from raw input data.",
    # General CS
    "algorithm":             "Step-by-step procedure for solving a computational problem.",
    "complexity":            "Measures of computational resources (time/space) an algorithm needs.",
    "graph neural network":  "GNN — neural network that operates directly on graph-structured data.",
    "federated learning":    "Training models across decentralised devices without sharing raw data.",
    "differential privacy":  "Adding noise to protect individual data privacy during learning.",
    "explainability":        "Methods to interpret and understand model predictions.",
    "benchmark":             "Standardised task/dataset used to compare model performance.",
    "hyperparameter":        "Configuration value set before training (learning rate, batch size, etc.).",
}


# ── Intent classification ─────────────────────────────────────────────────────

INTENT_PATTERNS: dict[str, list[str]] = {
    "explain":    [r"\b(what is|what are|explain|define|describe|meaning of|tell me about)\b"],
    "summarise":  [r"\b(summarize|summarise|summary|tldr|overview|briefly)\b"],
    "search":     [r"\b(find|search|show|list|papers? on|research on|articles? about)\b"],
    "compare":    [r"\b(compare|difference|versus|vs\.?|better|pros and cons)\b"],
    "recent":     [r"\b(latest|recent|new|2023|2024|2025|current|state.?of.?the.?art)\b"],
    "author":     [r"\b(author|written by|who wrote|papers? by)\b"],
}


def classify_intent(query: str) -> str:
    q = query.lower()
    for intent, patterns in INTENT_PATTERNS.items():
        for p in patterns:
            if re.search(p, q):
                return intent
    return "search"


# ── Keyword extraction ─────────────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """TF-IDF-style keyword extraction from a single document."""
    words = [
        w.lower() for w in word_tokenize(text)
        if w.isalpha() and w.lower() not in STOP and len(w) > 2
    ]
    freq = Counter(words)
    # Boost CS domain terms
    scored = {}
    for w, cnt in freq.items():
        boost = 2.0 if any(w in term for term in CS_TERMS) else 1.0
        scored[w] = cnt * boost

    # Also extract bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    for bg in bigrams:
        if bg in CS_TERMS:
            scored[bg] = scored.get(bg, 0) + 3.0

    top = sorted(scored, key=lambda k: scored[k], reverse=True)[:top_n]
    return top


# ── Extractive summarisation ──────────────────────────────────────────────────

def summarise(text: str, n_sentences: int = 3) -> str:
    """Score sentences by keyword density and return the top n."""
    sentences = sent_tokenize(text)
    if len(sentences) <= n_sentences:
        return text

    keywords = set(extract_keywords(text, top_n=15))
    scores: list[float] = []

    for i, sent in enumerate(sentences):
        words = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
        # keyword coverage
        kw_score  = sum(1 for w in words if w in keywords) / max(len(words), 1)
        # position bonus (first and last sentences are often key)
        pos_score = 1.0 if i == 0 else 0.5 if i == len(sentences)-1 else 0.0
        # length penalty (very short sentences carry less info)
        len_score = min(len(words) / 20, 1.0)
        scores.append(kw_score + 0.3 * pos_score + 0.1 * len_score)

    top_idx = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:n_sentences]
    top_idx.sort()   # restore original order
    return " ".join(sentences[i] for i in top_idx)


# ── Concept extraction ────────────────────────────────────────────────────────

@dataclass
class Concept:
    term:        str
    definition:  str
    frequency:   int = 0


def extract_concepts(text: str) -> list[Concept]:
    """Find CS domain terms mentioned in text."""
    text_lower = text.lower()
    found: list[Concept] = []
    seen: set[str] = set()

    for term, definition in sorted(CS_TERMS.items(), key=lambda x: len(x[0]), reverse=True):
        if term in text_lower and term not in seen:
            freq = text_lower.count(term)
            found.append(Concept(term=term, definition=definition, frequency=freq))
            seen.add(term)

    return sorted(found, key=lambda c: c.frequency, reverse=True)


# ── Corpus-level TF-IDF keywords ─────────────────────────────────────────────

def build_corpus_tfidf(df: pd.DataFrame) -> dict[int, list[str]]:
    """
    Compute per-paper keywords using corpus-level IDF.
    Returns {row_index: [keywords]}
    """
    # Build vocab
    doc_words: list[list[str]] = []
    for text in (df["title"] + " " + df["abstract"]):
        words = [
            w.lower() for w in word_tokenize(str(text))
            if w.isalpha() and w.lower() not in STOP and len(w) > 2
        ]
        doc_words.append(words)

    N = len(doc_words)
    df_counts: Counter = Counter()
    for words in doc_words:
        df_counts.update(set(words))

    result = {}
    for i, words in enumerate(doc_words):
        tf = Counter(words)
        scored = {
            w: (cnt / len(words)) * math.log((N + 1) / (df_counts[w] + 1))
            for w, cnt in tf.items()
        }
        top = sorted(scored, key=lambda k: scored[k], reverse=True)[:8]
        result[i] = top

    return result
