"""
visualizations.py — Plotly charts for the arXiv CS Expert Chatbot.

Charts:
  - Category distribution bar chart
  - Papers-per-year timeline
  - Keyword co-occurrence network (concept map)
  - Search result relevance scatter
  - Top keyword bubble chart
"""

from __future__ import annotations
import math
from collections import Counter

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import config

COLORS = px.colors.qualitative.Plotly


# ── 1. Category distribution ──────────────────────────────────────────────────

def category_bar(df: pd.DataFrame) -> go.Figure:
    counts = {}
    for cats in df["categories"]:
        for c in str(cats).split():
            if c in config.CS_CATEGORIES:
                counts[config.CS_CATEGORIES[c]] = counts.get(c, 0) + 1

    # Alternative: use category_label column
    if "category_label" in df.columns:
        counts = df["category_label"].value_counts().to_dict()

    labels = list(counts.keys())
    values = list(counts.values())
    order  = sorted(range(len(values)), key=lambda i: values[i], reverse=True)

    fig = go.Figure(go.Bar(
        x=[labels[i] for i in order],
        y=[values[i] for i in order],
        marker_color=COLORS[:len(labels)],
        text=[values[i] for i in order],
        textposition="outside",
    ))
    fig.update_layout(
        title="Papers by CS Sub-Category",
        xaxis_title="Sub-category",
        yaxis_title="Number of papers",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(t=50, b=100),
        xaxis_tickangle=-30,
    )
    return fig


# ── 2. Papers per year timeline ───────────────────────────────────────────────

def papers_per_year(df: pd.DataFrame) -> go.Figure:
    year_counts = (
        df[df["year"].notna()]
        .groupby("year")
        .size()
        .reset_index(name="count")
    )
    year_counts = year_counts[year_counts["year"] >= 1990]

    fig = px.area(
        year_counts, x="year", y="count",
        title="Publications Over Time",
        labels={"year": "Year", "count": "Papers"},
        color_discrete_sequence=["#636EFA"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=320,
    )
    return fig


# ── 3. Keyword bubble chart ───────────────────────────────────────────────────

def keyword_bubbles(keywords: list[str], counts: list[int]) -> go.Figure:
    """Bubble chart of top keywords."""
    if not keywords:
        return go.Figure()

    # Layout keywords in a rough circle
    n = len(keywords)
    angles = [2 * math.pi * i / n for i in range(n)]
    x = [math.cos(a) * (0.5 + counts[i] / max(counts) * 0.5) for i, a in enumerate(angles)]
    y = [math.sin(a) * (0.5 + counts[i] / max(counts) * 0.5) for i, a in enumerate(angles)]

    fig = go.Figure(go.Scatter(
        x=x, y=y,
        mode="markers+text",
        text=keywords,
        textposition="top center",
        marker=dict(
            size=[20 + 40 * (c / max(counts)) for c in counts],
            color=counts,
            colorscale="Blues",
            showscale=False,
            opacity=0.8,
        ),
        hovertext=[f"{kw}: {c}" for kw, c in zip(keywords, counts)],
    ))
    fig.update_layout(
        title="Top Keywords in Results",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=380,
    )
    return fig


# ── 4. Concept network (force-directed simulation approximation) ──────────────

def concept_network(concepts: list[str], paper_title: str) -> go.Figure:
    """Star-topology concept map centred on the paper topic."""
    if not concepts:
        return go.Figure()

    n = len(concepts)
    cx, cy = 0.0, 0.0   # centre

    angles = [2 * math.pi * i / n for i in range(n)]
    nx = [math.cos(a) * 1.5 for a in angles]
    ny = [math.sin(a) * 1.5 for a in angles]

    # Edge traces (centre → concept)
    edge_x, edge_y = [], []
    for x, y in zip(nx, ny):
        edge_x += [cx, x, None]
        edge_y += [cy, y, None]

    edges = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#cccccc", width=1.5),
        hoverinfo="none",
    )

    # Concept nodes
    nodes = go.Scatter(
        x=nx, y=ny,
        mode="markers+text",
        text=concepts,
        textposition="top center",
        marker=dict(size=18, color="#636EFA", opacity=0.85),
        hoverinfo="text",
    )

    # Centre (paper) node
    centre = go.Scatter(
        x=[cx], y=[cy],
        mode="markers+text",
        text=[paper_title[:30] + "…" if len(paper_title) > 30 else paper_title],
        textposition="bottom center",
        marker=dict(size=28, color="#EF553B", symbol="star"),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edges, nodes, centre])
    fig.update_layout(
        title="Concept Map",
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=400,
    )
    return fig


# ── 5. Search relevance bar ───────────────────────────────────────────────────

def relevance_bar(results: list) -> go.Figure:
    """Horizontal bar chart of search result relevance scores."""
    titles = [r.title[:50] + "…" if len(r.title) > 50 else r.title for r in results]
    scores = [r.combined_score for r in results]

    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=titles[::-1],
        orientation="h",
        marker_color="#636EFA",
        text=[f"{s:.2f}" for s in scores[::-1]],
        textposition="inside",
    ))
    fig.update_layout(
        title="Search Result Relevance",
        xaxis_title="Combined Score",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(250, len(results) * 40),
        margin=dict(l=250),
    )
    return fig
