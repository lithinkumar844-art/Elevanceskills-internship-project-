"""
app.py — Streamlit UI for the arXiv CS Expert Chatbot.

Run:   streamlit run app.py
"""

from __future__ import annotations
import streamlit as st
from collections import Counter

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="arXiv CS Expert Chatbot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from data_loader   import load_papers
from retrieval     import RetrievalEngine
from nlp_pipeline  import (
    summarise, extract_keywords, extract_concepts,
    classify_intent, CS_TERMS
)
from llm_interface import (
    generate_paper_explanation,
    generate_concept_explanation,
    generate_chat_response,
)
from visualizations import (
    category_bar, papers_per_year,
    keyword_bubbles, concept_network, relevance_bar
)
import config


# ── Load & index ───────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="📚 Loading arXiv CS papers and building search index…")
def init_engine():
    df     = load_papers()
    engine = RetrievalEngine()
    engine.build(df, use_cache=True)
    return engine


engine = init_engine()
df     = engine.df
if df is None:
    raise RuntimeError("Retrieval engine did not load a paper dataset")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 arXiv CS Expert")
    st.caption(f"**{engine.size:,}** papers indexed")
    st.divider()

    st.markdown("### 🔍 Search filters")
    all_cats = list(config.CS_CATEGORIES.values())
    selected_cats = st.multiselect(
        "CS sub-categories",
        options=all_cats,
        default=all_cats[:3],
        help="Filter results to specific sub-fields",
    )
    cat_keys = [
        k for k, v in config.CS_CATEGORIES.items()
        if v in selected_cats
    ]

    years = df["year"].dropna().astype(int)
    y_min, y_max = int(years.min()), int(years.max())
    if y_min == y_max:
        y_min = y_min - 10  # prevent equal min/max crash
    year_range = st.slider("Year range", y_min, y_max, (max(y_min, 2013), y_max))
    top_k = st.slider("Results per query", 3, 15, 5)

    st.divider()
    st.markdown("### ⚙️ LLM settings")
    use_llm = st.toggle(
        "Use local LLM (Ollama)",
        value=False,
        help="Requires Ollama running locally with a model like Mistral",
    )
    if use_llm:
        ollama_model = st.text_input("Model name", value="mistral")
        config.OLLAMA_MODEL = ollama_model
    config.USE_LLM = use_llm

    st.divider()
    st.markdown("### 📖 CS concept glossary")
    chosen_concept = st.selectbox(
        "Look up a concept",
        options=["— select —"] + sorted(CS_TERMS.keys()),
    )

    st.divider()
    st.caption("⚠️ For research & educational purposes only.")


# ── Main layout ────────────────────────────────────────────────────────────────
st.title("🔬 arXiv CS Expert Chatbot")
st.caption("Powered by the arXiv dataset · Hybrid BM25 + TF-IDF retrieval · NLP summarisation")

tab_chat, tab_search, tab_analytics, tab_concepts = st.tabs(
    ["💬 Chat", "🔍 Paper Search", "📊 Analytics", "🧠 Concepts"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    # Example prompts
    examples = [
        "What is a transformer in NLP?",
        "Summarize recent work on federated learning",
        "Explain attention mechanisms",
        "Find papers on graph neural networks",
        "What is the latest work on object detection?",
        "How does reinforcement learning work?",
    ]
    st.markdown("**Quick start:**")
    ex_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if ex_cols[i % 3].button(ex, use_container_width=True, key=f"ex_{i}"):
            st.session_state["prefill"] = ex

    st.divider()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    prefill = st.session_state.pop("prefill", None)
    query   = st.chat_input("Ask about CS research…")
    if prefill and not query:
        query = prefill

    if query:
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                results = engine.search(
                    query,
                    top_k=top_k,
                    year_min=year_range[0],
                    year_max=year_range[1],
                    categories=cat_keys if cat_keys else None,
                )
                result_dicts = [
                    {"title": r.title, "abstract": r.abstract,
                     "authors": r.authors, "year": r.year,
                     "url": r.url, "category_label": r.category_label}
                    for r in results
                ]
                response = generate_chat_response(
                    query, result_dicts, st.session_state.messages
                )

            st.markdown(response, unsafe_allow_html=True)

            # Show concept map for top result
            if results:
                top_concepts = [c.term for c in extract_concepts(results[0].abstract)[:6]]
                if top_concepts:
                    with st.expander("🗺️ Concept map for top result"):
                        st.plotly_chart(
                            concept_network(top_concepts, results[0].title),
                            use_container_width=True,
                        )

        st.session_state.messages.append({"role": "assistant", "content": response})

    if st.session_state.messages:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PAPER SEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_search:
    col_search, col_opts = st.columns([3, 1])
    with col_search:
        search_query = st.text_input(
            "Search papers",
            placeholder="e.g. 'attention is all you need' or 'BERT language model'",
            key="paper_search",
        )
    with col_opts:
        sort_by = st.selectbox("Sort by", ["Relevance", "Year (newest)", "Year (oldest)"])

    if search_query:
        with st.spinner("Searching…"):
            s_results = engine.search(
                search_query,
                top_k=top_k,
                year_min=year_range[0],
                year_max=year_range[1],
                categories=cat_keys if cat_keys else None,
            )

        if sort_by == "Year (newest)":
            s_results.sort(key=lambda r: r.year or 0, reverse=True)
        elif sort_by == "Year (oldest)":
            s_results.sort(key=lambda r: r.year or 9999)

        if not s_results:
            st.warning("No papers found. Try broader keywords or remove filters.")
        else:
            st.success(f"Found **{len(s_results)}** papers")

            # Relevance chart
            st.plotly_chart(relevance_bar(s_results), use_container_width=True)

            # Keyword bubbles from results
            all_kws: list[str] = []
            for r in s_results:
                all_kws.extend(extract_keywords(r.abstract, top_n=5))
            kw_counts = Counter(all_kws).most_common(12)
            if kw_counts:
                kws, cnts = zip(*kw_counts)
                st.plotly_chart(
                    keyword_bubbles(list(kws), list(cnts)),
                    use_container_width=True,
                )

            st.divider()

            # Paper cards
            for i, r in enumerate(s_results):
                with st.expander(
                    f"**{i+1}.** {r.title}  "
                    f"*· {r.year or 'N/A'} · {r.category_label} · score: {r.combined_score:.2f}*"
                ):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.markdown(f"**Authors:** {r.authors[:120]}")
                        st.markdown(f"**Summary:**")
                        st.markdown(summarise(r.abstract, n_sentences=3))
                    with col_b:
                        concepts = extract_concepts(r.abstract)[:4]
                        if concepts:
                            st.markdown("**Key concepts:**")
                            for c in concepts:
                                st.markdown(
                                    f'<span style="background:#636EFA22;padding:2px 6px;'
                                    f'border-radius:4px;font-size:0.85em">{c.term}</span>',
                                    unsafe_allow_html=True,
                                )

                    if st.button("📝 Full explanation", key=f"exp_{i}"):
                        explanation = generate_paper_explanation(
                            {"title": r.title, "abstract": r.abstract,
                             "authors": r.authors, "year": r.year,
                             "url": r.url, "category_label": r.category_label},
                            search_query,
                        )
                        st.markdown(explanation)

                    if r.url:
                        st.markdown(f"🔗 [Open on arXiv]({r.url})")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("### 📊 Dataset analytics")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total papers", f"{len(df):,}")
    col2.metric("Sub-categories", df["category_label"].nunique())
    col3.metric("Year range", f"{int(df['year'].min())}–{int(df['year'].max())}")
    col4.metric("Avg abstract length",
                f"{int(df['abstract'].str.len().mean())} chars")

    st.plotly_chart(category_bar(df), use_container_width=True)
    st.plotly_chart(papers_per_year(df), use_container_width=True)

    st.markdown("### 🔑 Most common keywords across all papers")
    with st.spinner("Computing keywords…"):
        sample = df.sample(min(500, len(df)), random_state=42)
        all_kws: list[str] = []
        for _, row in sample.iterrows():
            all_kws.extend(extract_keywords(row["abstract"], top_n=5))
        top_kws = Counter(all_kws).most_common(16)
        if top_kws:
            kws, cnts = zip(*top_kws)
            st.plotly_chart(
                keyword_bubbles(list(kws), list(cnts)),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONCEPTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_concepts:
    st.markdown("### 🧠 CS Concept Explorer")
    st.caption(f"Glossary of **{len(CS_TERMS)}** key computer science concepts with research context")

    # From sidebar selector
    if chosen_concept != "— select —":
        defn = CS_TERMS[chosen_concept]
        related = engine.search(chosen_concept, top_k=3)
        related_dicts = [
            {"title": r.title, "year": r.year, "category_label": r.category_label}
            for r in related
        ]
        explanation = generate_concept_explanation(chosen_concept, defn, related_dicts)
        st.markdown(f"## `{chosen_concept}`")
        st.markdown(explanation)

        if related:
            st.markdown("#### 📄 Papers mentioning this concept")
            for r in related:
                st.markdown(f"- [{r.title}]({r.url}) · *{r.year}*")

            top_concepts = [c.term for c in extract_concepts(related[0].abstract)[:6]
                            if c.term != chosen_concept]
            if top_concepts:
                st.plotly_chart(
                    concept_network(top_concepts, chosen_concept),
                    use_container_width=True,
                )

        st.divider()

    # Browse all concepts
    st.markdown("### 📚 Browse all concepts")
    search_concept = st.text_input("Filter concepts", placeholder="e.g. attention, gradient")
    filtered = {
        k: v for k, v in CS_TERMS.items()
        if not search_concept or search_concept.lower() in k.lower()
    }

    cols = st.columns(2)
    for i, (term, defn) in enumerate(sorted(filtered.items())):
        with cols[i % 2]:
            with st.expander(f"**{term}**"):
                st.markdown(defn)
                if st.button("Find papers →", key=f"cp_{term}"):
                    st.session_state["prefill"] = f"Find papers on {term}"
                    st.rerun()
