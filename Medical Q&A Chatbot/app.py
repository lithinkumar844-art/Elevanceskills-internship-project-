"""
app.py — Streamlit UI for the MedQuAD Medical Q&A Chatbot.

Run with:
    streamlit run app.py
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Q&A Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ────────────────────────────────────────────────────────────────────
from data_loader       import load_dataset
from retrieval         import RetrievalEngine
from entity_recognition import (
    recognise_entities, highlight_entities, entity_summary, ENTITY_COLORS
)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "medquad_raw"


# ── Load & cache model ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading MedQuAD dataset and building index…")
def load_engine():
    df     = load_dataset(DATA_DIR)
    engine = RetrievalEngine()
    engine.build(df, use_cache=True)
    return engine, df


engine, df = load_engine()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Caduceus.svg/240px-Caduceus.svg.png", width=60)
    st.title("🏥 MedQuAD\nQ&A Chatbot")
    st.caption(f"**{engine.size:,}** QA pairs indexed")
    st.divider()

    top_k = st.slider("Results to retrieve", min_value=1, max_value=10, value=3)
    min_score = st.slider("Min. relevance score", min_value=0.0, max_value=1.0,
                          value=0.10, step=0.01)

    show_entities = st.toggle("Highlight medical entities", value=True)
    show_all_results = st.toggle("Show all retrieved results", value=False)

    st.divider()
    st.markdown("**Question types in dataset**")
    if "qtype" in df.columns:
        qtypes = df["qtype"].value_counts().head(8)
        for qt, cnt in qtypes.items():
            st.caption(f"• {qt} ({cnt:,})")

    st.divider()
    st.markdown("**Entity legend**")
    for label, color in ENTITY_COLORS.items():
        st.markdown(
            f'<span style="background:{color};padding:2px 8px;border-radius:3px;'
            f'color:white;font-size:0.8em;font-weight:600">{label}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.caption("⚠️ For informational purposes only. Always consult a healthcare professional.")


# ── Main layout ────────────────────────────────────────────────────────────────
st.title("🏥 Medical Question & Answer Chatbot")
st.caption("Powered by the MedQuAD dataset — 47,457 QA pairs from 12 NIH websites")

# Example questions
EXAMPLE_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "How is leukemia diagnosed?",
    "What treatments are available for Parkinson's disease?",
    "What causes hypertension?",
    "How can I prevent heart disease?",
    "What are the side effects of chemotherapy?",
    "What is multiple sclerosis?",
    "How is asthma treated?",
]

st.markdown("**Try an example question:**")
cols = st.columns(4)
for i, eq in enumerate(EXAMPLE_QUESTIONS):
    if cols[i % 4].button(eq, use_container_width=True):
        st.session_state["prefill_query"] = eq

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            st.markdown(msg["content"], unsafe_allow_html=True)

# Handle pre-fill from example buttons
prefill = st.session_state.pop("prefill_query", None)

# Query input
query = st.chat_input("Ask a medical question…", key="chat_input")
if prefill and not query:
    query = prefill

if query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # ── Entity recognition on query ────────────────────────────────────────
    query_entities = recognise_entities(query)

    # ── Retrieve answers ───────────────────────────────────────────────────
    results = engine.search(query, top_k=top_k)
    results = [r for r in results if r.score >= min_score]

    # ── Build response ─────────────────────────────────────────────────────
    with st.chat_message("assistant"):
        if not results:
            no_ans = (
                "😔 I couldn't find a relevant answer in the MedQuAD dataset "
                "for your question. Please try rephrasing, or consult a "
                "healthcare professional."
            )
            st.warning(no_ans)
            st.session_state.messages.append({"role": "assistant", "content": no_ans})

        else:
            # ── Entity summary for query ───────────────────────────────────
            if show_entities and query_entities:
                ent_sum = entity_summary(query_entities)
                pills = []
                for label, terms in ent_sum.items():
                    color = ENTITY_COLORS[label]
                    for t in terms:
                        pills.append(
                            f'<span style="background:{color};padding:2px 8px;'
                            f'border-radius:12px;color:white;font-size:0.8em;'
                            f'margin:2px;display:inline-block">'
                            f'<b>{label}:</b> {t}</span>'
                        )
                if pills:
                    st.markdown(
                        "**Entities detected in your question:** " + " ".join(pills),
                        unsafe_allow_html=True,
                    )
                    st.divider()

            # ── Best answer ────────────────────────────────────────────────
            best = results[0]

            st.markdown(f"### 📋 {best.focus}")
            st.caption(
                f"**Type:** {best.qtype.title()} &nbsp;|&nbsp; "
                f"**Source:** {best.source} &nbsp;|&nbsp; "
                f"**Relevance:** {best.score:.0%}"
            )

            # Matched question
            with st.expander("📌 Matched question", expanded=False):
                st.markdown(f"*{best.question}*")

            # Answer with entity highlighting
            answer_text = best.answer
            if show_entities:
                answer_entities = recognise_entities(answer_text)
                highlighted = highlight_entities(answer_text, answer_entities)
                st.markdown(highlighted, unsafe_allow_html=True)

                if answer_entities:
                    ent_sum2 = entity_summary(answer_entities)
                    with st.expander("🔬 Entities found in answer"):
                        for label, terms in ent_sum2.items():
                            color = ENTITY_COLORS[label]
                            st.markdown(
                                f'**{label.title()}s:** ' + ", ".join(
                                    f'<span style="background:{color};padding:1px 6px;'
                                    f'border-radius:8px;color:white;font-size:0.85em">{t}</span>'
                                    for t in terms
                                ),
                                unsafe_allow_html=True,
                            )
            else:
                st.markdown(answer_text)

            # Source link
            if best.url:
                st.markdown(f"📎 [Read more on {best.source}]({best.url})")

            # ── Additional results ─────────────────────────────────────────
            if show_all_results and len(results) > 1:
                st.divider()
                st.markdown("#### 🔍 Other relevant results")
                for r in results[1:]:
                    with st.expander(f"[{r.score:.0%}] {r.focus} — {r.qtype.title()}"):
                        st.caption(f"*{r.question}*")
                        st.markdown(r.answer[:600] + ("…" if len(r.answer) > 600 else ""))
                        if r.url:
                            st.markdown(f"[Source]({r.url})")

            # Save to history (plain text version)
            plain = f"**{best.focus}** ({best.qtype})\n\n{best.answer[:800]}…"
            st.session_state.messages.append({"role": "assistant", "content": plain})

    # ── Medical disclaimer ─────────────────────────────────────────────────
    st.info(
        "⚕️ **Disclaimer:** This chatbot provides general health information from "
        "NIH datasets. It is not a substitute for professional medical advice, "
        "diagnosis, or treatment.",
        icon="⚠️"
    )

# ── Empty state ────────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 What I can do")
        st.markdown("""
- Answer medical questions from NIH datasets
- Explain symptoms, causes & treatments
- Identify diseases and conditions
- Retrieve drug information
        """)
    with col2:
        st.markdown("### 🏷️ Medical entities I detect")
        st.markdown("""
- 🔴 **Symptoms** (fever, pain, fatigue…)
- 🟦 **Diseases** (cancer, diabetes…)
- 🔵 **Treatments** (surgery, chemo…)
- 🟢 **Drugs** (aspirin, insulin…)
- 🟣 **Body parts** (heart, brain…)
- 🟠 **Tests** (MRI, biopsy, CBC…)
        """)
    with col3:
        st.markdown("### 📚 Data sources")
        st.markdown("""
- CancerGov (NIH)
- NINDS (neurology)
- NIDDK (diabetes/digestive)
- CDC
- MedlinePlus
- GARD (rare diseases)
        """)
