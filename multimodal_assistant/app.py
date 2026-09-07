"""
Streamlit demo UI for the Multi-Modal AI Assistant.

Run with:
    streamlit run app.py

Lets you chat with the assistant, optionally attaching an image to
any message, and shows the internal decision trace (ambiguity check,
evidence extraction, validation outcome) so the pipeline's reasoning
is visible -- not just the final answer.
"""

import tempfile
from pathlib import Path

import streamlit as st

from core.orchestrator import MultimodalAssistant

st.set_page_config(page_title="Multi-Modal AI Assistant", page_icon="🖼️", layout="wide")
st.title("🖼️ Multi-Modal AI Assistant")
st.caption(
    "Text + image chat with contextual memory, ambiguity handling, "
    "and evidence-based response validation."
)

if "assistant" not in st.session_state:
    try:
        st.session_state.assistant = MultimodalAssistant()
        st.session_state.init_error = None
    except EnvironmentError as e:
        st.session_state.assistant = None
        st.session_state.init_error = str(e)

if "display_history" not in st.session_state:
    st.session_state.display_history = []  # list of dicts for rendering

if st.session_state.init_error:
    st.error(st.session_state.init_error)
    st.stop()

# --- Render prior turns ---
for item in st.session_state.display_history:
    with st.chat_message(item["role"]):
        if item.get("image"):
            st.image(item["image"], width=240)
        st.write(item["text"])
        if item.get("trace"):
            with st.expander("Show reasoning trace"):
                for line in item["trace"]:
                    st.markdown(f"- {line}")
                if item.get("unsupported_claims"):
                    st.warning(f"Corrected unsupported claim(s): {item['unsupported_claims']}")

# --- Input area ---
uploaded_image = st.file_uploader("Attach an image (optional)", type=["png", "jpg", "jpeg", "webp"])
user_message = st.chat_input("Ask something...")

if user_message:
    image_path = None
    image_display = None
    if uploaded_image is not None:
        suffix = Path(uploaded_image.name).suffix or ".png"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(uploaded_image.getbuffer())
        tmp.close()
        image_path = tmp.name
        image_display = uploaded_image

    st.session_state.display_history.append(
        {"role": "user", "text": user_message, "image": image_display}
    )

    with st.spinner("Thinking..."):
        result = st.session_state.assistant.process(user_message, image_path=image_path)

    st.session_state.display_history.append(
        {
            "role": "assistant",
            "text": result.answer,
            "trace": result.trace,
            "unsupported_claims": result.unsupported_claims,
        }
    )
    st.rerun()
