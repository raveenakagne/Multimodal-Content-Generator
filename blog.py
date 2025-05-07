# blog.py
# ------------------------------------------------------------------
# Streamlit UI for blog generation with four LLM back‑ends.
# Requires:  streamlit, PyMuPDF (fitz), inference.py in the same repo
# ------------------------------------------------------------------

import os, time, csv, hashlib
from functools import lru_cache

import streamlit as st
import fitz  # PyMuPDF

from inference import generate, available_models

LOG_PATH = "latency_log.csv"

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
@lru_cache(maxsize=128)
def cached_generate(prompt: str, model_key: str, max_tokens: int = 700):
    """Cache wrapper so repeated prompts don’t re‑hit the endpoint."""
    return generate(prompt, model_key, max_tokens)


def pdf_to_text(file_bytes: bytes, limit: int = 1500) -> str:
    """Extract text from a PDF (first `limit` characters)."""
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        txt = "".join(p.get_text() for p in doc)
    return txt.strip()[:limit]


def log_latency(model_key: str, latency: float, prompt_hash: str):
    """Append latency record to CSV for later analysis."""
    header_needed = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["timestamp", "model", "prompt_hash", "latency_s"])
        w.writerow([time.time(), model_key, prompt_hash, round(latency, 3)])


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


# ──────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Blog Generator", layout="centered")
st.title("Multimodal Project · Blog Generator")

# Back‑end selector
model_key = st.selectbox("Choose LLM back‑end", available_models())

# Tone selector
tone = st.text_input("Desired tone (e.g., informative, persuasive). Leave blank for neutral.")

# Tabs for topic input vs PDF upload
tab_topic, tab_pdf = st.tabs(["Topic text", "Upload PDF"])

topic_text = ""

with tab_topic:
    topic_text = st.text_area("Describe the blog topic (or paste text)", key="topic_input")

with tab_pdf:
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_input")
    if pdf_file:
        try:
            topic_text = pdf_to_text(pdf_file.read())
            st.success("PDF parsed. Extracted text loaded as topic context.")
            st.text_area("Extracted preview (editable)", value=topic_text, key="pdf_preview")
        except Exception as e:
            st.error(f"PDF parsing failed: {e}")

# Generate button
if st.button("Generate blog"):
    if not topic_text.strip():
        st.warning("Please provide topic text or upload a PDF.")
        st.stop()

    prompt = (
        f"Write a {tone.strip() or 'neutral'} blog article of ~700 words.\n"
        f"Include: headline, introduction, 3‑5 subsections with short headers, "
        f"and a concise conclusion.\n\n"
        f"Topic/context:\n'''{topic_text.strip()}'''"
    )

    prompt_tag = short_hash(prompt)

    with st.spinner("Generating…"):
        article, latency = cached_generate(prompt, model_key)

    log_latency(model_key, latency, prompt_tag)

    st.caption(f"Latency {latency:.1f} s · Model {model_key}")
    st.markdown("---")
    st.markdown(article)
