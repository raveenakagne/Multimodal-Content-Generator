import streamlit as st, time, csv, os, hashlib, fitz
from functools import lru_cache
from utils.session import init_sidebar
from inference import generate

init_sidebar()
st.title("Blog Generator")

@lru_cache(maxsize=128)
def cached_generate(p,m): return generate(p,m,700)

def short_hash(t): return hashlib.sha256(t.encode()).hexdigest()[:8]

tone   = st.text_input("Tone (optional)")
topic  = st.text_area("Topic or paste content")
pdf    = st.file_uploader("…or upload PDF", type=["pdf"])

if pdf:
    with fitz.open(stream=pdf.read(),filetype="pdf") as d:
        topic = "".join(p.get_text() for p in d)[:1500]
        st.success("PDF text loaded.")

if st.button("Generate blog") and topic.strip():
    prompt = (f"Write a {tone or 'neutral'} ~700‑word blog with 3‑5 "
              f"sub‑headers on:\n{topic.strip()}")
    art, lat = cached_generate(prompt, st.session_state.model_key)
    st.caption(f"{lat:.1f}s · {st.session_state.model_key}")
    st.markdown("---"); st.markdown(art)
