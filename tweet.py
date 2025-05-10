"""
Streamlit · Social Post Generator (X / Threads)
==============================================
Generates a concise social‑media post for either **Twitter/X** (≤ 240 chars) or
**Meta Threads** (≤ 500 chars). LLM back‑ends are abstracted via
**llm_inference.py**, so you can swap models without touching this UI.
 
Run locally
-----------
```bash
pip install -U streamlit fitz requests python-dotenv
streamlit run ui_post_generator.py
```
 
Environment
-----------
* `.env` (or exported env vars) must contain the secrets used in
  *llm_inference.py*: `TOGETHER_API_KEY`, `GCP_PROJECT`, `GCP_REGION`, and
  `GOOGLE_APPLICATION_CREDENTIALS` (for the Vertex service‑account JSON).
"""
from __future__ import annotations
 
import csv
import hashlib
import os
import re
import time
from functools import lru_cache
from typing import List
 
import streamlit as st
from dotenv import load_dotenv
from requests.exceptions import HTTPError
 
from inference import generate, available_models
 
# ─── Config ────────────────────────────────────────────────────
load_dotenv()
LOG_PATH = "latency_log.csv"
PLATFORM_LIMITS = {"twitter": 240, "threads": 500}
MAX_TOKENS = 160  # safe upper bound for either platform
 
# ─── Helpers ───────────────────────────────────────────────────
@lru_cache(maxsize=256)
def cached_generate(prompt: str, model_key: str, max_tokens: int = MAX_TOKENS):
    """Memoised wrapper so identical prompts don’t re‑hit the endpoint."""
    tries, delay = 0, 1.5
    while True:
        try:
            return generate(prompt, model_key, max_tokens)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 429 and tries < 2:
                time.sleep(delay)
                tries += 1
                delay *= 2
                continue
            raise
 
def log_latency(model_key: str, latency: float, prompt_hash: str):
    header_needed = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if header_needed:
            w.writerow(["timestamp", "model", "prompt_hash", "latency_s"])
        w.writerow([time.time(), model_key, prompt_hash, round(latency, 2)])
 
def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]
 
# ─── UI ────────────────────────────────────────────────────────
all_models: List[str] = available_models()
 
st.set_page_config(page_title="Social Post Generator", layout="centered")
st.title("Multimodal Project · Social Post Generator")
 
# Platform tabs instead of dropdown
platform_tabs = st.tabs(["Twitter / X", "Threads"])
platform = "twitter"  # default
with platform_tabs[0]:
    platform = "twitter"
    st.caption("Compose a ≤240‑character tweet")
    model_key = st.selectbox("Choose LLM back‑end", all_models, key="model_tw")
    tone = st.text_input("Tone (playful, newsy, sarcastic…). Leave blank for default", key="tone_tw")
    prompt_topic = st.text_area("What’s the tweet about?", placeholder="e.g. Halloween party theme", key="topic_tw")
    submit = st.button("Generate tweet", key="btn_tw")
 
with platform_tabs[1]:
    platform_threads = "threads"
    st.caption("Compose a ≤500‑character Threads post")
    model_key_th = st.selectbox("Choose LLM back‑end", all_models, key="model_th")
    tone_th = st.text_input("Tone (playful, newsy, sarcastic…). Leave blank for default", key="tone_th")
    prompt_topic_th = st.text_area("What’s the post about?", placeholder="e.g. Halloween party theme", key="topic_th")
    submit_th = st.button("Generate Threads post", key="btn_th")
 
# Determine which submit was clicked
if "submit" not in st.session_state:
    st.session_state.submit = False
if submit:
    st.session_state.submit = True
    platform_sel = "twitter"
    model_sel = model_key
    tone_sel = tone
    topic_sel = prompt_topic
elif submit_th:
    st.session_state.submit = True
    platform_sel = "threads"
    model_sel = model_key_th
    tone_sel = tone_th
    topic_sel = prompt_topic_th
else:
    platform_sel = None
 
if st.session_state.get("submit") and platform_sel:
    limit = PLATFORM_LIMITS[platform_sel]
 
    if not topic_sel.strip():
        st.warning("Please enter a topic.")
    else:
        prompt = (
            "You are an expert social‑media copywriter. Craft a catchy, engaging "
            f"{platform_sel.capitalize()} post under {limit} characters on the topic below. "
            "Add up to four relevant hashtags.\n\n"
            f"Tone: {tone_sel or 'default'}\n"
            f"Topic: {topic_sel.strip()}"
        )
        prompt_tag = short_hash(prompt)
 
        with st.spinner("Generating…"):
            try:
                raw_post, latency = cached_generate(prompt, model_sel)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()
 
        # ── Post‑processing ────────────────────────────────────
        post = raw_post.strip()
 
        # 1. Strip any <think> / reasoning blocks
        post = re.sub(r"(?is)<think[^>]*>.*?(</think>|$)", "", post)
 
        # 2. Drop leading quotes / spaces
        post = post.lstrip('"\u201c\u201d\'\n ').strip()
 
                # 3. Collapse multiple lines → single spacified paragraph
        post = " ".join(line.strip() for line in post.splitlines() if line.strip())
 
        # 4. Auto‑add hashtags if the model forgot them Auto hashtags if missing
        if "#" not in post:
            tags = [f"#{w.lower()}" for w in re.findall(r"[A-Za-z]+", topic_sel)[:4]]
            if tags:
                post = f"{post.rstrip('. ')} {' '.join(tags)}".strip()
 
        # 5. Truncate to platform limit
        if len(post) > limit:
            post = post[: limit - 1].rstrip() + "…"
 
        log_latency(model_sel, latency, prompt_tag)
 
        st.caption(f"Latency {latency:.1f}s · Model {model_sel} · Platform {platform_sel}")
        st.markdown("---")
        st.markdown(post)