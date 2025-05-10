# podcast.py
# ------------------------------------------------------------------
# Streamlit UI · Podcast snippet generator
#   1) Topic -> LLM script (~220‑260 words, ≈2 min)
#   2) Script -> MP3 via Deepgram TTS
#
# Requirements:
#   pip install streamlit requests
# ------------------------------------------------------------------

import os, time, tempfile, re, requests
from functools import lru_cache
from typing import Tuple

import streamlit as st
from inference import generate, available_models

# ─── secrets ────────────────────────────────────────────────────
try:
    import creds
except ImportError:
    creds = None

DG_KEY = getattr(creds, "DEEPGRAM_API_KEY", None) or os.getenv("DEEPGRAM_API_KEY")
if not DG_KEY:
    st.error("Deepgram API key missing – add to creds.py or set DEEPGRAM_API_KEY.")
    st.stop()

# ─── helpers ────────────────────────────────────────────────────
def clean_script(raw: str) -> str:
    """Remove meta‑text, markdown, and stage directions before TTS."""
    raw = re.sub(r"(?i)^here[’']?s a script[^\n]*\n?", "", raw).strip()
    raw = re.sub(r"\*\*", "", raw)                       # strip markdown bold
    raw = re.sub(r"^\[.*?\]$", "", raw, flags=re.MULTILINE)  # stage cues
    return re.sub(r"[ ]{2,}", " ", raw).strip()

@lru_cache(maxsize=64)
def tts_deepgram(text: str) -> Tuple[str, float]:
    url = "https://api.deepgram.com/v1/speak"
    headers = {"Authorization": f"Token {DG_KEY}", "Content-Type": "application/json"}
    payload = {"text": text}

    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    if r.status_code != 200:
        st.error(f"TTS error {r.status_code}: {r.text}")
        r.raise_for_status()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(r.content)
    tmp.close()
    return tmp.name, time.time() - t0

# ─── Streamlit UI ───────────────────────────────────────────────
st.set_page_config(page_title="Podcast Snippet Generator", layout="centered")
st.title("Podcast Snippet Generator · LLM + Deepgram")

with st.sidebar:
    model_key = st.selectbox("LLM for script", available_models())

topic = st.text_area("Enter podcast topic")

# Step 1 – generate script
if st.button("Generate Script", key="btn_script") and topic.strip():
    prompt = (
        "Write only the spoken lines of a podcast script (~220‑260 words, friendly host tone). "
        "Do NOT include markdown, stage directions, or a header like 'Here's a script'. "
        "Begin directly with the host’s first line. Provide an opening hook, 2‑3 informative points, and a concise closing.\n\n"
        f"Topic: {topic.strip()}"
    )
    with st.spinner("LLM creating script…"):
        script, t_script = generate(prompt, model_key)

    st.session_state["script"] = script
    st.text_area("Podcast Script (editable)", script, height=220, key="script_box")
    st.caption(f"Script latency {t_script:.1f}s · Model {model_key}")

# Retrieve script for audio generation
script_raw = st.session_state.get("script", "")
script_clean = clean_script(script_raw)

# Step 2 – generate audio
if st.button("Generate Audio", key="btn_audio") and script_clean:
    with st.spinner("Deepgram synthesising audio…"):
        audio_path, t_audio = tts_deepgram(script_clean)

    audio_bytes = open(audio_path, "rb").read()
    st.audio(audio_bytes, format="audio/mp3")
    st.download_button("Download MP3", data=audio_bytes,
                       file_name="podcast_snippet.mp3")
    st.caption(f"TTS latency {t_audio:.1f}s")
elif st.button("Generate Audio (no script)", key="btn_audio_warn"):
    st.warning("Generate or paste a script first.")
