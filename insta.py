import os, time, base64, hashlib, requests
from functools import lru_cache
from typing import Tuple

import streamlit as st
from PIL import Image
from inference import generate, available_models

# ─── Secrets ───
try:
    import creds
except ImportError:
    creds = None

HF_TOKEN = getattr(creds, "hf_token", None) or os.getenv("HF_TOKEN")
if not HF_TOKEN:
    st.error("HF_TOKEN required for BLIP image captioning.")
    st.stop()

STABILITY_KEY = getattr(creds, "stability_key", None) or os.getenv("STABILITY_KEY")
if not STABILITY_KEY:
    st.error("STABILITY_KEY required for image generation.")
    st.stop()

# ─── Image generation using Stability SD3 ───
@lru_cache(maxsize=64)
def prompt_to_image(prompt: str) -> Tuple[bytes, float]:
    import json
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {
        "Authorization": f"Bearer {STABILITY_KEY}",
        "Accept": "application/json"
    }
    files = {
        "prompt": (None, json.dumps([{"text": prompt}])),
        "mode": (None, "text-to-image"),
        "output_format": (None, "png"),
        "aspect_ratio": (None, "1:1"),
    }

    t0 = time.time()
    r = requests.post(url, headers=headers, files=files, timeout=90)
    if not r.ok:
        st.error(f"Image API error {r.status_code}: {r.text}")
        r.raise_for_status()
    b64 = r.json()["image"]
    return base64.b64decode(b64), time.time() - t0

# ─── Vision model (BLIP) for raw image → text caption ───
@lru_cache(maxsize=128)
def raw_img_to_text(img_bytes: bytes) -> Tuple[str, float]:
    url = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    files = {"file": img_bytes}
    t0 = time.time()
    r = requests.post(url, headers=headers, files=files, timeout=90)
    if not r.ok:
        st.error(f"Captioning error {r.status_code}: {r.text}")
        r.raise_for_status()
    return r.json()[0]["generated_text"], time.time() - t0

# ─── Streamlit UI ───
st.set_page_config(page_title="Instagram Generator", layout="centered")
st.title("Instagram Content Generator · SD3 + LLM")

mode = st.radio("Select mode", ["Prompt → Image + Caption", "Image → Caption + Hashtags"],
                horizontal=True)

with st.sidebar:
    model_key = st.selectbox("Text LLM for captions", available_models())

# ─── Mode A: Prompt → Image + Caption ───
if mode.startswith("Prompt"):
    prompt_text = st.text_area("Describe the image concept")
    if st.button("Generate", key="btn_prompt") and prompt_text.strip():
        with st.spinner("Generating image…"):
            img_bytes, t_img = prompt_to_image(prompt_text.strip())
        st.image(img_bytes, caption=f"SD3 · {t_img:.1f}s")

        caption_prompt = (
            f"You are an Instagram content strategist.\n"
            f"Write an engaging caption under 150 characters for this:\n"
            f"'{prompt_text.strip()}'\n"
            f"Add 3–5 trending hashtags."
        )
        with st.spinner("Generating caption…"):
            final_caption, t_cap = generate(caption_prompt, model_key)

        st.text_area("Generated Caption", final_caption, height=120)
        st.caption(f"LLM: {model_key} · {t_cap:.1f}s")

# ─── Mode B: Image → Caption + Hashtags ───
else:
    uploaded = st.file_uploader("Upload JPEG/PNG", type=["jpg", "jpeg", "png"])
    if st.button("Generate", key="btn_image") and uploaded:
        img_bytes = uploaded.read()
        st.image(img_bytes, caption="Uploaded image")

        with st.spinner("Extracting image description…"):
            raw_text, t_vision = raw_img_to_text(img_bytes)

        polish_prompt = (
            f"You are a social media strategist.\n"
            f"Polish this image description into an Instagram caption under 150 characters.\n"
            f"Add 3–5 trending hashtags.\n\nDescription:\n{raw_text}"
        )
        with st.spinner("Finalizing caption…"):
            final_caption, t_llm = generate(polish_prompt, model_key)

        st.text_area("Generated Caption", final_caption, height=120)
        st.caption(f"Vision {t_vision:.1f}s · LLM {model_key} · {t_llm:.1f}s")
    elif st.button("Generate", key="btn_image_warn"):
        st.warning("Please upload an image first.")
