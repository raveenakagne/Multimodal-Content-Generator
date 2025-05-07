# insta.py  –  Instagram generator using Vertex AI Imagen 3 + BLIP‑2
# ------------------------------------------------------------------
# Modes
#   • Prompt  →  Imagen‑3 PNG + IG caption + hashtags
#   • Image   →  caption + hashtags
#
# Requirements
#   pip install streamlit pillow google-cloud-aiplatform vertexai requests
#   export GCP_PROJECT=your‑project   GCP_REGION=us-central1
#   export GOOGLE_APPLICATION_CREDENTIALS=/path/key.json
#   export HF_TOKEN=hf_...
# ------------------------------------------------------------------

import os, time, base64, hashlib, requests
from functools import lru_cache
from typing import Tuple
import creds
import streamlit as st
from PIL import Image

from inference import generate, available_models    # text LLMs for captions

# ─── Vertex AI initialisation ─────────────────────────────────────
import vertexai
from vertexai.preview import generative_models as gen
from vertexai.preview.generative_models import GenerationConfig

vertexai.init(
    project=os.getenv("GCP_PROJECT"),
    location=os.getenv("GCP_REGION", "us-central1")
)

IMG_MODEL = gen.GenerativeModel("publishers/google/models/imagen-3.0-generate-002")



# ─── HF token for BLIP‑2 image captioning ────────────────────────
# current line (remove it)
# HF_TOKEN = os.getenv("HF_TOKEN")

# replace with:
HF_TOKEN = (getattr(creds, "hf_token", None) or os.getenv("HF_TOKEN"))

if not HF_TOKEN:
    st.error("HF_TOKEN env var required for BLIP‑2 vision captioning.")
    st.stop()

# ─── Helper functions ────────────────────────────────────────────
def sha10(txt: str) -> str:
    return hashlib.sha256(txt.encode()).hexdigest()[:10]
@lru_cache(maxsize=64)
def prompt_to_image(prompt: str) -> Tuple[bytes, float]:
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    headers = {
        "Authorization": f"Bearer {os.getenv('STABILITY_KEY')}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": prompt,
        "output_format": "png",
        "mode": "text-to-image",
    }

    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    b64 = r.json()["image"]  # Base64 PNG
    return base64.b64decode(b64), time.time() - t0



@lru_cache(maxsize=128)
def img_to_caption(img_bytes: bytes) -> Tuple[str, float]:
    url = "https://api-inference.huggingface.co/models/Salesforce/blip2-flan-t5-xl"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": base64.b64encode(img_bytes).decode(),
               "parameters": {"max_new_tokens": 30}}
    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    return r.json()[0]["generated_text"], time.time() - t0

# ─── Streamlit UI ────────────────────────────────────────────────
st.set_page_config(page_title="Instagram Generator", layout="centered")
st.title("Instagram Content Generator · Vertex Imagen 3")

mode = st.radio(
    "Select mode",
    ["Prompt → Image + Caption", "Image → Caption + Hashtags"],
    horizontal=True
)

with st.sidebar:
    st.subheader("Caption LLM")
    model_key = st.selectbox("Choose text model", available_models())

# ── Mode A : prompt → image + caption ────────────────────────────
if mode.startswith("Prompt"):
    prompt_text = st.text_area("Describe the image concept")

    if st.button("Generate") and prompt_text.strip():
        with st.spinner("Generating image…"):
            img_bytes, t_img = prompt_to_image(prompt_text.strip())

        st.image(img_bytes, caption=f"Imagen 3 • {t_img:.1f}s")

        cap_prompt = (
            f"Write an engaging Instagram caption (<150 chars) for an image about: "
            f"\"{prompt_text.strip()}\". Include 3‑5 trending hashtags."
        )
        with st.spinner("Generating caption…"):
            caption, t_cap = generate(cap_prompt, model_key)

        st.text_area("Caption", caption, height=120)
        st.caption(f"LLM {model_key} • {t_cap:.1f}s")

# ── Mode B : image → caption + hashtags ──────────────────────────
else:
    up = st.file_uploader("Upload JPEG or PNG", type=["jpg", "jpeg", "png"])

    if st.button("Generate") and up:
        img_bytes = up.read()
        st.image(img_bytes, caption="Uploaded")

        with st.spinner("Describing image…"):
            raw_caption, t_raw = img_to_caption(img_bytes)

        polish_prompt = (
            f"Rewrite the caption below (<150 chars) for Instagram and add 3‑5 hashtags.\n"
            f"Caption: {raw_caption}"
        )
        with st.spinner("Polishing caption…"):
            caption, t_pol = generate(polish_prompt, model_key)

        st.text_area("Caption", caption, height=120)
        st.caption(f"Vision {t_raw:.1f}s • LLM {model_key} {t_pol:.1f}s")
    elif st.button("Generate"):
        st.warning("Please upload an image first.")
