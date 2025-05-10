# short_video/tiktok.py  (debug‑friendly)
import sys, pathlib, hashlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import sys, pathlib, tempfile          # ← add tempfile here
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
from short_video import builder as sv        # access internal helpers
from inference import available_models

st.set_page_config(page_title="Short‑Form Video Generator", layout="centered")
st.title("Short‑Form Video Generator  🎬  (debug view)")

topic = st.text_input("Video topic", placeholder="e.g. Hidden benefits of green tea")

col1, col2 = st.columns(2)
model_key = col1.selectbox("LLM model", available_models())
use_voice = col2.checkbox("Add voice‑over (Deepgram)", value=True)

def _hash(b: bytes): return hashlib.md5(b).hexdigest()[:6]

if st.button("Generate") and topic.strip():
    with st.spinner("Step 1/4: LLM script …"):
        sents = sv.script_from_llm(topic.strip(), model_key)
    st.subheader("▪︎ Script sentences")
    for i, s in enumerate(sents, 1):
        st.write(f"{i}. {s}")

    with st.spinner("Step 2/4: Deriving image prompts …"):
        prompts = sv.image_prompts_from_script(sents, model_key)
    st.subheader("▪︎ Image prompts")
    for i, p in enumerate(prompts, 1):
        st.write(f"{i}. {p}")

    with st.spinner("Step 3/4: Generating images (SD‑3) …"):
        img_paths = sv.images_from_prompts(prompts)

    st.subheader("▪︎ Generated images")
    dup_flag = False
    hashes = []
    cols = st.columns(5)
    for i, (p, c) in enumerate(zip(img_paths, cols)):
        data = p.read_bytes()
        digest = _hash(data)
        hashes.append(digest)
        c.image(data, caption=f"#{i+1}  {digest}")
    if len(set(hashes)) < 5:
        dup_flag = True
        st.error("⚠ Detected duplicate images — SD‑3 may be repeating.")

    audio_path = None
    if use_voice:
        with st.spinner("Deepgram voice‑over …"):
            audio_path, _ = sv.voice_over(sents)

    with st.spinner("Step 4/4: Building video …"):
        mp4 = pathlib.Path(tempfile.mkdtemp()) / "short.mp4"
        sv.build_video(img_paths, audio_path, mp4)

    st.success("Video ready ↓")
    st.video(str(mp4))
    st.download_button("Download MP4", open(mp4, "rb").read(),
                       file_name="short_video.mp4")

    # cleanup temp images
    for p in img_paths:
        p.unlink(missing_ok=True)

elif st.button("Generate"):
    st.warning("Please enter a topic first.")
