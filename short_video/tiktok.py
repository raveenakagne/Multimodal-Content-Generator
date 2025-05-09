# short_video/tiktok.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import streamlit as st
from short_video.builder import generate_short
from inference import available_models

st.set_page_config(page_title="Short‑Form Video Generator", layout="centered")
st.title("Short‑Form Video Generator  🎬")

topic = st.text_input("Video topic", placeholder="e.g. Hidden benefits of green tea")

col1, col2 = st.columns(2)
with col1:
    model_key = st.selectbox("LLM for script", available_models())
with col2:
    voice = st.checkbox("Add voice‑over (Deepgram)", value=True)

if st.button("Generate Video") and topic.strip():
    with st.spinner("Rendering… this can take up to a minute"):
        mp4_path = generate_short(topic.strip(), model_key, voice)
    st.video(str(mp4_path))
    st.download_button("Download MP4",
                       data=open(mp4_path, "rb").read(),
                       file_name="short_video.mp4")
elif st.button("Generate Video") and not topic.strip():
    st.warning("Please enter a topic first.")
