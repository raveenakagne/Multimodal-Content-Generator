import streamlit as st
from utils.session import init_sidebar
from podcast import clean_script, tts_deepgram
from inference import generate

init_sidebar()
st.title("Podcast Snippet")

topic = st.text_area("Topic")
if st.button("Generate Script") and topic.strip():
    p = ("Podcast script (~240 words) friendly tone, no stage directions.\n"
         f"Topic: {topic}")
    script, lat = generate(p, st.session_state.model_key,340)
    st.session_state.script = script
    st.text_area("Script", script, height=200)
    st.caption(f"{lat:.1f}s · {st.session_state.model_key}")

if st.button("Generate Audio") and st.session_state.get("script"):
    mp3, lat = tts_deepgram(clean_script(st.session_state.script))
    st.audio(open(mp3,"rb").read(), format="audio/mp3")
    st.caption(f"TTS {lat:.1f}s")
