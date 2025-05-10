import streamlit as st, time, hashlib, re
from utils.session import init_sidebar
from inference import generate

init_sidebar()
st.title("Social Post Generator")

platform = st.radio("Platform", ["Twitter / X","Threads"])
limit = 240 if platform.startswith("Twitter") else 500
tone  = st.text_input("Tone (optional)")
topic = st.text_area("Post topic")

def short_hash(t): return hashlib.sha256(t.encode()).hexdigest()[:8]

if st.button("Generate post") and topic.strip():
    prompt = (f"Craft a {platform} post under {limit} chars. "
              f"Tone: {tone or 'default'}. Topic: {topic}. "
              "Add ≤4 hashtags.")
    txt, lat = generate(prompt, st.session_state.model_key,160)
    txt = re.sub(r"\s+", " ", txt.strip())
    txt = txt[:limit-1] + "…" if len(txt) > limit else txt
    st.caption(f"{lat:.1f}s · {st.session_state.model_key}")
    st.text_area("Post", txt, height=120)
