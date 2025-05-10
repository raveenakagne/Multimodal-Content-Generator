import streamlit as st, time
from inference import generate
from utils.session import init_sidebar
from insta import prompt_to_image   # reuse generator only

init_sidebar()
st.title("Instagram Image + Caption")

prompt = st.text_area("Describe image concept")

if st.button("Generate"):
    if not prompt.strip(): st.warning("Enter a prompt."); st.stop()
    with st.spinner("SD‑3 generating…"):
        img, t_img = prompt_to_image(prompt.strip())
    st.image(img, caption=f"SD3 · {t_img:.1f}s")

    cap_prompt = (f"Write an IG caption (<150 chars) for: {prompt}. "
                  "Add 3‑5 trending hashtags.")
    with st.spinner("Caption…"):
        cap, t_llm = generate(cap_prompt, st.session_state.model_key,120)
    st.text_area("Caption", cap, height=120)
    st.caption(f"{t_llm:.1f}s · {st.session_state.model_key}")
