import streamlit as st, pathlib, tempfile
from utils.session import init_sidebar
from short_video.builder import script_from_llm, image_prompts_from_script,\
                                images_from_prompts, voice_over, build_video

init_sidebar()
st.title("Short‑Form Video")

topic = st.text_input("Video topic")
voice = st.checkbox("Voice‑over", value=True)

if st.button("Generate") and topic.strip():
    sents = script_from_llm(topic, st.session_state.model_key)
    st.write("**Script sentences**")
    for i,s in enumerate(sents,1): st.write(f"{i}. {s}")

    prompts = image_prompts_from_script(sents, st.session_state.model_key)
    imgs = images_from_prompts(prompts)
    st.write("**Images**"); cols = st.columns(5)
    for p,c in zip(imgs,cols): c.image(p.read_bytes(), width=110)

    audio = None
    if voice:
        st.info("Deepgram voice‑over…")
        audio,_ = voice_over(sents)

    mp4 = pathlib.Path(tempfile.mkdtemp())/"video.mp4"
    build_video(imgs, audio, mp4)
    st.video(str(mp4))
