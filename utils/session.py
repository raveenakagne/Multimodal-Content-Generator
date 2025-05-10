import streamlit as st
from inference import available_models

def init_sidebar():
    if "model_key" not in st.session_state:
        st.session_state.model_key = available_models()[0]

    with st.sidebar:
        st.title("🛠 Multimodal Suite")
        st.selectbox("LLM back‑end",
                     available_models(),
                     key="model_key")
        st.markdown("---")
        st.caption("Dark theme enforced")
