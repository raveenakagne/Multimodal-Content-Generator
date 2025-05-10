import streamlit as st
from utils.session import init_sidebar

# ─── Page config ────────────────────────────────────────────────
st.set_page_config(page_title="Multimodal Suite", layout="centered")

# ─── Dark-mode CSS injection ────────────────────────────────────
st.markdown(
    """
    <style>
      html, body {background-color:#0e1117; color:#fafafa;}
      footer, #MainMenu {visibility:hidden;}
      .block-container {padding-top:1.5rem;}
      a {color:#ff7f0e;}
      hr {border:1px solid #444;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar (model selector) ───────────────────────────────────
init_sidebar()

# ─── Hero section ───────────────────────────────────────────────
st.title("🔧 Multimodal Content Suite")
st.markdown(
    "_One app to generate **blogs**, **tweets**, **Insta posts**, **podcasts**,_  \n"
    "_**short videos**, and **evaluate** them — all powered by AI._"
)

# ─── Module cards ───────────────────────────────────────────────
modules = [
    ("📝 Blog",         "Long-form articles (~700 words)"),
    ("🐦 Twitter/X",    "Catchy posts ≤240 chars"),
    ("📸 Instagram",    "Generate image + caption"),
    ("🎙️ Podcast",      "Script & MP3 via TTS"),
    ("🎬 Short Video",  "15 s video from script"),
    ("📊 Evaluate",     "Compare LLM outputs"),
]

cols = st.columns(3, gap="large")
for (title, desc), col in zip(modules, cols * 2):
    col.markdown(f"**{title}**")
    col.caption(desc)

# ─── Divider + call-to-action ───────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.info("Select any tool from the sidebar to get started!")
