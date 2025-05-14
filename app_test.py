import streamlit as st
from utils.session import init_sidebar

# --- Page config ---
st.set_page_config(page_title="Multimodal Suite", layout="centered")

# --- Custom CSS for futuristic, minimal, and aesthetic UI ---
st.markdown(
    """
    <style>
      html, body {
          background-color: #0e1117;
          color: #fafafa;
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Minimalist font */
      }
      footer, #MainMenu {visibility:hidden;}
      .block-container {
          padding-top: 1.5rem;
          max-width: 1000px; /* Limit content width for better readability */
      }
      a {color:#00aaff;} /* Futuristic blue accent */
      hr {border:1px solid #333;} /* Darker border */

      /* Typography adjustments */
      h1 {
          font-size: 2.5em;
          font-weight: 600;
          color: #00aaff; /* Accent color for main title */
      }
      h2 {
          font-size: 1.8em;
          font-weight: 500;
      }
      .stMarkdown strong {
          color: #00aaff; /* Accent color for bold text */
      }
      .stMarkdown p {
          font-size: 1.1em;
          line-height: 1.6;
      }

      /* Module card styling */
      .module-card {
          background-color: #1a1a1a; /* Slightly lighter dark background */
          padding: 20px;
          border-radius: 10px;
          margin-bottom: 20px;
          border: 1px solid #333; /* Subtle border */
          transition: transform 0.3s ease-in-out;
      }
      .module-card:hover {
          transform: translateY(-10px); /* Hover effect */
          border-color: #00aaff; /* Accent color on hover */
      }
      .module-card h3 {
          margin-top: 0;
          color: #fafafa;
          font-weight: 600;
      }
      .module-card .caption {
          color: #bbb; /* Lighter text for description */
          font-size: 0.9em;
      }
      .module-card .icon {
          font-size: 2em; /* Icon size */
          margin-right: 15px;
          color: #00aaff; /* Accent color for icons */
      }

      /* Font Awesome icons */
      @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css');

    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar (model selector) ---
init_sidebar()

# --- Hero section ---
st.title("Multimodal Content Suite")
st.markdown(
    """
    _One app to generate **blogs**, **tweets**, **Insta posts**, **podcasts**,_  \\
    _**short videos**, and **evaluate** them — all powered by AI._
    """
)

# --- Module cards with icons and refined layout ---
modules = [
    ("📝 Blog", "Long-form articles (~700 words)", "fas fa-blog", "Blog.py"),
    ("🐦 Twitter/X", "Catchy posts ≤240 chars", "fab fa-twitter", "Tweet.py"),
    ("📸 Instagram", "Generate image + caption", "fab fa-instagram", "Instagram.py"),
    ("🎙️ Podcast", "Script & MP3 via TTS", "fas fa-podcast", "Podcast.py"),
    ("🎬 Short Video", "15 s video from script", "fas fa-video", "ShortVideo.py"),
    ("📊 Evaluate", "Compare LLM outputs", "fas fa-chart-bar", "EvalDashboard.py"),
]

# Use a 2-column layout
cols = st.columns(2, gap="large")

# Distribute modules across the columns
for i, (title, desc, icon_class, page) in enumerate(modules):
    with cols[i % 2]: # Use modulo to alternate columns
        # Make the module card clickable and link to the corresponding page
        link = f"/{page.replace('.py', '')}"
        st.markdown(
            f"""
            <a href="{link}" target="_self" style="text-decoration: none;">
                <div class="module-card">
                    <h3><i class="{icon_class} icon"></i> {title}</h3>
                    <p class="caption">{desc}</p>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )

# --- Divider + call-to-action ---
st.markdown("<hr>", unsafe_allow_html=True)
st.info("Select any tool from the sidebar to get started!")