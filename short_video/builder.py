# short_video/builder.py
# -------------------------------------------------------------
# Vertical short‑video builder (Python 3.13 friendly – no moviepy)
# -------------------------------------------------------------
import sys, pathlib, re, tempfile, threading, subprocess
from pathlib import Path
from typing import List, Tuple

# make project root importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from inference import generate
from insta import prompt_to_image            # SD‑3 helper
from short_video.voice_utils import clean_script, tts_deepgram

# ──────────────────────────────────────────────────────────────
# 1.  Script from LLM  ➜ 5 cleaned sentences
# ──────────────────────────────────────────────────────────────
def _denumber(line: str) -> str:
    """Strip '1.', '1)', '•', etc. from start of a line."""
    return re.sub(r"^\s*(?:[\u2022\-•]|\d+[\.\)]?)\s*", "", line).strip()

def _split_into_sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    return [_denumber(p) for p in parts if p.strip()]

def script_from_llm(topic: str, model_key="mistral") -> List[str]:
    prompt = (
        "Write exactly FIVE separate sentences for a vertical short‑form video.\n"
        "• Do NOT number or bullet the sentences.\n"
        "• Each sentence ≤ 24 words, descriptive and visual.\n"
        f"Topic: {topic}\n"
        "Output each sentence on its own new line."
    )
    raw, _ = generate(prompt, model_key, 300)
    sents = [_denumber(s) for s in raw.splitlines() if s.strip()]

    if len(sents) < 5:              # fallback if LLM ignored newlines
        sents = _split_into_sentences(raw)

    while len(sents) < 5:           # pad to exactly 5
        sents.append(sents[-1])

    return sents[:5]

# ──────────────────────────────────────────────────────────────
# 2.  Parallel SD‑3 image generation
# ──────────────────────────────────────────────────────────────
def images_from_sentences(sentences: List[str]) -> List[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="frames_"))
    paths = [tmp_dir / f"frame_{i}.png" for i in range(len(sentences))]

    def worker(idx, sent):
        img_bytes, _ = prompt_to_image(f"vertical 1080x1920 photoreal: {sent}")
        paths[idx].write_bytes(img_bytes)

    threads = [threading.Thread(target=worker, args=(i, s))
               for i, s in enumerate(sentences)]
    for t in threads: t.start()
    for t in threads: t.join()
    return paths

# ──────────────────────────────────────────────────────────────
# 3.  Optional Deepgram TTS (joined with soft pauses)
# ──────────────────────────────────────────────────────────────
def voice_over(sentences: List[str]) -> Tuple[Path, float]:
    joined = "  …  ".join(sentences)                 # ellipsis pause
    return tts_deepgram(clean_script(joined))

# ──────────────────────────────────────────────────────────────
# 4.  Assemble video with FFmpeg  (Ken‑Burns zoom)
# ──────────────────────────────────────────────────────────────
def build_video(img_paths: List[Path], audio_path: Path | None,
                out_mp4: Path, fps: int = 30):
    cmd = ["ffmpeg", "-y"]
    # inputs: each image looped for 3 s
    for p in img_paths:
        cmd += ["-loop", "1", "-t", "3", "-i", str(p)]
    if audio_path:
        cmd += ["-i", str(audio_path)]

    # zoom‑pan filters
    zoom_filters = [
        (f"[{i}:v]scale=1620:2880,"
         f"zoompan=z='zoom+0.0005':d={3*fps}:s=1080x1920,setsar=1[v{i}]")
        for i in range(len(img_paths))
    ]
    concat_inputs = "".join(f"[v{i}]" for i in range(len(img_paths)))
    filter_complex = ";".join(zoom_filters) + ";" + \
        f"{concat_inputs}concat=n={len(img_paths)}:v=1:a=0,format=yuv420p[v]"  # ← no trailing ';'

    cmd += ["-filter_complex", filter_complex, "-map", "[v]"]
    if audio_path:
        audio_idx = len(img_paths)         # last input is audio
        cmd += ["-map", f"{audio_idx}:a", "-shortest"]

    cmd += ["-r", str(fps), "-pix_fmt", "yuv420p", str(out_mp4)]
    subprocess.run(cmd, check=True)

# ──────────────────────────────────────────────────────────────
# 5.  High‑level helper exposed to Streamlit
# ──────────────────────────────────────────────────────────────
def generate_short(topic: str, model_key: str, use_voice=True) -> Path:
    sents = script_from_llm(topic, model_key)
    print("✔ Script:", " | ".join(sents))

    img_paths = images_from_sentences(sents)
    print("✔ Images generated")

    audio_path = None
    if use_voice:
        audio_path, _ = voice_over(sents)
        print("✔ Voice‑over ready")

    out = Path(tempfile.mktemp(suffix=".mp4"))
    build_video(img_paths, audio_path, out)
    print("✔ Video built →", out)

    for p in img_paths: p.unlink(missing_ok=True)
    return out
