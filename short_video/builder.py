# short_video/builder.py
# ------------------------------------------------------------------
# Build a vertical MP4 short (Python 3.13, no moviepy, FFmpeg only).
# Steps:  topic → 5‑sentence script → 5 distinct image prompts →
#         SD‑3 images (parallel) → optional Deepgram VO → FFmpeg concat
# ------------------------------------------------------------------
import sys, pathlib, re, tempfile, threading, subprocess, textwrap
from pathlib import Path
from typing import List, Tuple
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from tqdm import tqdm
from inference import generate
from insta import prompt_to_image
from short_video.voice_utils import clean_script, tts_deepgram

# ─── helpers ──────────────────────────────────────────────────
def _denumber(line: str) -> str:
    return re.sub(r"^\s*(?:[\u2022\-•]|\d+[\.\)]?)\s*", "", line).strip()

def _split_sentences(txt: str) -> List[str]:
    return [_denumber(s) for s in re.split(r'(?<=[\.!?])\s+', txt) if s.strip()]

# ─── 1. script (5 sentences) ─────────────────────────────────
def script_from_llm(topic: str, model="mistral") -> List[str]:
    prm = (
        "Write exactly FIVE separate sentences (≤24 words each) "
        f"for a vertical short‑form video on: {topic} .\n"
        "Do NOT number or bullet. Put each sentence on its own line."
    )
    raw, _ = generate(prm, model, 300)
    sents = [_denumber(s) for s in raw.splitlines() if s.strip()]
    if len(sents) < 5:
        sents = _split_sentences(raw)
    while len(sents) < 5:
        sents.append(sents[-1])
    return sents[:5]

# ─── 2. derive 5 distinct image prompts ─────────────────────
def image_prompts_from_script(sentences: List[str], model="mistral") -> List[str]:
    joined = "\n".join(sentences)
    prm = textwrap.dedent(f"""
        Transform each of the following lines into a vivid, photorealistic
        VERTICAL image prompt (no camera words). Keep order, one per line.

        {joined}
    """)
    raw, _ = generate(prm, model, 200)
    prompts = [p.strip() for p in raw.splitlines() if p.strip()]
    if len(prompts) < 5:
        prompts += prompts * 5
    return prompts[:5]

# ─── 3. download SD‑3 images in parallel ─────────────────────
def images_from_prompts(prompts: List[str]) -> List[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="frames_"))
    paths = [tmp_dir / f"frame_{i}.png" for i in range(len(prompts))]

    def worker(i, prm):
        img_bytes, _ = prompt_to_image(f"vertical 1080x1920 photoreal, {prm}")
        paths[i].write_bytes(img_bytes)

    threads = [threading.Thread(target=worker, args=(i, p))
               for i, p in enumerate(prompts)]
    for t in threads: t.start()
    for t in threads: t.join()
    return paths

# ─── 4. Deepgram VO (optional) ───────────────────────────────
def voice_over(sentences: List[str]) -> Tuple[Path, float]:
    return tts_deepgram(clean_script("  …  ".join(sentences)))

# ─── 5. FFmpeg concat with Ken‑Burns zoom ────────────────────
# ─── simple 3‑second‑per‑image stitcher (no zoom) ─────────────
# ─── motion Ken‑Burns (two‑pass) ──────────────────────────────
def build_video(imgs: List[Path], audio: Path | None,
                out_mp4: Path, fps: int = 30):
    """
    1. For each image -> clip_i.mp4 (3 s, slow zoom‑in).
    2. Concatenate the 5 clips with the concat demuxer.
    3. Overlay optional audio, re‑encode libx264.
    """

    tmp_dir = Path(tempfile.mkdtemp(prefix="clips_"))
    clip_paths = []

    # --- pass 1  :  one clip per image --------------------------------
    for idx, p in enumerate(imgs):
        clip = tmp_dir / f"clip_{idx}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-loop", "1", "-i", str(p),
            "-vf",
            f"scale=1620:2880,"
            f"zoompan=z='zoom+0.0006':d={3*fps}:s=1080x1920,setsar=1",
            "-t", "3",
            "-r", str(fps),
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264",
            "-preset", "veryfast",
            str(clip)
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        clip_paths.append(clip)

    # write concat list
    list_file = tmp_dir / "list.txt"
    with list_file.open("w") as f:
        for c in clip_paths:
            f.write(f"file '{c.as_posix()}'\n")

    # --- pass 2  :  concat + audio ------------------------------------
    cmd = ["ffmpeg", "-y",
           "-f", "concat", "-safe", "0", "-i", str(list_file)]

    if audio:
        cmd += ["-i", str(audio)]

    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", str(fps),
        "-preset", "veryfast"
    ]
    if audio:
        cmd += ["-shortest"]          # cut video if audio shorter

    cmd += [str(out_mp4)]
    subprocess.run(cmd, check=True)

    # cleanup temp clips
    for c in clip_paths:
        c.unlink(missing_ok=True)
    list_file.unlink(missing_ok=True)
    tmp_dir.rmdir()

# ─── 6. high‑level generator ─────────────────────────────────
def generate_short(topic: str, model_key: str, voice=True) -> Path:
    sents   = script_from_llm(topic, model_key)
    prompts = image_prompts_from_script(sents, model_key)

    print("✔ Script:", " | ".join(sents))
    print("✔ Image prompts:", prompts)

    img_paths = images_from_prompts(prompts)
    print("✔ Images ready.")

    audio_path = None
    if voice:
        audio_path, _ = voice_over(sents)
        print("✔ Voice‑over ready.")

    out = Path(tempfile.mktemp(suffix=".mp4"))
    build_video(img_paths, audio_path, out)
    print("✔ Video built →", out)

    for p in img_paths: p.unlink(missing_ok=True)
    return out
