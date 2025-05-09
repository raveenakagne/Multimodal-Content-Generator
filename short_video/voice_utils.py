# short_video/voice_utils.py
import os, time, tempfile, requests, re
from pathlib import Path
from typing import Tuple

try:
    import creds
except ImportError:
    creds = None

DG_KEY = getattr(creds, "DEEPGRAM_API_KEY", None) or os.getenv("DEEPGRAM_API_KEY")

def clean_script(raw: str) -> str:
    raw = re.sub(r"(?i)^here[’']?s a script[^\n]*\n?", "", raw).strip()
    raw = re.sub(r"\*\*", "", raw)
    raw = re.sub(r"^\[.*?\]$", "", raw, flags=re.MULTILINE).strip()
    return re.sub(r"[ ]{2,}", " ", raw)

def tts_deepgram(text: str) -> Tuple[Path, float]:
    if not DG_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY missing.")
    url = "https://api.deepgram.com/v1/speak"
    headers = {"Authorization": f"Token {DG_KEY}", "Content-Type": "application/json"}
    payload = {"text": text}
    t0 = time.time()
    r = requests.post(url, headers=headers, json=payload, timeout=90)
    r.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp.write(r.content)
    tmp.close()
    return Path(tmp.name), time.time() - t0
