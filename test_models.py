"""
test_models.py
Quick sanity‑check that all model end‑points in inference.py respond.
Run:  python test_models.py
Env:  HF_TOKEN, and (optionally) GOOGLE_APPLICATION_CREDENTIALS for Gemini
"""
from inference import available_models, generate
import textwrap, traceback

PROMPT = "Respond with: Hello, world."

def truncate(txt, n=1000):
    return textwrap.shorten(txt.replace("\n", " "), width=n, placeholder="…")

for key in available_models():
    print(f"\n=== {key.upper()} ===")
    try:
        out, latency = generate(PROMPT, key, max_tokens=16)
        print(f"✅  {truncate(out)}   ({latency:.1f}s)")
    except Exception as e:
        print(f"❌  ERROR — {e.__class__.__name__}: {e}")
        # optional: uncomment below to dump full stack trace
        # traceback.print_exc()
