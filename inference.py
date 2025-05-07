"""
inference.py   ·   provider‑agnostic wrapper
-------------------------------------------------------------
generate(prompt, model_key, max_tokens=700)  -> (reply, latency_s)
available_models()  -> list[str]
-------------------------------------------------------------
Dependencies:
  pip install requests vertexai together
  # ( vertexai comes from `google-cloud-aiplatform` ≥ 1.50 )
"""

import os, time, requests, importlib
from typing import Tuple

# ─────────────────────────────────────────────────────────────
# 0.  secrets   (creds.py is .gitignored)
# ─────────────────────────────────────────────────────────────
try:
    import creds                # creds.hf_token / together_api_key / gcp_project
except ImportError:
    creds = None

# ─────────────────────────────────────────────────────────────
# 1.  Registry  – one line per model
# ─────────────────────────────────────────────────────────────
_MODELS = {
    # Vertex AI managed models
    "llama":  {"type": "vertex",   "id": "publishers/meta/models/llama-4-maverick-17b-128e-instruct-maas"},   # example
    "gemini": {"type": "vertex",   "id": "gemini-2.0-flash-lite-001"},
    # Together AI serverless chat models
    "mistral": {"type": "together", "id": "mistralai/Mistral-7B-Instruct-v0.2"},
    "deepseek":{"type": "together", "id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"},
}

# ─────────────────────────────────────────────────────────────
# 2.  Together AI token
# ─────────────────────────────────────────────────────────────
def _resolve_together_token() -> str:
    tok = (getattr(creds, "together_api_key", None)
           or os.getenv("TOGETHER_API_KEY"))
    if not tok:
        raise RuntimeError("Together AI key not found "
                           "(creds.together_api_key or TOGETHER_API_KEY env).")
    return tok

_TOGETHER_HEADERS = {
    "Authorization": f"Bearer {_resolve_together_token()}",
    "Content-Type": "application/json"
}

# ─────────────────────────────────────────────────────────────
# 3.  Together chat completions   :contentReference[oaicite:0]{index=0}
# ─────────────────────────────────────────────────────────────
def _together_call(model_id: str, prompt: str, max_tokens: int) -> Tuple[str, float]:
    url = "https://api.together.xyz/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    rsp = requests.post(url, headers=_TOGETHER_HEADERS, json=payload, timeout=120)
    rsp.raise_for_status()
    text = rsp.json()["choices"][0]["message"]["content"]
    return text, time.time() - t0

# ─────────────────────────────────────────────────────────────
# 4.  Vertex AI call (works for Llama and Gemini)  :contentReference[oaicite:1]{index=1}
# ─────────────────────────────────────────────────────────────
def _vertex_call(model_id: str, prompt: str, max_tokens: int) -> Tuple[str, float]:
    import vertexai
    from vertexai.preview.generative_models import GenerativeModel

    vertexai.init(
        project=getattr(creds, "gcp_project", None) or os.getenv("GCP_PROJECT"),
        location=getattr(creds, "gcp_region", None)  or os.getenv("GCP_REGION", "us-central1")
    )
    model = GenerativeModel(model_id)

    t0 = time.time()
    rsp = model.generate_content(prompt,
                                 generation_config={"max_output_tokens": max_tokens})
    return rsp.text, time.time() - t0

# ─────────────────────────────────────────────────────────────
# 5.  Public API
# ─────────────────────────────────────────────────────────────
def generate(prompt: str, model_key: str, max_tokens: int = 700):
    entry = _MODELS[model_key]
    if entry["type"] == "vertex":
        return _vertex_call(entry["id"], prompt, max_tokens)
    if entry["type"] == "together":
        return _together_call(entry["id"], prompt, max_tokens)
    raise ValueError(f"Unknown provider in registry for {model_key}")

def available_models():
    return list(_MODELS.keys())
