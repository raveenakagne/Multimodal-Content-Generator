# eval/driver.py
# -------------------------------------------------------------
# Generate model outputs for evaluation.
# Usage:
#   python -m eval.driver --raw data/input_prompts_only.csv \
#                         --rag data/input_with_rag_contexts.csv \
#                         --limit 10          # optional
# -------------------------------------------------------------

import sys, pathlib, time, argparse
import pandas as pd
from inference import generate

# make project root importable
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

BASE_MODELS = ["llama", "mistral", "gemini", "deepseek"]
RAG_MODELS  = [m + "_rag" for m in BASE_MODELS]

def main(raw_csv, rag_csv, limit):
    run_dir = pathlib.Path("runs") / time.strftime("%y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------- RAW prompts ----------
    df_raw = pd.read_csv(raw_csv)
    if limit:
        df_raw = df_raw.head(limit)
    total_raw = len(df_raw) * len(BASE_MODELS)
    step = 0
    print(f"Generating {total_raw} raw combinations…")

    for idx, row in df_raw.iterrows():
        for key in BASE_MODELS:
            step += 1
            print(f"[RAW {step}/{total_raw}] {key}", end="\r")
            text, lat = generate(row["prompt"], key, 700)
            rows.append(dict(
                id       = idx,
                model    = key,
                prompt   = row["prompt"],
                context  = "",
                latency  = lat,
                output   = text.replace("\n", " ")
            ))
    print()  # newline after progress

    # ---------- RAG prompts ----------
    df_rag = pd.read_csv(rag_csv)
    if limit:
        df_rag = df_rag.head(limit)
    context_cols = [c for c in df_rag.columns if c.lower().startswith("context")]
    total_rag = len(df_rag) * len(RAG_MODELS)
    step = 0
    print(f"Generating {total_rag} RAG combinations…")

    for idx, row in df_rag.iterrows():
        joined_ctx = "\n\n".join(str(row[c]) for c in context_cols)
        prompt_plus_ctx = f"{row['prompt']}\n\n{joined_ctx}"
        for key in RAG_MODELS:
            step += 1
            print(f"[RAG {step}/{total_rag}] {key}", end="\r")
            base = key.replace("_rag", "")
            text, lat = generate(prompt_plus_ctx, base, 700)
            rows.append(dict(
                id       = idx,
                model    = key,
                prompt   = row["prompt"],
                context  = joined_ctx,
                latency  = lat,
                output   = text.replace("\n", " ")
            ))
    print()

    out_path = run_dir / "generations.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n✓ generations saved to {out_path}")
    print(f"Run folder: {run_dir.name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw",  required=True, help="CSV with prompt only")
    ap.add_argument("--rag",  required=True, help="CSV with prompt+context cols")
    ap.add_argument("--limit", type=int, default=None, help="Process first N prompts")
    args = ap.parse_args()
    main(args.raw, args.rag, args.limit)
