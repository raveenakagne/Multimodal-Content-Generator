# eval/judge.py
import os, argparse, json
import pandas as pd
from inference import generate

# ---------------- Metric rubrics -----------------
METRICS = {
    "correctness": (
        "Correctness — Does the response convey the same factual meaning as the "
        "ideal reference answer? If no reference is available, rate factual coherence. "
        "1 = totally incorrect, 5 = fully correct."
    ),
    "relevance": (
        "Relevance — How well does the response address the user's prompt in an "
        "informative and concise manner? 1 = off‑topic, 5 = perfectly relevant."
    ),
    "faithfulness": (
        "Faithfulness — For RAG outputs, how strictly does the response adhere to the "
        "provided context without hallucinating extra facts? 1 = many hallucinations, "
        "5 = entirely grounded."
    ),
    "helpfulness": (
        "Helpfulness — Does the response satisfy the user's need and provide useful "
        "next steps or insights? 1 = useless, 5 = extremely helpful."
    ),
    "truthfulness": (
        "Truthfulness — Is the information factually accurate based on common "
        "knowledge? 1 = false or misleading, 5 = fully true."
    ),
    "virality": (
        "Virality potential — How likely is this content to attract engagement on its "
        "intended platform? Consider hook, emotion, clarity, shareability. "
        "1 = unlikely to spread, 5 = highly shareable."
    ),
}

JUDGE_MODEL = "gemini"     # fast, neutral

# -------------- prompt template ------------------
def build_prompt(metric_key, prompt, output, context=""):
    rubric = METRICS[metric_key]
    return (
        f"You are an impartial evaluator.\n{rubric}\n\n"
        "Respond ONLY with a single integer 1, 2, 3, 4, or 5.\n\n"
        f"USER PROMPT:\n{prompt}\n"
        + (f"\nRETRIEVED CONTEXT:\n{context}\n" if context else "") +
        f"\nMODEL OUTPUT:\n{output}\n\nScore:"
    )

# -------------- run judging ----------------------
def judge_row(row):
    results = []
    for m in METRICS:
        ctx = row.get("context", "") if "_rag" in row["model"] else ""
        j_prompt = build_prompt(m, row["prompt"], row["output"], ctx)
        score_txt, _ = generate(j_prompt, JUDGE_MODEL, max_tokens=5)
        score = int(score_txt.strip()[0])
        results.append(dict(id=row["id"], model=row["model"],
                            metric=m, score=score))
    return results

def main(run):
    path = f"runs/{run}/generations.csv"
    gen = pd.read_csv(path)
    all_scores = []
    for _, r in gen.iterrows():
        all_scores.extend(judge_row(r))
    out = f"runs/{run}/scores.csv"
    pd.DataFrame(all_scores).to_csv(out, index=False)
    print(f"✓ scores written to {out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("run")
    main(p.parse_args().run)
