# eval/judge.py
import os, argparse, json
import pandas as pd
from inference import generate

# ---------------- Metric rubrics -----------------
METRICS = {
    "correctness": (
        "Correctness — To what extent does the output convey the *exact* same factual and logical content as the expected answer or source truth? "
        "Do not reward partial correctness or close approximations. Penalize any factual drift or logical inconsistency. "
        "1 = entirely wrong or contradictory, 5 = perfectly aligned and factually consistent."
    ),
    "relevance": (
        "Relevance — How *precisely and fully* does the response address the user's prompt? "
        "Vague, generic, or verbose answers must be penalized even if somewhat related. "
        "1 = completely unrelated, 5 = directly on-topic with sharp focus and no padding."
    ),
    "faithfulness": (
        "Faithfulness — How *strictly and accurately* does the response adhere to the provided information or context, if any? "
        "Do not tolerate hallucinated details, speculative claims, or shifts from source logic. "
        "1 = major deviations or fabrications, 5 = entirely consistent with the provided or implied source."
    ),
    "helpfulness": (
        "Helpfulness — How effectively does the response *resolve* the user's need with actionable, insightful, or complete guidance? "
        "If the user would still need to search or guess, score lower. "
        "1 = offers no usable value, 5 = fully satisfies the user’s goal with clarity."
    ),
    "truthfulness": (
        "Truthfulness — How *objectively accurate* is the response based on established, verifiable facts? "
        "Do not excuse outdated, misleading, or unverifiable claims. "
        "1 = contains falsehoods or misinformation, 5 = entirely accurate and aligned with consensus knowledge."
    ),
    "virality": (
        "Virality potential — How likely is the response to generate *widespread attention* in its intended channel (e.g., social media, public forums)? "
        "Consider originality, clarity, punch, and emotional or intellectual impact. "
        "1 = dull or forgettable, 5 = highly engaging and likely to be shared."
    ),
}

JUDGE_MODEL = "gemini"     # fast, neutral

# -------------- prompt template ------------------
def build_prompt(metric_key, prompt, output, context=""):
    rubric = METRICS[metric_key]
    return (
        f"You are an impartial evaluator.\n"
        f"{rubric}\n\n"
        "Score with extreme precision. Do not be lenient. Avoid middle scores unless clearly warranted.\n"
        "Respond ONLY with a single integer: 1, 2, 3, 4, or 5 — nothing else.\n\n"
        f"USER PROMPT:\n{prompt.strip()}\n"
        + (f"\nREFERENCE CONTEXT OR EXPECTED SOURCE:\n{context.strip()}\n" if context else "") +
        f"\nMODEL OUTPUT:\n{output.strip()}\n\n"
        "Score:"
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
