# eval/dashboard.py  ·  grouped thin bars
import streamlit as st, pandas as pd, altair as alt, pathlib

st.set_page_config(page_title="Eval dashboard", layout="wide")

root = pathlib.Path("runs")
runs = sorted([p.name for p in root.iterdir() if p.is_dir()])
run  = st.selectbox("Run", runs)

metric = st.selectbox("Metric", ["relevance","correctness","faithfulness",
                                 "helpfulness","truthfulness","virality"])

# ─── load and prep ────────────────────────────────────────────
scores = pd.read_csv(root/run/"scores.csv")
gen    = pd.read_csv(root/run/"generations.csv")
scores = scores[scores.metric == metric]

scores["system"] = scores.model.apply(lambda m: "RAG" if m.endswith("_rag") else "BASE")
scores["family"] = scores.model.str.replace("_rag","")

# aggregate
agg = (scores.groupby(["family","system"])
              .score.mean()
              .reset_index())

# ensure consistent order
families = ["deepseek","gemini","llama","mistral"]
agg["family"] = pd.Categorical(agg.family, categories=families, ordered=True)
agg = agg.sort_values(["family","system"])

# ─── thin grouped bar (Altair 5) ─────────────────────────────
st.header(f"{metric.title()} — base vs rag ")

bar = (
    alt.Chart(agg)
       .mark_bar(width=25)
       .encode(
           x=alt.X("family:N", title=None, axis=alt.Axis(labelAngle=0)),
           xOffset="system:N",                # groups inside family
           y=alt.Y("score:Q", title=f"mean {metric}"),
           color=alt.Color("system:N", title=None,
                           scale=alt.Scale(domain=["BASE","RAG"],
                                            range=["#1f77b4","#ff7f0e"]))
       )
       .properties(width=400, height=350)
)

st.altair_chart(bar, use_container_width=True)

# ─── latency boxplot (unchanged) ──────────────────────────────
st.subheader("Latency distribution")
box = (
    alt.Chart(gen).mark_boxplot(size=10).encode(
        x=alt.X("model:N", title=None, axis=alt.Axis(labelAngle=30)),
        y="latency:Q")
    .properties(width=800, height=300))
st.altair_chart(box, use_container_width=True)
