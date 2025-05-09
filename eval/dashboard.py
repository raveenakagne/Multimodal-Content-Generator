# eval/dashboard.py   run:  streamlit run eval/dashboard.py
import streamlit as st, pandas as pd, os
run = st.selectbox("Choose run", sorted(os.listdir("runs")))
agg = pd.read_csv(f"runs/{run}/agg.csv")
metric = st.selectbox("Metric", agg["metric"].unique())
df = agg[agg.metric==metric].sort_values("mean", ascending=False)

st.header(f"{metric.title()} — base vs RAG")
st.bar_chart(df.set_index("model")["mean"])

st.header("Latency distribution")
gen = pd.read_csv(f"runs/{run}/generations.csv")
st.bar_chart(gen.groupby("model")["latency"].mean())
