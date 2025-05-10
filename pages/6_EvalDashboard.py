import streamlit as st, pandas as pd, altair as alt, pathlib
from utils.session import init_sidebar

init_sidebar()
st.title("Evaluation Dashboard")

root = pathlib.Path("runs")
runs = sorted([p.name for p in root.iterdir() if p.is_dir()])
run = st.selectbox("Run ID", runs)
metric = st.selectbox("Metric", ["relevance","correctness","faithfulness",
                                 "helpfulness","truthfulness","virality"])

sc = pd.read_csv(root/run/"scores.csv")
gen= pd.read_csv(root/run/"generations.csv")
sc = sc[sc.metric==metric]
sc["sys"]    = sc.model.apply(lambda m:"RAG" if m.endswith("_rag") else "BASE")
sc["family"] = sc.model.str.replace("_rag","")

agg = (sc.groupby(["family","sys"]).score.mean().reset_index())
families = ["deepseek","gemini","llama","mistral"]
agg["family"]=pd.Categorical(agg.family, categories=families, ordered=True)
chart = (alt.Chart(agg).mark_bar(width=25).encode(
            x=alt.X("family:N", axis=alt.Axis(labelAngle=0)),
            xOffset="sys:N",
            y=alt.Y("score:Q", title=f"mean {metric}"),
            color="sys:N").properties(width=500, height=350))
st.altair_chart(chart,use_container_width=True)

st.subheader("Latency")
st.altair_chart(
    alt.Chart(gen).mark_boxplot(size=10).encode(
        x=alt.X("model:N", axis=alt.Axis(labelAngle=30)),
        y="latency:Q"), use_container_width=True)
