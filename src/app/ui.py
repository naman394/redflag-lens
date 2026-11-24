"""Streamlit dashboard for credit analysts.

  streamlit run src/app/ui.py
Upload an annual report (or pick a sample) -> summary, red flags by category, click a finding to see the page.
"""
import json, sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.app.pipeline import LOGS, ReportError, analyze, page_image  # noqa: E402

st.set_page_config(page_title="RedFlag Lens", layout="wide")
st.title("RedFlag Lens")
st.caption("First-pass red-flag screening of Indian annual reports. Every finding links to its page. "
           "Findings mean *review recommended*, not a judgement of fraud.")

SAMPLES = {p.stem: p for p in sorted((ROOT / "data/raw/inbox").glob("*.pdf"))}
with st.sidebar:
    up = st.file_uploader("Upload annual report (PDF)", type="pdf")
    sample = st.selectbox("…or pick a sample report", ["—"] + list(SAMPLES), index=(["—"] + list(SAMPLES)).index("GENSOL_FY2024") if "GENSOL_FY2024" in SAMPLES else 0)
    use_llm = st.checkbox("LLM summary (verified against evidence)", value=True)
    run = st.button("Analyse", type="primary")


@st.cache_data(show_spinner=False)
def run_analysis(data: bytes, name: str, use_llm: bool):
    return analyze(data, company=name, use_llm=use_llm)


if run:
    if up is not None:
        data, name = up.getvalue(), Path(up.name).stem
    elif sample != "—":
        data, name = SAMPLES[sample].read_bytes(), sample
    else:
        st.warning("Upload a PDF or pick a sample."); st.stop()
    with st.spinner("Reading the report… (about a minute for 300 pages)"):
        try:
            st.session_state.result = run_analysis(data, name, use_llm)
            st.session_state.pdf = data
        except ReportError as e:
            st.error(str(e)); st.stop()

r = st.session_state.get("result")
if not r:
    st.info("Pick **GENSOL_FY2024** (held out from training; the company collapsed in 2025) and press **Analyse**.")
    st.stop()

c0, c1, c2, c3, c4 = st.columns(5)
c0.metric("Severe warning signs", r.get("n_severe", 0),
          help="Going concern, default, audit qualification, insolvency, fraud, rating downgrade and similar "
               "early-warning signals. Healthy companies in our backtest have 0-1.")
c1.metric("Red flags (high confidence)", r["n_findings"],
          help=f"{r['n_review']} more in the review tier ({r.get('n_demoted', 0)} of them are template/boilerplate text)")
c2.metric("Specific (events, amounts)", r["n_specific"])
c3.metric("Sentences read", f"{r['sentences_analysed']:,}")
c4.metric("Time", f"{r['total_s']:.0f} s")

if r["message"]:
    st.success(r["message"])

st.subheader("Summary")
src = {"llm_verified": "LLM summary; every claim checked against the cited sentences",
       "template": "Built directly from the findings (LLM unavailable)"}.get(r["summary_source"], "")
st.caption(src)
for c in r["summary"]:
    pages = ", ".join(f"p.{f['page']}" for f in r["findings"] if f["id"] in c["evidence"])
    st.markdown(f"- {c['claim']}  \n  <small>evidence: {pages}</small>", unsafe_allow_html=True)

st.subheader("Red flags by category")
if r["by_category"]:
    st.bar_chart(pd.Series(r["by_category"]).sort_values(ascending=False))

st.subheader("Findings")
df = pd.DataFrame(r["findings"])
if df.empty:
    st.stop()
if "severe" not in df:
    df["severe"] = False
f1, f2, f3, f4 = st.columns(4)
cats = f1.multiselect("Category", sorted(df.category.unique()), default=sorted(df.category.unique()))
spec_only = f2.checkbox("Specific only", value=True)
severe_only = f4.checkbox("Severe only", value=False)
show_review = f3.checkbox(f"Include review tier (+{r['n_review']})", value=False,
                          help="Lower-confidence findings kept so that recall stays >= 0.85")
view = df[df.category.isin(cats) & (show_review | (df.tier == "high"))]
if severe_only:
    view = view[view.severe]
if spec_only:
    view = view[view.specificity == "Specific"]
view = view.reset_index(drop=True)
sel = st.dataframe(view.assign(severe=view.severe.map({True: "⚠", False: ""}))[["severe", "page", "section", "category", "specificity", "score", "text"]], use_container_width=True,
                   hide_index=True, on_select="rerun", selection_mode="single-row")

rows = sel.selection.rows if sel and sel.selection else []
if rows:
    f = view.iloc[rows[0]]
    left, right = st.columns([3, 2])
    left.image(page_image(st.session_state.pdf, int(f.page), f.text), caption=f"Page {f.page}")
    right.markdown(f"**{f.category} · {f.specificity}** (confidence {f.score:.2f})")
    right.write(f.text)
    if f.amounts:
        right.write("Amounts: " + ", ".join(f.amounts))
    b1, b2 = right.columns(2)
    for col, vote, label in [(b1, 1, "👍 Correct"), (b2, -1, "👎 Not a red flag")]:
        if col.button(label):
            LOGS.mkdir(exist_ok=True)
            with open(LOGS / "feedback.jsonl", "a") as fh:
                fh.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "company": r["company"],
                                     "page": int(f.page), "finding_text": f.text, "vote": vote}) + "\n")
            right.success("Saved. Feedback becomes new training labels.")
