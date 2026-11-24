"""M3 baselines: (1) keyword rules, (2) TF-IDF + logistic regression.

  python src/models/baselines.py
Saves models to models/tfidf_lr.joblib, predictions to reports/preds/, rows to reports/experiments.csv.
"""
import re, sys
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from common import CATS, ROOT, SEED, Timer, load, log_run, metrics, save_preds  # noqa: E402

TARGET_RECALL = 0.85

# --- 1. keyword rules: what an analyst does with Ctrl+F -------------------------------------------
RISK_KW = re.compile(
    r"going concern|default|overdue|penalt|show cause|litigation|lawsuit|suit filed|arbitration|claim|"
    r"NCLT|NCLAT|insolvency|IBC|winding up|fraud|wilful|invok|downgrad|impair|write.?off|provision for|"
    r"qualified opinion|emphasis of matter|material uncertainty|adverse|non.?compliance|contraven|"
    r"SEBI|RBI|enforcement directorate|investigation|search and seizure|disputed|demand notice|"
    r"breach|covenant|restructur|moratorium|NPA|non.?performing|liquidity|negative net worth|erosion|"
    r"pledge|resign|unsecured|guarantee", re.I)
CAT_KW = [
    ("Regulatory", r"SEBI|RBI|tax|GST|excise|customs|show cause|penalt|regulat|licen[cs]e|demand notice|enforcement"),
    ("Legal", r"court|suit|litigation|arbitration|claim|NCLT|NCLAT|insolvency|IBC|lawsuit|petition"),
    ("Operational", r"plant|fire|accident|cyber|IT system|internal control|employee|shutdown|supply"),
    ("Business", r"competition|demand|customer|price|market share|economy|macro"),
]


def kw_category(t: str) -> str:
    for c, pat in CAT_KW:
        if re.search(pat, t, re.I):
            return c
    return "Financial"


def run_keyword(splits):
    out = {}
    for name, d in splits.items():
        p = d.text.str.contains(RISK_KW).astype(int).values
        pc = np.where(p == 1, [kw_category(t) for t in d.text], "None")
        ps = np.where(p == 1, np.where(d.text.str.contains(r"₹|Rs\.|crore|lakh|million|\d{4}"), "Specific", "Generic"), "None")
        out[name] = metrics(d.is_risk, p, p.astype(float), d.category, pc, d.specificity, ps)
        if name == "test":
            save_preds("keyword", d, p, p.astype(float), pc, ps)
    log_run("keyword", {"rules": "RISK_KW regex, first-match category"}, out, "Ctrl+F baseline")
    return out


# --- 2. TF-IDF + logistic regression ----------------------------------------------------------------
def make_vec():
    return (TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True))


def fit_vec(vecs, texts):
    return hstack([v.fit_transform(texts) for v in vecs]).tocsr()


def tf(vecs, texts):
    return hstack([v.transform(texts) for v in vecs]).tocsr()


def pick_threshold(y, s, target=TARGET_RECALL):
    """Highest threshold whose recall on val is still >= target (keeps precision as high as possible)."""
    for t in np.round(np.arange(0.95, 0.04, -0.01), 2):
        if ((s >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) >= target:
            return float(t)
    return 0.05


def run_tfidf(splits):
    tr, va = splits["train"], splits["val"]
    vecs = make_vec()
    Xtr = fit_vec(vecs, tr.input)
    risk = LogisticRegression(C=4, class_weight="balanced", max_iter=2000, random_state=SEED).fit(Xtr, tr.is_risk)
    rtr = tr[tr.is_risk == 1]
    cat = LogisticRegression(C=4, class_weight="balanced", max_iter=2000, random_state=SEED).fit(tf(vecs, rtr.input), rtr.category)
    spec = LogisticRegression(C=4, class_weight="balanced", max_iter=2000, random_state=SEED).fit(tf(vecs, rtr.input), rtr.specificity)
    thr = pick_threshold(va.is_risk.values, risk.predict_proba(tf(vecs, va.input))[:, 1])
    print(f"tfidf threshold for val recall >= {TARGET_RECALL}: {thr}")

    res = {}
    for name, d in splits.items():
        with Timer() as t:
            X = tf(vecs, d.input)
            s = risk.predict_proba(X)[:, 1]
            pc_all, ps_all = cat.predict(X), spec.predict(X)
        for tag, th in [("", 0.5), ("_tuned", thr)]:
            p = (s >= th).astype(int)
            pc, ps = np.where(p == 1, pc_all, "None"), np.where(p == 1, ps_all, "None")
            res[name + tag] = metrics(d.is_risk, p, s, d.category, pc, d.specificity, ps)
            if name == "test":
                save_preds(f"tfidf_lr{tag}", d, p, s, pc, ps)
        print(f"  {name}: {len(d)} sentences in {t.s*1000:.0f} ms ({t.s/len(d)*1000:.3f} ms/sentence)")
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump({"vecs": vecs, "risk": risk, "cat": cat, "spec": spec, "threshold": thr}, ROOT / "models/tfidf_lr.joblib")
    log_run("tfidf_lr", {"word_ngrams": "1-2", "char_ngrams": "3-5", "C": 4, "class_weight": "balanced", "threshold": thr}, res)
    return res


if __name__ == "__main__":
    splits = {s: load(s) for s in ["train", "val", "test"]}
    for s, d in splits.items():
        print(s, len(d), "risk:", int(d.is_risk.sum()))
    import json
    print(json.dumps({"keyword": run_keyword(splits), "tfidf_lr": run_tfidf(splits)}, indent=1))
