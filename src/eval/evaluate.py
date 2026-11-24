"""M4 evaluation: comparison table with bootstrap CIs, error analysis, robustness, latency.

  python src/eval/evaluate.py
Reads reports/preds/*_test.csv (written by the model scripts). Writes reports/eval/.
"""
import json, random, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.app.pipeline import Classifier  # noqa: E402

OUT = ROOT / "reports/eval"
PREDS = ROOT / "reports/preds"
RNG = np.random.default_rng(42)


def boot_ci(y, p, fn, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(fn(y[i], p[i]))
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def add_nil_variants():
    """Improvement iteration: apply the nil-statement rule on top of each model's test predictions."""
    sys.path.insert(0, str(ROOT / "src/models"))
    from rules import is_nil_statement
    for f in sorted(PREDS.glob("*_test.csv")):
        if f.stem.endswith("_nil_test") or f.stem.startswith(("zero_shot", "keyword")):
            continue
        d = pd.read_csv(f)
        nil = d.text.map(is_nil_statement)
        d.loc[nil, "pred_risk"] = 0
        d.loc[nil, ["pred_category", "pred_specificity"]] = "None"
        d.to_csv(PREDS / f"{f.stem.replace('_test', '')}_nil_test.csv", index=False)


def comparison():
    add_nil_variants()
    rows = []
    for f in sorted(PREDS.glob("*_test.csv")):
        d = pd.read_csv(f)
        y, p = d.is_risk.values, d.pred_risk.values
        present = sorted(set(d.category[d.is_risk == 1]))
        rows.append({
            "model": f.stem.replace("_test", ""),
            "precision": round(precision_score(y, p, zero_division=0), 3),
            "recall": round(recall_score(y, p, zero_division=0), 3),
            "recall_95ci": boot_ci(y, p, lambda a, b: recall_score(a, b, zero_division=0)),
            "f1": round(f1_score(y, p, zero_division=0), 3),
            "f1_95ci": boot_ci(y, p, lambda a, b: f1_score(a, b, zero_division=0)),
            "category_macro_f1": round(f1_score(d.category.fillna("None"), d.pred_category.fillna("None"), labels=present, average="macro", zero_division=0), 3),
            "flagged": int(p.sum()), "true_risk": int(y.sum()),
        })
    t = pd.DataFrame(rows).sort_values("f1", ascending=False)
    t.to_csv(OUT / "comparison_test.csv", index=False)
    print(t.to_string(index=False))
    return t


NEG = re.compile(r"\b(no|not|nil|none|neither|nor|without)\b", re.I)
BOILER = re.compile(r"we are required|inherent limitation|reasonable assurance|responsib|key audit matter|"
                    r"in our (professional )?judgment|policy|is recognised|are recognised|refers to the risk", re.I)


def error_type(r) -> str:
    t = r.text
    if r.is_risk == 1 and r.pred_risk == 1:
        return "category_confusion" if r.category != r.pred_category else "correct"
    if r.pred_risk == 1:  # false positive
        if NEG.search(t):
            return "FP: negated / nil statement"
        if BOILER.search(t):
            return "FP: boilerplate / policy text"
        if re.search(r"exposed to|may|could|risk", t, re.I):
            return "FP: generic risk wording"
        return "FP: other"
    if r.is_risk == 1:  # false negative
        if len(t.split()) < 12 or not re.search(r"[.;:]$", t):
            return "FN: fragment / heading-like"
        if r.specificity == "Generic":
            return "FN: generic risk statement"
        return "FN: specific event missed"
    return "correct"


def error_analysis(model: str):
    d = pd.read_csv(PREDS / f"{model}_test.csv")
    d["error_type"] = d.apply(error_type, axis=1)
    counts = d.error_type.value_counts()
    errs = d[d.error_type != "correct"][["error_type", "section", "category", "pred_category", "score", "text"]]
    errs.sort_values("error_type").to_csv(OUT / f"errors_{model}.csv", index=False)
    cc = d[(d.is_risk == 1) & (d.pred_risk == 1)]
    conf = pd.crosstab(cc.category, cc.pred_category) if len(cc) else pd.DataFrame()
    print(f"\nerror types for {model}:\n{counts.to_string()}")
    return {"counts": counts.to_dict(), "category_confusion_on_detected": conf.to_dict()}


# ---------------------------------------------------------------- robustness
def perturb_rupee(t):
    return t.replace("₹", random.choice(["Rs.", "INR", "Rs", "`"]))


def perturb_typos(t, rate=0.04):
    chars = list(t)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and random.random() < rate:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def perturb_negate(t):
    """Turn an adverse fact into a CARO-style nil confirmation (label should flip to 0)."""
    return "According to the information and explanations given to us, there were no instances where " + t[0].lower() + t[1:]


def robustness(clf: Classifier, test: pd.DataFrame):
    random.seed(0)
    inp = lambda d: ("[" + d.section + "] " + d.text).tolist()  # noqa: E731
    base_s, _, _ = clf.predict(inp(test))
    base = (base_s >= clf.threshold).astype(int)
    res = {"model": clf.name, "threshold": clf.threshold}
    y = test.is_risk.values
    res["clean"] = {"recall": round(recall_score(y, base), 3), "precision": round(precision_score(y, base, zero_division=0), 3)}
    for name, fn in [("rupee_variants", perturb_rupee), ("typo_noise_4pct", perturb_typos)]:
        t2 = test.assign(text=test.text.map(fn))
        s, _, _ = clf.predict(inp(t2))
        p = (s >= clf.threshold).astype(int)
        res[name] = {"recall": round(recall_score(y, p), 3), "precision": round(precision_score(y, p, zero_division=0), 3),
                     "changed_predictions": int((p != base).sum())}
    money = test[test.text.str.contains("₹")]
    res["rupee_variants"]["sentences_with_rupee"] = int(len(money))
    risky = test[(test.is_risk == 1) & (test.specificity == "Specific")]
    s, _, _ = clf.predict(inp(risky.assign(text=risky.text.map(perturb_negate))))
    res["negation_flip"] = {"n": int(len(risky)), "still_flagged_after_negation": int((s >= clf.threshold).sum()),
                            "note": "lower is better: these became nil confirmations"}
    by_group = {}
    for g, d in test.groupby("group"):
        p = base[test.group.values == g]
        by_group[g] = {"n": int(len(d)), "risk": int(d.is_risk.sum()),
                       "recall": round(recall_score(d.is_risk, p, zero_division=0), 3),
                       "precision": round(precision_score(d.is_risk, p, zero_division=0), 3)}
    res["by_group"] = by_group
    by_co = {}
    for c, d in test.groupby("company"):
        p = base[test.company.values == c]
        by_co[c] = {"n": int(len(d)), "risk": int(d.is_risk.sum()), "flagged": int(p.sum()),
                    "recall": round(recall_score(d.is_risk, p, zero_division=0), 3) if d.is_risk.sum() else None}
    res["by_company_unseen_sectors"] = by_co
    return res


def latency(clf: Classifier, test: pd.DataFrame):
    texts = ("[" + test.section + "] " + test.text).tolist()
    clf.predict(texts[:16])
    t = time.perf_counter(); clf.predict(texts); s = time.perf_counter() - t
    return {"model": clf.name, "ms_per_sentence": round(s / len(texts) * 1000, 2),
            "est_classify_s_per_report_3038_sentences": round(s / len(texts) * 3038, 1)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = comparison()
    best = table.iloc[0].model
    out = {"comparison": table.to_dict("records"), "best_by_f1": best}
    for m in [x for x in ["finbert", "finbert_tuned", "tfidf_lr", "tfidf_lr_tuned"] if (PREDS / f"{x}_test.csv").exists()]:
        out[f"errors_{m}"] = error_analysis(m)
    test = pd.read_json(ROOT / "data/labels/v1.0/test.jsonl", lines=True)
    out["robustness"], out["latency"] = [], []
    for name in ["tfidf_lr", "finbert"]:
        if name == "finbert" and not (ROOT / "models/finbert").exists():
            continue
        clf = Classifier(name)
        out["robustness"].append(robustness(clf, test))
        out["latency"].append(latency(clf, test))
    (OUT / "eval_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({"robustness": out["robustness"], "latency": out["latency"]}, indent=1))


if __name__ == "__main__":
    main()
