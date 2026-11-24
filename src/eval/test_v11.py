"""Dataset v1.1: freeze the 600-sentence test set and re-score every model on it.

Test extension labels (300 new sentences from the same 5 test companies):
  annotator A = Claude (claude-opus-5-5), annotator B = Qwen2.5-32B-Instruct-AWQ (vLLM, temperature 0),
  tie-breaker C = an open 70B model from a different family (name in reports/label_agreement_v11.json),
  run only on the A/B disagreements. Same voting rule as consensus.py (v1.0).
Reports metrics on test600 (v1.0 test + extension) and on the extension alone (a fresh holdout: no model or
threshold was ever chosen while it existed). Bootstrap 95% CIs on F1.
  python src/eval/test_v11.py  ->  data/labels/v1.1/, reports/eval/test600_comparison.csv
"""
import json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models")); sys.path.insert(0, str(ROOT / "src/label"))
from common import CATS, REPORTS, metrics  # noqa: E402
from consensus import FIELDS, load as load_labels, vote  # noqa: E402

L = ROOT / "data/labels"
OUT = L / "v1.1"
rng = np.random.default_rng(0)


def ci(y, p, n=2000):
    y, p = np.asarray(y), np.asarray(p)
    b = [f1_score(y[i], p[i], zero_division=0) for i in (rng.integers(0, len(y), len(y)) for _ in range(n))]
    return f"{np.percentile(b, 2.5):.2f}–{np.percentile(b, 97.5):.2f}"


def build_labels():
    meta = pd.read_json(L / "test_ext.jsonl", lines=True).set_index("sentence_id")
    A, B = load_labels(L / "test_ext.claude.jsonl"), load_labels(L / "test_ext.qwen32b.jsonl")
    C = load_labels(L / "test_ext.tiebreak.jsonl")
    ids = meta.index
    stats = {"n_ext": len(ids), "annotator_A": "claude-opus-5-5", "annotator_B": "Qwen/Qwen2.5-32B-Instruct-AWQ",
             "tiebreaker_C": str(pd.read_json(L / "test_ext.tiebreak.jsonl", lines=True).annotator.iloc[0])}
    for f in FIELDS:
        stats[f"kappa_A_vs_B_{f}"] = round(cohen_kappa_score(A.loc[ids, f].astype(str), B.loc[ids, f].astype(str)), 3)
        stats[f"agree_A_vs_B_{f}"] = round(float((A.loc[ids, f].astype(str) == B.loc[ids, f].astype(str)).mean()), 3)
    rows, n_adj = [], 0
    for s in ids:
        la, lb = (tuple(str(X.loc[s, f]) for f in FIELDS) for X in (A, B))
        lc = tuple(str(C.loc[s, f]) for f in FIELDS) if s in C.index else la
        lab, adj = vote(la, lb, lc)
        n_adj += adj
        rows.append(meta.loc[s].to_dict() | {"sentence_id": s, "is_risk": int(lab[0]), "category": lab[1],
                                             "specificity": lab[2], "label_source": "ext-adjudicated" if adj else "ext-agreed"})
    ext = pd.DataFrame(rows)
    stats["ext_adjudicated"], stats["ext_risk"] = n_adj, int(ext.is_risk.sum())
    old = pd.read_json(L / "v1.0/test.jsonl", lines=True)
    test = pd.concat([old, ext[old.columns]], ignore_index=True)
    OUT.mkdir(exist_ok=True)
    for f in ["train.jsonl", "val.jsonl", "train_aug.jsonl"]:
        shutil.copy(L / "v1.0" / f, OUT / f)
    test.to_json(OUT / "test.jsonl", orient="records", lines=True, force_ascii=False)
    stats["test600_n"], stats["test600_risk"] = len(test), int(test.is_risk.sum())
    stats["test600_categories"] = test[test.is_risk == 1].category.value_counts().to_dict()
    (REPORTS / "label_agreement_v11.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=1))
    return test, len(old)


def main():
    test, n_old = build_labels()
    test["category"] = test.category.fillna("None").astype(str)
    y, yc = test.is_risk.values, test.category.values
    ext_mask = np.arange(len(test)) >= n_old
    preds = {}  # name -> (binary, score, category)

    loc = pd.read_csv(REPORTS / "preds/local_models_testext.csv")
    def old(name):
        d = pd.read_csv(REPORTS / f"preds/{name}_test.csv")
        return d.pred_risk.values, d.score.values, d.pred_category.fillna("None").astype(str).values
    kp, ks, kc = old("keyword")
    preds["Keyword rules"] = (np.r_[kp, loc.keyword_risk], np.r_[ks, loc.keyword_risk], np.r_[kc, loc.keyword_cat])
    tp, ts, tc = old("tfidf_lr")
    te = (loc.tfidf_score >= 0.5).astype(int)
    preds["TF-IDF + LR"] = (np.r_[tp, te], np.r_[ts, loc.tfidf_score], np.r_[tc, np.where(te, loc.tfidf_cat, "None")])
    zp, zs, zc = old("zero_shot_open_llm")
    zx = pd.read_csv(REPORTS / "preds/zero_shot_open_llm_testext.csv")
    preds["Zero-shot Qwen2.5-7B"] = (np.r_[zp, zx.pred_risk], np.r_[zs, zx.pred_risk], np.r_[zc, zx.pred_category.fillna("None")])
    fp, fs, fc = old("finbert")
    fe = (loc.finbert_score >= 0.5).astype(int)
    preds["FinBERT seed 42 (shipped)"] = (np.r_[fp, fe], np.r_[fs, loc.finbert_score], np.r_[fc, np.where(fe, loc.finbert_cat, "None")])
    for cfg, label in [("finbert", "FinBERT, 3-seed ens."), ("finbert_aug", "FinBERT + aug, 3-seed ens."),
                       ("debv3b", "DeBERTa-v3-base, 3-seed ens."), ("debv3b_aug", "DeBERTa-v3-base + aug, 3-seed ens."),
                       ("debv3l", "DeBERTa-v3-large, 3-seed ens."), ("debv3l_aug", "DeBERTa-v3-large + aug, 3-seed ens.")]:
        P = np.mean([np.r_[np.load(REPORTS / f"probs/{cfg}_s{s}_test.npy"), np.load(REPORTS / f"probs/{cfg}_s{s}_testext.npy")]
                     for s in (42, 1, 2)], 0)
        s = 1 - P[:, 0]; p = (s >= 0.5).astype(int)
        preds[label] = (p, s, np.where(p == 1, np.array(CATS)[P[:, 1:].argmax(1)], "None"))

    rows = []
    for name, (p, s, c) in preds.items():
        for subset, m in [("test600", np.ones(len(y), bool)), ("extension300", ext_mask), ("original300", ~ext_mask)]:
            c = pd.Series(c).fillna("None").astype(str).values
            r = metrics(y[m], p[m], s[m], yc[m], c[m])
            rows.append({"model": name, "subset": subset, "n_risk": int(y[m].sum()), "precision": r["risk_precision"],
                         "recall": r["risk_recall"], "f1": r["risk_f1"], "f1_ci95": ci(y[m], p[m]),
                         "pr_auc": r["risk_pr_auc"], "category_macro_f1": r["category_macro_f1"]})
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "eval/test600_comparison.csv", index=False)
    for sub in ["test600", "extension300"]:
        print(f"\n== {sub} ==\n" + df[df.subset == sub].drop(columns="subset").to_string(index=False))


if __name__ == "__main__":
    main()
