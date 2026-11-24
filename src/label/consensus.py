"""Freeze dataset v1.0: training labels + an LLM-consensus test set.

Train / val labels
  pilot_300 (Claude, adjudicated) + trainval_new labelled by gemini-4-neon.
Test labels (5 held-out companies, never seen by the training labeller)
  annotator A = Claude (claude-opus-5-5), annotator B = gemini-4-argon, tie-breaker C = gemini-4-carbon.
  A and B agree -> that label. Otherwise majority of A, B, C ("adjudicated"). If all three differ -> C.

Output: data/labels/v1.0/{train,val,test}.jsonl and agreement stats in reports/label_agreement.json
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
L = ROOT / "data/labels"
OUT = L / "v1.0"
FIELDS = ["is_risk", "category", "specificity"]


def load(p: Path) -> pd.DataFrame:
    d = pd.read_json(p, lines=True, dtype={"sentence_id": str})
    d["is_risk"] = d.is_risk.astype(int)
    d.loc[d.is_risk == 0, ["category", "specificity"]] = "None"
    return d.drop_duplicates("sentence_id", keep="last").set_index("sentence_id")


def vote(a, b, c):
    if a == b:
        return a, False
    m, n = Counter([a, b, c]).most_common(1)[0]
    return (m if n >= 2 else c), True


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_json(L / "test.jsonl", lines=True).set_index("sentence_id")
    A, B, C = load(L / "test.claude.jsonl"), load(L / "test.gemini-4-argon.jsonl"), load(L / "test.gemini-4-carbon.jsonl")
    ids = meta.index
    stats = {"n_test": len(ids)}
    for f in FIELDS:
        stats[f"kappa_claude_vs_argon_{f}"] = round(cohen_kappa_score(A.loc[ids, f].astype(str), B.loc[ids, f].astype(str)), 3)
        stats[f"agree_claude_vs_argon_{f}"] = round(float((A.loc[ids, f].astype(str) == B.loc[ids, f].astype(str)).mean()), 3)

    rows, n_adj = [], 0
    for s in ids:
        # vote on the joint label so category/specificity stay consistent with is_risk
        la, lb, lc = (tuple(str(X.loc[s, f]) for f in FIELDS) for X in (A, B, C))
        lab, adj = vote(la, lb, lc)
        n_adj += adj
        r = meta.loc[s].to_dict() | {"sentence_id": s, "is_risk": int(lab[0]), "category": lab[1], "specificity": lab[2],
                                      "label_source": "adjudicated" if adj else "agreed"}
        rows.append(r)
    test = pd.DataFrame(rows)
    stats["test_adjudicated"] = n_adj
    stats["test_risk"] = int(test.is_risk.sum())

    pilot = pd.read_json(ROOT / "data/annotations/pilot_300_labelled.jsonl", lines=True)
    pilot["label_source"] = "claude-pilot-v1"
    new = pd.read_json(L / "trainval_new.jsonl", lines=True)
    neon = load(L / "trainval_new.gemini-4-neon.jsonl")
    new = new.join(neon[FIELDS], on="sentence_id", how="inner")
    new["label_source"] = "gemini-4-neon"
    tv = pd.concat([pilot, new], ignore_index=True)
    tv.loc[tv.is_risk.astype(int) == 0, ["category", "specificity"]] = "None"
    keep = ["sentence_id", "company", "fy", "group", "split", "page", "section", "stratum", "text", "is_risk", "category", "specificity", "label_source"]
    for name, d in [("train", tv[tv.split == "train"]), ("val", tv[tv.split == "val"]), ("test", test)]:
        d[keep].to_json(OUT / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)
        stats[f"{name}_n"], stats[f"{name}_risk"] = len(d), int(d.is_risk.astype(int).sum())
        stats[f"{name}_categories"] = d[d.is_risk.astype(int) == 1].category.value_counts().to_dict()
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/label_agreement.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
