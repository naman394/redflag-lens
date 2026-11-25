"""Option D check: two-step cascade on CPU. FinBERT scores every sentence; only sentences with FinBERT score >= t
are re-scored by the saved DeBERTa-v3-large + aug model (models/debv3l_aug); the rest are treated as non-risk.

Part 1: quality on test set v1.1 (600 sentences, 37 risk) for several t, against each model alone (threshold 0.5).
Part 2: share of sentences passed to DeBERTa and wall-clock time on full test-company reports.
  python src/eval/cascade.py
Writes reports/eval/cascade.json
"""
import gzip, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.app.pipeline import Classifier  # noqa: E402

TS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
REPORTS = ["TCS", "JETAIRWAYS"]


def prf(y, p):
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(P, 3), "recall": round(R, 3), "f1": round(2 * P * R / (P + R) if P + R else 0.0, 3),
            "tp": tp, "fp": fp, "fn": fn}


def timed(clf, x):
    t = time.time(); s = clf.predict(x)[0]; return s, time.time() - t


def main():
    fb, db = Classifier("finbert"), Classifier("deberta")
    assert fb.name == "finbert" and db.name == "deberta"
    d = pd.read_json(ROOT / "data/labels/v1.1/test.jsonl", lines=True)
    x = ("[" + d.section.astype(str) + "] " + d.text.astype(str)).tolist(); y = d.is_risk.values
    sf, _ = timed(fb, x); sd, _ = timed(db, x)
    out = {"test_v11": {"finbert_alone": prf(y, (sf >= 0.5).astype(int)), "deberta_alone": prf(y, (sd >= 0.5).astype(int)),
                        "cascade": {}}}
    for t in TS:
        final = np.where(sf >= t, sd, 0.0)
        out["test_v11"]["cascade"][str(t)] = {**prf(y, (final >= 0.5).astype(int)),
                                              "risks_lost_at_step1": int(((sf < t) & (y == 1)).sum()),
                                              "share_passed_on_test": round(float((sf >= t).mean()), 3)}
    print(json.dumps(out["test_v11"], indent=1), flush=True)

    sents = []
    with gzip.open(ROOT / "data/interim/sentences.jsonl.gz", "rt") as f:
        for line in f:
            s = json.loads(line)
            if s["company"] in REPORTS and not s.get("is_fragment"):
                sents.append(s)
    out["full_reports"] = {}
    for c in REPORTS:
        xs = [f"[{s['section']}] {s['text']}" for s in sents if s["company"] == c]
        sfr, tf = timed(fb, xs)
        row = {"sentences": len(xs), "finbert_s": round(tf, 1), "finbert_ms_per_sentence": round(1000 * tf / len(xs), 1),
               "share_passed": {str(t): round(float((sfr >= t).mean()), 3) for t in TS}}
        t0 = 0.05
        sel = [xx for xx, s in zip(xs, sfr) if s >= t0]
        _, td = timed(db, sel)
        row.update({f"deberta_on_passed_t{t0}_s": round(td, 1), "deberta_ms_per_sentence": round(1000 * td / max(len(sel), 1), 1),
                    f"cascade_total_t{t0}_s": round(tf + td, 1)})
        out["full_reports"][c] = row
        print(c, row, flush=True)
    (ROOT / "reports/eval/cascade.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
