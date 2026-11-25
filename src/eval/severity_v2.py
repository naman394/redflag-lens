"""Before/after for the severity lexicon v2 (RBI EWS additions + context filters), without re-running the models.

Severity is a text rule applied to the classifier's high-confidence findings, so re-applying it to the saved
cascade outputs gives exactly what a full re-run would. Compares v1 vs v2 severe-per-1,000 and the
distressed-vs-healthy ROC-AUC on: the 8-report backtest, and the 6 fresh reports (2 + 4).

NOTE on honesty: the v2 rules were written from RBI's EWS list and checked on TRAINING-company sentences only
(see docs), but two of the false-alarm types (business transfer "as a going concern", merger at NCLT) were first
noticed on fresh reports. The fresh 6 are therefore development data for v2; a clean exam needs new reports.

Run: .venv/bin/python src/eval/severity_v2.py   -> reports/eval/severity_v2.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/models"))
from rules import is_severe, is_severe_v1  # noqa: E402

SETS = {
    "backtest_8": (ROOT / "reports/backtest/cascade", None),
    "fresh_6": (None, [("reports/backtest/new_unseen/cascade", "SPICEJET_FY2023", "distressed"),
                       ("reports/backtest/new_unseen/cascade", "NESTLEIND_FY2024", "healthy"),
                       ("reports/backtest/new_unseen_2/cascade", "FRETAIL_FY2020", "distressed"),
                       ("reports/backtest/new_unseen_2/cascade", "IDEA_FY2020", "distressed"),
                       ("reports/backtest/new_unseen_2/cascade", "BAJAJAUTO_FY2024", "healthy"),
                       ("reports/backtest/new_unseen_2/cascade", "BRITANNIA_FY2024", "healthy")]),
    # clean exam: downloaded and run AFTER the v2 rules were frozen (rules.py md5 c1164931...)
    "clean_4": (None, [("reports/backtest/new_unseen_3/cascade", "SINTEX_FY2020", "distressed"),
                       ("reports/backtest/new_unseen_3/cascade", "RELINFRA_FY2020", "distressed"),
                       ("reports/backtest/new_unseen_3/cascade", "HEROMOTOCO_FY2024", "healthy"),
                       ("reports/backtest/new_unseen_3/cascade", "DABUR_FY2023", "healthy")]),
}


def auc(df, col):
    d, h = df[df.group == "distressed"][col], df[df.group == "healthy"][col]
    pairs = [(a > b) + 0.5 * (a == b) for a in d for b in h]
    return round(sum(pairs) / len(pairs), 3)


def row(path, name, group):
    r = json.loads(path.read_text())
    high = [f for f in r["findings"] if f["tier"] == "high"]
    n = r["sentences_analysed"]
    v1 = [f for f in high if is_severe_v1(f["text"])]
    v2 = [f for f in high if is_severe(f["text"])]
    return {"report": name, "group": group, "sentences": n, "severe_v1": len(v1), "severe_v2": len(v2),
            "v1_per_1000": round(1000 * len(v1) / n, 2), "v2_per_1000": round(1000 * len(v2) / n, 2),
            "added": [f"p{f['page']}: {f['text'][:160]}" for f in v2 if f not in v1],
            "removed": [f"p{f['page']}: {f['text'][:160]}" for f in v1 if f not in v2]}


def main():
    out = {}
    bt = pd.read_csv(ROOT / "reports/backtest/cascade/summary.csv")
    rows = [row(ROOT / f"reports/backtest/cascade/{r.report}.json", r.report,
                "distressed" if r.group.startswith("distressed") else "healthy") for r in bt.itertuples()]
    fresh = [row(ROOT / d / f"{n}.json", n, g) for d, n, g in SETS["fresh_6"][1]]
    clean = [row(ROOT / d / f"{n}.json", n, g) for d, n, g in SETS["clean_4"][1]]
    for key, rs in [("backtest_8", rows), ("fresh_6", fresh), ("clean_4", clean), ("all_fresh_10", fresh + clean)]:
        df = pd.DataFrame(rs)
        out[key] = {"auc_v1": auc(df, "v1_per_1000"), "auc_v2": auc(df, "v2_per_1000"), "reports": rs}
        print(f"\n== {key}: AUC v1 {out[key]['auc_v1']} -> v2 {out[key]['auc_v2']}")
        print(df[["report", "group", "severe_v1", "severe_v2", "v1_per_1000", "v2_per_1000"]].to_string(index=False))
    (ROOT / "reports/eval/severity_v2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
