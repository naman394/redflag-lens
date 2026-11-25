"""Fresh test on two annual reports that were never in any split (no training, validation or test use).

SPICEJET_FY2023 (airline in financial distress) and NESTLEIND_FY2024 (healthy). Both PDFs were downloaded
from BSE / NSE into data/raw/new_test/ (git-ignored, like data/raw/inbox/). Uses the app model (REDFLAG_MODEL,
default finbert; other models write to a subfolder) and no LLM.
  python src/eval/new_reports.py
Writes reports/backtest/new_unseen/<report>.json and summary.csv
"""
import json, os, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.app.pipeline import analyze  # noqa: E402

REPORTS = [("SPICEJET_FY2023", "distressed (new, unseen)"), ("NESTLEIND_FY2024", "healthy (new, unseen)")]
# Second fresh batch (NEW_BATCH=2): 2 distressed + 2 healthy, none in any split or earlier test.
REPORTS_2 = [("FRETAIL_FY2020", "distressed (new, unseen)"), ("IDEA_FY2020", "distressed (new, unseen)"),
             ("BAJAJAUTO_FY2024", "healthy (new, unseen)"), ("BRITANNIA_FY2024", "healthy (new, unseen)")]
# Third batch (NEW_BATCH=3): clean exam for severity v2. Rules frozen before these reports were opened.
REPORTS_3 = [("SINTEX_FY2020", "distressed (new, unseen)"), ("RELINFRA_FY2020", "distressed (new, unseen)"),
             ("HEROMOTOCO_FY2024", "healthy (new, unseen)"), ("DABUR_FY2023", "healthy (new, unseen)")]


def main():
    model = os.environ.get("REDFLAG_MODEL", "finbert")
    batch = os.environ.get("NEW_BATCH", "1")
    out = ROOT / "reports/backtest" / {"1": "new_unseen", "2": "new_unseen_2", "3": "new_unseen_3"}[batch] \
        / ("" if model == "finbert" else model)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, group in {"1": REPORTS, "2": REPORTS_2, "3": REPORTS_3}[batch]:
        r = analyze((ROOT / f"data/raw/new_test/{name}.pdf").read_bytes(), company=name, use_llm=False)
        (out / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        n = r["sentences_analysed"]
        rows.append({"report": name, "group": group, "sentences": n, "flags": r["n_findings"],
                     "flags_per_1000": round(1000 * r["n_findings"] / n, 1), "severe": r["n_severe"],
                     "severe_per_1000": round(1000 * r["n_severe"] / n, 2), "seconds": r["total_s"]})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)


if __name__ == "__main__":
    main()
