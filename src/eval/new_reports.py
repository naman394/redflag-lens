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


def main():
    model = os.environ.get("REDFLAG_MODEL", "finbert")
    out = ROOT / "reports/backtest/new_unseen" / ("" if model == "finbert" else model)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, group in REPORTS:
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
