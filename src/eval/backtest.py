"""Report-level backtest: run the full pipeline on held-out reports and compare red-flag density.

GENSOL FY2024 was never used for training, validation or testing (the company collapsed in 2025).
JETAIRWAYS and YESBANK (distressed) and TCS and MARUTI (healthy) are test-split companies; RCOM (distressed),
TITAN and NTPC (healthy) are validation companies (used only for threshold choice, never for training).
Severe warning signs use the fixed lexicon in src/models/rules.py (written before this backtest was re-run).
  python src/eval/backtest.py
Writes reports/backtest/<company>.json and reports/backtest/summary.csv
"""
import json, os, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.app.pipeline import analyze  # noqa: E402

REPORTS = [("GENSOL_FY2024", "distressed (backtest, collapsed 2025)"), ("JETAIRWAYS_FY2018", "distressed (test)"),
           ("YESBANK_FY2019", "distressed (test)"), ("TCS_FY2019", "healthy (test)"), ("MARUTI_FY2019", "healthy (test)"),
           ("RCOM_FY2018", "distressed (val)"), ("TITAN_FY2019", "healthy (val)"), ("NTPC_FY2019", "healthy (val)")]


def main():
    model = os.environ.get("REDFLAG_MODEL", "finbert")
    out = ROOT / "reports/backtest" / ("" if model == "finbert" else model)  # other models write to a subfolder
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, group in REPORTS:
        r = analyze((ROOT / f"data/raw/inbox/{name}.pdf").read_bytes(), company=name, use_llm=False)
        (out / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
        rows.append({"report": name, "group": group, "pages": r["pages"], "sentences": r["sentences_analysed"],
                     "high_conf_flags": r["n_findings"], "specific_flags": r["n_specific"],
                     "flags_per_1000_sentences": round(1000 * r["n_findings"] / r["sentences_analysed"], 1),
                     "specific_per_1000": round(1000 * r["n_specific"] / r["sentences_analysed"], 1),
                     "severe_flags": r["n_severe"], "severe_per_1000": round(1000 * r["n_severe"] / r["sentences_analysed"], 2),
                     "demoted_template_or_boilerplate": r["n_demoted"],
                     "review_tier": r["n_review"], "seconds": r["total_s"], **{f"cat_{k}": v for k, v in r["by_category"].items()}})
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).fillna(0)
    df.to_csv(out / "summary.csv", index=False)
    (out / "auc.json").write_text(json.dumps(auc_table(df), indent=1))


def auc(df, col):
    """ROC-AUC of a per-report score for distressed vs healthy (share of pairs ordered correctly, ties = 0.5)."""
    d = df[df.group.str.startswith("distressed")][col]; h = df[df.group.str.startswith("healthy")][col]
    return round(sum((a > b) + 0.5 * (a == b) for a in d for b in h) / (len(d) * len(h)), 3)


def auc_table(df):
    unseen = df[~df.group.str.contains("val")]  # Gensol + test companies only: never used for any choice
    return {col: {"all_8": auc(df, col), "gensol_and_test_5": auc(unseen, col)}
            for col in ["flags_per_1000_sentences", "specific_per_1000", "severe_per_1000"]}


if __name__ == "__main__":
    main()
