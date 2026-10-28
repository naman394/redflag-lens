"""Step 4: company-wise split + pilot labelling sample.

- Every company is in exactly one split (no leakage of a company's boilerplate across splits).
- GENSOL is held out entirely as the backtest / demo case.
- Pilot sample (300 sentences) comes from train+val companies only; the test set is never shown to an LLM.
  Sampling: 40% high-signal sections, 20% risk-keyword hits elsewhere, 40% random (keeps realistic negatives).
- A 100-sentence subset is exported for blind human labelling (agreement check).
"""
import json, random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data/interim/sentences.jsonl"
SPLITS = ROOT / "data/splits"
ANN = ROOT / "data/annotations"
SEED = 42

SPLIT = {
    "test": ["JETAIRWAYS", "YESBANK", "TCS", "MARUTI", "SUNPHARMA"],
    "val": ["RCOM", "TITAN", "NTPC"],
    "backtest": ["GENSOL"],
}
HIGH_SIGNAL = {"AUDITOR_REPORT", "CARO", "IFC_REPORT", "NOTES_CONTINGENT", "SECRETARIAL_AUDIT"}


def main() -> None:
    df = pd.read_json(SENT, lines=True)
    assigned = {c: s for s, cs in SPLIT.items() for c in cs}
    df["split"] = df["company"].map(assigned).fillna("train")

    SPLITS.mkdir(parents=True, exist_ok=True)
    for s in ["train", "val", "test", "backtest"]:
        comps = sorted(df.loc[df.split == s, "company"].unique())
        (SPLITS / f"{s}_companies.txt").write_text("\n".join(comps) + "\n")
    summary = df.groupby(["split", "group"]).agg(companies=("company", "nunique"), sentences=("text", "size"))
    print(summary, "\n")
    summary.to_csv(SPLITS / "split_summary.csv")

    pool = df[df.split.isin(["train", "val"])]
    n = 300
    hs = pool[pool.section.isin(HIGH_SIGNAL)].sample(int(n * 0.4), random_state=SEED)
    rest = pool.drop(hs.index)
    kw = rest[rest.kw_hit].sample(int(n * 0.2), random_state=SEED)
    rnd = rest.drop(kw.index).sample(n - len(hs) - len(kw), random_state=SEED)
    pilot = pd.concat([hs.assign(stratum="high_signal"), kw.assign(stratum="keyword"), rnd.assign(stratum="random")])
    pilot = pilot.sample(frac=1, random_state=SEED).reset_index(drop=True)  # shuffle so strata are mixed

    cols = ["sentence_id", "company", "fy", "group", "split", "page", "section", "stratum", "text"]
    ANN.mkdir(parents=True, exist_ok=True)
    pilot[cols].to_json(ANN / "pilot_300_unlabelled.jsonl", orient="records", lines=True, force_ascii=False)

    # blind sheet for humans: first 100 of the shuffled pilot, no model labels shown
    sheet = pilot[["sentence_id", "section", "text"]].head(100).copy()
    for c in ["is_risk (1/0)", "category (Financial/Legal/Regulatory/Operational/Business/None)", "specificity (Generic/Specific/None)", "annotator", "notes"]:
        sheet[c] = ""
    sheet.to_excel(ANN / "pilot_100_blind_for_humans.xlsx", index=False)
    sheet.to_csv(ANN / "pilot_100_blind_for_humans.csv", index=False)
    print(pilot.stratum.value_counts().to_string())
    print(f"\npilot -> {ANN/'pilot_300_unlabelled.jsonl'}; blind sheet -> pilot_100_blind_for_humans.xlsx")


if __name__ == "__main__":
    main()
