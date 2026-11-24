"""Build the v1.0 labelling sets (fragments excluded, company-wise splits kept).

  trainval_new.jsonl  1,900 new sentences from train + val companies (the 300 pilot sentences are reused on top)
  test.jsonl          300 sentences from the 5 test companies (never shown to the training labeller)

Sampling per set: 35% high-signal sections, 25% risk-keyword hits elsewhere, 40% random.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SENT = ROOT / "data/interim/sentences.jsonl"
OUT = ROOT / "data/labels"
SEED = 7
HIGH_SIGNAL = {"AUDITOR_REPORT", "CARO", "IFC_REPORT", "NOTES_CONTINGENT", "SECRETARIAL_AUDIT"}
COLS = ["sentence_id", "company", "fy", "group", "split", "page", "section", "stratum", "text"]


def split_of(company: str) -> str:
    for s in ["train", "val", "test", "backtest"]:
        if company in (ROOT / f"data/splits/{s}_companies.txt").read_text().split():
            return s
    raise ValueError(company)


def stratified(pool: pd.DataFrame, n: int) -> pd.DataFrame:
    hs = pool[pool.section.isin(HIGH_SIGNAL)].sample(int(n * 0.35), random_state=SEED)
    rest = pool.drop(hs.index)
    kw = rest[rest.kw_hit].sample(int(n * 0.25), random_state=SEED)
    rnd = rest.drop(kw.index).sample(n - len(hs) - len(kw), random_state=SEED)
    out = pd.concat([hs.assign(stratum="high_signal"), kw.assign(stratum="keyword"), rnd.assign(stratum="random")])
    return out.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    df = pd.read_json(SENT, lines=True)
    df = df[~df.is_fragment]
    df["split"] = df.company.map({c: split_of(c) for c in df.company.unique()})
    pilot_ids = {json.loads(l)["sentence_id"] for l in open(ROOT / "data/annotations/pilot_300_labelled.jsonl")}

    OUT.mkdir(parents=True, exist_ok=True)
    tv = stratified(df[df.split.isin(["train", "val"]) & ~df.sentence_id.isin(pilot_ids)], 1900)
    te = stratified(df[df.split == "test"], 300)
    tv[COLS].to_json(OUT / "trainval_new.jsonl", orient="records", lines=True, force_ascii=False)
    te[COLS].to_json(OUT / "test.jsonl", orient="records", lines=True, force_ascii=False)
    for name, d in [("trainval_new", tv), ("test", te)]:
        print(name, len(d), d.split.value_counts().to_dict(), d.stratum.value_counts().to_dict())


if __name__ == "__main__":
    main()
