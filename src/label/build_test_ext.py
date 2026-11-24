"""Test-set extension (dataset v1.1): 300 more sentences from the same 5 test companies, same stratified sampling
(35% high-signal sections / 25% keyword hits / 40% random), excluding the 300 already in the v1.0 test set.
  python src/label/build_test_ext.py  ->  data/labels/test_ext.jsonl
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import build_sets as B  # noqa: E402

B.SEED = 8  # different draw from v1.0 (seed 7)
df = pd.read_json(B.SENT, lines=True)
df = df[~df.is_fragment]
df["split"] = df.company.map({c: B.split_of(c) for c in df.company.unique()})
old = set(pd.read_json(B.OUT / "test.jsonl", lines=True).sentence_id)
pool = df[(df.split == "test") & ~df.sentence_id.isin(old)]
ext = B.stratified(pool, 300)
assert not set(ext.sentence_id) & old
ext[B.COLS].to_json(B.OUT / "test_ext.jsonl", orient="records", lines=True, force_ascii=False)
print(len(ext), ext.company.value_counts().to_dict(), ext.stratum.value_counts().to_dict())
