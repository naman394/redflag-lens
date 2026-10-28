"""Step 6: agreement between the pilot labels (Claude) and a human's blind labels.

Usage:
  1. A team member fills data/annotations/pilot_100_blind_for_humans.xlsx (no peeking at pilot_300_labelled).
  2. Save it as data/annotations/pilot_100_human_<name>.xlsx
  3. python src/prep/06_agreement.py data/annotations/pilot_100_human_<name>.xlsx
Prints Cohen's kappa for is_risk, category and specificity, plus the disagreements to discuss.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]


def clean(x):
    x = str(x).strip()
    return {"": "None", "nan": "None", "-": "None", "1.0": "1", "0.0": "0"}.get(x, x).capitalize() if not x.isdigit() else x


def main(path: str) -> None:
    human = pd.read_excel(path) if path.endswith("xlsx") else pd.read_csv(path)
    human.columns = ["sentence_id", "section", "text", "is_risk", "category", "specificity", "annotator", "notes"][: len(human.columns)]
    human = human.dropna(subset=["is_risk"])
    ref = pd.read_json(ROOT / "data/annotations/pilot_300_labelled.jsonl", lines=True)
    m = human.merge(ref, on="sentence_id", suffixes=("_h", "_c"))
    print(f"compared sentences: {len(m)}")
    for f in ["is_risk", "category", "specificity"]:
        a = m[f + "_h"].map(clean); b = m[f + "_c"].astype(str).map(clean)
        k = cohen_kappa_score(a, b)
        print(f"  {f:12} kappa = {k:.3f}   raw agreement = {(a == b).mean():.1%}")
    print("\nis_risk confusion (rows = human, cols = pilot):")
    print(confusion_matrix(m.is_risk_h.astype(int), m.is_risk_c.astype(int)))
    d = m[(m.is_risk_h.astype(int) != m.is_risk_c.astype(int)) | (m.category_h.map(clean) != m.category_c.map(clean))]
    out = ROOT / "data/annotations/disagreements.csv"
    d[["sentence_id", "text_c", "is_risk_h", "is_risk_c", "category_h", "category_c", "specificity_h", "specificity_c", "notes_c"]].to_csv(out, index=False)
    print(f"\n{len(d)} disagreements -> {out} (discuss these, then update guide.md)")


if __name__ == "__main__":
    main(sys.argv[1])
