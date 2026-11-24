"""Step 7: second, independent LLM annotator for the agreement check (Option A).

The pilot labels (pilot_300_labelled.*) came from one LLM pass. This script labels the same
100 blind sentences with a *different* model, using only guide.md (it never sees the pilot labels).
Then it computes LLM-LLM Cohen's kappa and writes the disagreements for a human to settle.

Report this as LLM-LLM agreement plus human adjudication of disagreements, NOT as human kappa.

The second annotator is a Google Gemini model (different vendor from the pilot annotator),
so the two annotators don't share the same blind spots.

Usage:
  pip install google-genai pandas scikit-learn openpyxl
  export GEMINI_API_KEY=...               # never commit the key
  python src/prep/07_llm_second_annotator.py --list-models    # find the exact model id
  python src/prep/07_llm_second_annotator.py --model <gemini-model-id>
  python src/prep/07_llm_second_annotator.py --model <gemini-model-id> --compare-only

Outputs (data/annotations/):
  pilot_100_llm2_<model>.csv           second annotator's labels
  disagreements_llm2_<model>.csv       sentences where the two LLMs differ -> human fills `human_*` columns
"""
import argparse
import json
from pathlib import Path
from typing import List, Literal

from google import genai
from google.genai import types
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data/annotations"
BATCH = 10

Category = Literal["Financial", "Legal", "Regulatory", "Operational", "Business", "None"]
Spec = Literal["Specific", "Generic", "None"]


class Label(BaseModel):
    sentence_id: str
    is_risk: int  # 0 or 1 (Gemini enums must be strings; checked below)
    category: Category
    specificity: Spec
    notes: str


class Labels(BaseModel):
    labels: List[Label]


SYSTEM = """You are an annotator for a dataset of sentences from Indian listed-company annual reports.
Apply the annotation guide below exactly. Label each sentence independently, using only the sentence
text and its section name. If is_risk = 0, category and specificity must be "None".
Use notes for BAD_EXTRACTION or a secondary category; otherwise leave notes empty.

<guide>
{guide}
</guide>"""


def label_batch(client, model, guide, rows):
    items = "\n".join(
        json.dumps({"sentence_id": r.sentence_id, "section": r.section, "text": r.text}, ensure_ascii=False)
        for r in rows.itertuples()
    )
    resp = client.models.generate_content(
        model=model,
        contents=f"Label every sentence below (one label per sentence_id).\n\n{items}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM.format(guide=guide),
            response_mime_type="application/json",
            response_schema=Labels,
        ),
    )
    parsed = resp.parsed if resp.parsed is not None else Labels.model_validate_json(resp.text)
    bad = [l.sentence_id for l in parsed.labels if l.is_risk not in (0, 1)]
    if bad:
        raise RuntimeError(f"is_risk not 0/1 for {bad[:3]}")
    got = {l.sentence_id: l for l in parsed.labels}
    missing = set(rows.sentence_id) - set(got)
    if missing:
        raise RuntimeError(f"model skipped {len(missing)} sentences: {sorted(missing)[:3]}")
    return [got[s].model_dump() for s in rows.sentence_id]


def norm(x):
    x = str(x).strip()
    if x in ("", "nan", "-"):
        return "None"
    if x in ("1.0", "0.0"):
        return x[0]
    return x if x.isdigit() else x.capitalize()


def compare(out_path, model_tag):
    a = pd.read_csv(out_path)
    ref = pd.read_json(ANN / "pilot_300_labelled.jsonl", lines=True)
    m = a.merge(ref, on="sentence_id", suffixes=("_llm2", "_llm1"))
    print(f"\ncompared sentences: {len(m)}  (llm1 = pilot labels, llm2 = {model_tag})")
    for f in ["is_risk", "category", "specificity"]:
        x, y = m[f + "_llm2"].map(norm), m[f + "_llm1"].map(norm)
        print(f"  {f:12} kappa = {cohen_kappa_score(x, y):.3f}   raw agreement = {(x == y).mean():.1%}")
    print("\nis_risk confusion (rows = llm2, cols = llm1):")
    print(confusion_matrix(m.is_risk_llm2.astype(int), m.is_risk_llm1.astype(int)))

    diff = (m.is_risk_llm2.map(norm) != m.is_risk_llm1.map(norm)) | (m.category_llm2.map(norm) != m.category_llm1.map(norm))
    d = m[diff][["sentence_id", "section_llm1" if "section_llm1" in m else "section", "text_llm1" if "text_llm1" in m else "text",
                 "is_risk_llm1", "category_llm1", "specificity_llm1", "is_risk_llm2", "category_llm2", "specificity_llm2"]].copy()
    d.columns = ["sentence_id", "section", "text", "is_risk_A", "category_A", "specificity_A", "is_risk_B", "category_B", "specificity_B"]
    for c in ["human_is_risk", "human_category", "human_specificity", "human_name", "human_notes"]:
        d[c] = ""
    dpath = ANN / f"disagreements_llm2_{model_tag}.csv"
    d.to_csv(dpath, index=False)
    print(f"\n{len(d)} disagreements -> {dpath}")
    print("A human fills the human_* columns for these only (~1 min each). Use the guide, pick A, B or neither.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Gemini model id, e.g. from --list-models")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--compare-only", action="store_true")
    args = ap.parse_args()
    if args.list_models:
        lc = genai.Client()
        for m in lc.models.list():
            if "gemini" in m.name:
                print(m.name.removeprefix("models/"))
        return
    if not args.model:
        ap.error("--model is required (run --list-models to see ids)")
    tag = args.model.replace("models/", "")
    out_path = ANN / f"pilot_100_llm2_{tag}.csv"

    if not args.compare_only:
        blind = pd.read_csv(ANN / "pilot_100_blind_for_humans.csv")
        blind = blind.rename(columns={blind.columns[0]: "sentence_id", blind.columns[1]: "section", blind.columns[2]: "text"})
        guide = (ANN / "guide.md").read_text()
        client = genai.Client()  # reads GEMINI_API_KEY
        results = []
        for i in range(0, len(blind), BATCH):
            rows = blind.iloc[i:i + BATCH]
            results += label_batch(client, args.model, guide, rows)
            print(f"labelled {len(results)}/{len(blind)}")
        out = pd.DataFrame(results).merge(blind[["sentence_id", "section", "text"]], on="sentence_id")
        out["annotator"] = args.model
        out.to_csv(out_path, index=False)
        print(f"saved -> {out_path}")

    compare(out_path, tag)


if __name__ == "__main__":
    main()
