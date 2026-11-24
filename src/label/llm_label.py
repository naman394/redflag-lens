"""Label a .jsonl of sentences with a Gemini model, applying guide.md. Resumable and parallel.

  python src/label/llm_label.py data/labels/trainval_new.jsonl --model gemini-4-neon
  -> data/labels/trainval_new.gemini-4-neon.jsonl   (one label per sentence_id)

The model sees only the guide, the section name and the sentence text, never other annotators' labels.
Needs GEMINI_API_KEY (see .env).
"""
import argparse, json, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal

from google import genai
from google.genai import types
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
GUIDE = (ROOT / "data/annotations/guide.md").read_text()
BATCH = 10

SYSTEM = f"""You are an annotator for a dataset of sentences from Indian listed-company annual reports.
Apply the annotation guide below exactly. Label each sentence independently, using only the sentence
text and its section name. If is_risk = 0, category and specificity must be "None".
Use notes for BAD_EXTRACTION or a secondary category; otherwise leave notes empty.

<guide>
{GUIDE}
</guide>"""


class Label(BaseModel):
    sentence_id: str
    is_risk: int
    category: Literal["Financial", "Legal", "Regulatory", "Operational", "Business", "None"]
    specificity: Literal["Specific", "Generic", "None"]
    notes: str


class Labels(BaseModel):
    labels: List[Label]


def label_batch(client, model: str, rows: list[dict]) -> list[dict]:
    items = "\n".join(json.dumps({k: r[k] for k in ("sentence_id", "section", "text")}, ensure_ascii=False) for r in rows)
    for attempt in range(4):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=f"Label every sentence below (one label per sentence_id).\n\n{items}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM, response_mime_type="application/json", response_schema=Labels),
            )
            parsed = resp.parsed if resp.parsed is not None else Labels.model_validate_json(resp.text)
            got = {l.sentence_id: l for l in parsed.labels}
            want = [r["sentence_id"] for r in rows]
            if set(want) - set(got) or any(got[s].is_risk not in (0, 1) for s in want):
                raise ValueError("missing or invalid labels")
            out = []
            for s in want:
                d = got[s].model_dump()
                if d["is_risk"] == 0:
                    d["category"], d["specificity"] = "None", "None"
                out.append(d)
            return out
        except Exception as e:  # network, rate limit or schema problem: back off and retry
            if attempt == 3:
                raise
            time.sleep(2 ** attempt * 3)
            print(f"  retry {attempt + 1}: {type(e).__name__}: {str(e)[:120]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    src = Path(a.infile)
    out_path = src.with_name(f"{src.stem}.{a.model}.jsonl")
    rows = [json.loads(l) for l in open(src)]
    done = {json.loads(l)["sentence_id"] for l in open(out_path)} if out_path.exists() else set()
    todo = [r for r in rows if r["sentence_id"] not in done]
    batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]
    print(f"{len(rows)} sentences, {len(done)} already labelled, {len(batches)} batches to run with {a.model}")

    client = genai.Client()
    lock = threading.Lock()
    n = len(done)
    with open(out_path, "a") as out, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(label_batch, client, a.model, b) for b in batches]
        for f in as_completed(futs):
            res = f.result()
            with lock:
                for d in res:
                    d["annotator"] = a.model
                    out.write(json.dumps(d, ensure_ascii=False) + "\n")
                out.flush()
                n += len(res)
                if n % 200 < BATCH:
                    print(f"  {n}/{len(rows)}")
    print(f"done -> {out_path}")


if __name__ == "__main__":
    main()
