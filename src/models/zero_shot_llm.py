"""M3 candidate: zero-shot LLM (the "why not just ChatGPT?" baseline).

Labels the test set with gemini-4-neon given only guide.md (same setup as the training labeller, so this is
the ceiling the trained models are trying to reach cheaply). Also records latency and estimated cost per report.
  python src/models/zero_shot_llm.py
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load, log_run, metrics, save_preds  # noqa: E402

MODEL = "gemini-4-neon"


def main():
    src = ROOT / "data/labels/test.jsonl"
    out = ROOT / f"data/labels/test.{MODEL}.jsonl"
    t = time.perf_counter()
    if not out.exists():
        subprocess.run([sys.executable, str(ROOT / "src/label/llm_label.py"), str(src), "--model", MODEL], check=True)
    secs = time.perf_counter() - t
    pred = pd.read_json(out, lines=True).drop_duplicates("sentence_id", keep="last").set_index("sentence_id")
    d = load("test")
    p = pred.loc[d.sentence_id, "is_risk"].astype(int).values
    pc = np.where(p == 1, pred.loc[d.sentence_id, "category"].values, "None")
    ps = np.where(p == 1, pred.loc[d.sentence_id, "specificity"].values, "None")
    m = metrics(d.is_risk, p, p.astype(float), d.category, pc, d.specificity, ps)
    save_preds("zero_shot_llm", d, p, p.astype(float), pc, ps)
    sents_per_report = 3038  # median non-fragment sentences per report in this corpus
    note = f"{MODEL}; {secs:.0f}s for {len(d)} sentences (6 parallel workers, 0 if cached); ~{sents_per_report // 10} API calls per full report"
    log_run("zero_shot_llm", {"model": MODEL, "batch": 10, "prompt": "guide v1.2"}, {"test": m}, note)
    print(json.dumps(m, indent=1), "\n", note)


if __name__ == "__main__":
    main()
