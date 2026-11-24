"""M3 candidate: zero-shot open-weights LLM (the "why not just ask an LLM?" baseline).

An open model (default Qwen2.5-7B-Instruct) gets guide v1.2 and batches of 10 test sentences, no training.
Runs on a GPU (RunPod RTX 4090 in our runs). Open weights matter for the use case: a bank can run it on its own
servers, so borrower documents never leave the building. Records latency so we can compare cost per report.
  python src/models/zero_shot_open_llm.py [--model Qwen/Qwen2.5-7B-Instruct]
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, load, log_run, metrics, save_preds  # noqa: E402

GUIDE = (ROOT / "data/annotations/guide.md").read_text()
CATS = {"Financial", "Legal", "Regulatory", "Operational", "Business"}
SYSTEM = ("You are an annotator for sentences from Indian annual reports. Apply this guide exactly.\n\n" + GUIDE +
          "\n\nReturn ONLY a JSON list, one object per sentence, in the same order: "
          '[{"id": <id>, "is_risk": 0 or 1, "category": "Financial|Legal|Regulatory|Operational|Business|None", '
          '"specificity": "Specific|Generic|None"}]')


def parse(text: str, ids: list) -> dict:
    m = re.search(r"\[.*\]", text, re.S)
    out = {}
    try:
        for o in json.loads(m.group(0)) if m else []:
            out[int(o["id"])] = o
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--infile", help="unlabelled .jsonl to score instead of the test split (predictions only)")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda").eval()

    import pandas as pd
    d = (pd.read_json(a.infile, lines=True) if a.infile else load("test")).reset_index(drop=True)
    prompts, groups = [], []
    for i in range(0, len(d), 10):
        g = list(range(i, min(i + 10, len(d))))
        items = "\n".join(json.dumps({"id": j, "section": d.section[j], "text": d.text[j]}, ensure_ascii=False) for j in g)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Label these sentences:\n" + items}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        groups.append(g)

    labels, failed = {}, 0
    t = time.perf_counter()
    for b in range(0, len(prompts), 8):
        enc = tok(prompts[b:b + 8], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=900, do_sample=False)
        for k, g in enumerate(groups[b:b + 8]):
            res = parse(tok.decode(gen[k][enc.input_ids.shape[1]:], skip_special_tokens=True), g)
            failed += sum(j not in res for j in g)
            labels.update(res)
    secs = time.perf_counter() - t

    p = np.array([int(labels.get(j, {}).get("is_risk", 0) in (1, "1")) for j in range(len(d))])
    cat = [str(labels.get(j, {}).get("category", "None")) for j in range(len(d))]
    spec = [str(labels.get(j, {}).get("specificity", "None")) for j in range(len(d))]
    pc = np.where(p == 1, [c if c in CATS else "Financial" for c in cat], "None")
    ps = np.where(p == 1, [s if s in ("Specific", "Generic") else "Specific" for s in spec], "None")
    if a.infile:  # extension set: save predictions, metrics are computed once its labels are frozen
        out = Path(__file__).resolve().parents[2] / "reports/preds/zero_shot_open_llm_testext.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"sentence_id": d.sentence_id, "pred_risk": p, "pred_category": pc, "pred_specificity": ps}).to_csv(out, index=False)
        print(f"{len(d)} scored -> {out}; {failed} unparsed; {secs:.0f}s")
        return
    m = metrics(d.is_risk, p, p.astype(float), d.category, pc, d.specificity, ps)
    save_preds("zero_shot_open_llm", d, p, p.astype(float), pc, ps)
    per_sent = secs / len(d)
    note = (f"{a.model} bf16 on {torch.cuda.get_device_name()}; {secs:.0f}s for {len(d)} sentences "
            f"({per_sent * 1000:.0f} ms/sentence, ~{per_sent * 3038 / 60:.1f} min per 3,038-sentence report); "
            f"{failed} sentences unparsed (counted as 0)")
    log_run("zero_shot_open_llm", {"model": a.model, "batch_sentences": 10, "prompt": "guide v1.2", "greedy": True}, {"test": m}, note)
    print(json.dumps(m, indent=1), "\n", note)


if __name__ == "__main__":
    main()
