"""M3 iteration v10/v11: any Hugging Face encoder with the same 6-way head as finbert.py (None + 5 categories).

  python src/models/encoder.py --base microsoft/deberta-v3-large --name debv3l --seed 42 [--aug] [--save]
--aug adds data/labels/v1.0/train_aug.jsonl (real nil statements + template/boilerplate text from train companies,
all non-risk). Saves val/test probabilities to reports/probs/<name>_s<seed>_{val,test}.npy so seeds can be
ensembled (src/models/ensemble.py). Model selection uses VALIDATION only; test is reported, never used to choose.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from common import CATS, DATA, REPORTS, ROOT, Timer, load, log_run, metrics  # noqa: E402

LABELS = ["None"] + CATS
L2I = {l: i for i, l in enumerate(LABELS)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP = os.environ.get("ENCODER_AMP", "1") == "1"  # bf16 autocast; set ENCODER_AMP=0 for fp32 (DeBERTa NaNs on transformers 5)
if not AMP:
    torch.backends.cuda.matmul.allow_tf32 = True


def batches(tok, texts, labels=None, bs=16, shuffle=False, seed=0):
    dl = DataLoader(list(range(len(texts))), batch_size=bs, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))
    for b in dl:
        enc = tok([texts[i] for i in b], truncation=True, max_length=128, padding=True, return_tensors="pt").to(DEV)
        yield enc, (torch.tensor([labels[i] for i in b], device=DEV) if labels is not None else None)


@torch.no_grad()
def predict(model, tok, texts):
    model.eval()
    out = []
    for enc, _ in batches(tok, texts, bs=64):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda" and AMP):
            out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--aug", action="store_true")
    ap.add_argument("--save", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    name = a.name + ("_aug" if a.aug else "")

    splits = {s: load(s) for s in ["train", "val", "test"]}
    tr = splits["train"]
    if a.aug:
        ex = pd.read_json(DATA / "train_aug.jsonl", lines=True)
        ex["input"] = "[" + ex.section.astype(str) + "] " + ex.text.astype(str)
        ex["is_risk"], ex["category"] = 0, "None"
        tr = pd.concat([tr, ex[["input", "is_risk", "category"]]], ignore_index=True)
    texts = tr.input.tolist()
    y = [L2I[c if r else "None"] for c, r in zip(tr.category, tr.is_risk)]

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForSequenceClassification.from_pretrained(a.base, num_labels=len(LABELS), ignore_mismatched_sizes=True).to(DEV)
    counts = np.bincount(y, minlength=len(LABELS))
    w = torch.tensor((len(y) / (len(LABELS) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float, device=DEV)
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(y) + a.bs - 1) // a.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    with Timer() as t_train:
        for ep in range(a.epochs):
            model.train(); tot = 0
            for enc, lab in batches(tok, texts, y, bs=a.bs, shuffle=True, seed=a.seed + ep):
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda" and AMP):
                    logits = model(**enc).logits
                loss = lossf(logits.float(), lab)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); tot += loss.item()
            pv = predict(model, tok, splits["val"].input.tolist())
            print(f"[{name} s{a.seed}] epoch {ep+1}: loss {tot:.2f} val "
                  f"{metrics(splits['val'].is_risk, (pv[:, 0] < 0.5).astype(int), 1 - pv[:, 0])}", flush=True)

    (REPORTS / "probs").mkdir(parents=True, exist_ok=True)
    res = {}
    for sp in ["val", "test"]:
        d = splits[sp]
        with Timer() as t:
            P = predict(model, tok, d.input.tolist())
        np.save(REPORTS / f"probs/{name}_s{a.seed}_{sp}.npy", P)
        s = 1 - P[:, 0]; p = (s >= 0.5).astype(int)
        pc = np.where(p == 1, np.array(CATS)[P[:, 1:].argmax(1)], "None")
        res[sp] = metrics(d.is_risk, p, s, d.category, pc)
        ms = round(t.s / len(d) * 1000, 2)
    ext = ROOT / "data/labels/test_ext.jsonl"  # v1.1 test extension: scored here, labels joined later
    if ext.exists():
        e = pd.read_json(ext, lines=True)
        np.save(REPORTS / f"probs/{name}_s{a.seed}_testext.npy",
                predict(model, tok, ("[" + e.section.astype(str) + "] " + e.text.astype(str)).tolist()))
    if a.save:
        out = ROOT / f"models/{name}"
        model.save_pretrained(out); tok.save_pretrained(out)
        (out / "meta.json").write_text(json.dumps({"labels": LABELS, "base": a.base, "seed": a.seed, "aug": a.aug}))
    log_run(name, {"base": a.base, "epochs": a.epochs, "lr": a.lr, "batch": a.bs, "seed": a.seed, "aug": a.aug,
                   "train_rows": len(y), "train_s": round(t_train.s, 1), "device": DEV, "ms_per_sentence": ms}, res)
    print("RESULT", json.dumps({"name": name, "seed": a.seed, **{k: v for k, v in res.items()}}), flush=True)


if __name__ == "__main__":
    main()
