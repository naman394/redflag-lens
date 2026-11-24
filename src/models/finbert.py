"""M3 candidate: fine-tuned FinBERT, one 6-way head (None + 5 risk categories).

is_risk = 1 - P(None); category = argmax over the 5 risk classes; specificity comes from the TF-IDF head.
  python src/models/finbert.py [--epochs 4]
Runs on Apple MPS if available, else CPU. Saves to models/finbert/.
"""
import argparse, json, sys
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent))
from baselines import pick_threshold, tf  # noqa: E402
from common import CATS, ROOT, SEED, Timer, load, log_run, metrics, save_preds  # noqa: E402

BASE = "ProsusAI/finbert"
LABELS = ["None"] + CATS
L2I = {l: i for i, l in enumerate(LABELS)}
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def batches(tok, texts, labels=None, bs=16, shuffle=False):
    idx = list(range(len(texts)))
    ds = DataLoader(idx, batch_size=bs, shuffle=shuffle, generator=torch.Generator().manual_seed(torch.initial_seed() % 2**31))
    for b in ds:
        enc = tok([texts[i] for i in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.to(DEV) for k, v in enc.items()}
        yield enc, (torch.tensor([labels[i] for i in b], device=DEV) if labels is not None else None)


@torch.no_grad()
def predict(model, tok, texts):
    model.eval()
    out = []
    for enc, _ in batches(tok, texts, bs=32):
        out.append(torch.softmax(model(**enc).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    tag_seed = "" if a.seed == SEED else f"_s{a.seed}"  # the default seed's run is the one shipped in the app

    splits = {s: load(s) for s in ["train", "val", "test"]}
    tr = splits["train"]
    y = [L2I[c if r else "None"] for c, r in zip(tr.category, tr.is_risk)]
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(LABELS), ignore_mismatched_sizes=True).to(DEV)

    counts = np.bincount(y, minlength=len(LABELS))
    w = torch.tensor((len(y) / (len(LABELS) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float, device=DEV)  # softened class weights
    lossf = torch.nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(y) + 15) // 16)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    texts = tr.input.tolist()
    for ep in range(a.epochs):
        model.train(); tot = 0
        for enc, lab in batches(tok, texts, y, shuffle=True):
            loss = lossf(model(**enc).logits, lab)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(); tot += loss.item()
        pv = predict(model, tok, splits["val"].input.tolist())
        m = metrics(splits["val"].is_risk, (pv[:, 0] < 0.5).astype(int), 1 - pv[:, 0])
        print(f"epoch {ep+1}: loss {tot:.2f}  val {m}")

    va = splits["val"]
    thr = pick_threshold(va.is_risk.values, 1 - predict(model, tok, va.input.tolist())[:, 0])
    print("threshold for val recall >= 0.85:", thr)
    tfidf = joblib.load(ROOT / "models/tfidf_lr.joblib")

    res = {}
    for name, d in splits.items():
        with Timer() as t:
            P = predict(model, tok, d.input.tolist())
        s = 1 - P[:, 0]
        cat_all = np.array(CATS)[P[:, 1:].argmax(1)]
        spec_all = tfidf["spec"].predict(tf(tfidf["vecs"], d.input))
        for tag, th in [("", 0.5), ("_tuned", thr)]:
            p = (s >= th).astype(int)
            pc, ps = np.where(p == 1, cat_all, "None"), np.where(p == 1, spec_all, "None")
            res[name + tag] = metrics(d.is_risk, p, s, d.category, pc, d.specificity, ps)
            if name == "test":
                save_preds(f"finbert{tag_seed}{tag}", d, p, s, pc, ps)
        print(f"  {name}: {len(d)} sentences in {t.s:.1f}s on {DEV} ({t.s/len(d)*1000:.1f} ms/sentence)")

    if not tag_seed:
        out = ROOT / "models/finbert"
        model.save_pretrained(out); tok.save_pretrained(out)
        (out / "meta.json").write_text(json.dumps({"labels": LABELS, "threshold": thr, "base": BASE}))
    log_run("finbert", {"base": BASE, "epochs": a.epochs, "lr": a.lr, "max_len": 128, "batch": 16, "threshold": thr,
                        "device": DEV, "seed": a.seed}, res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
