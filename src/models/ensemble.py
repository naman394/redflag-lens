"""Compare encoder configs and seed ensembles. Selection uses VALIDATION only.

For each config (model x aug) it reports mean +- sd over seeds and the 3-seed ensemble (mean of probabilities).
The winner is the config with the best validation PR-AUC (threshold-free); test numbers are printed for reporting,
not for choosing.  python src/models/ensemble.py  ->  reports/model_comparison.csv
"""
import re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import CATS, REPORTS, load, metrics  # noqa: E402

splits = {s: load(s) for s in ["val", "test"]}
runs = defaultdict(dict)
for f in sorted((REPORTS / "probs").glob("*_s*_val.npy")):
    name, seed = re.match(r"(.+)_s(\d+)_val\.npy", f.name).groups()
    runs[name][int(seed)] = {sp: np.load(REPORTS / f"probs/{name}_s{seed}_{sp}.npy") for sp in ["val", "test"]}


def score(P, d):
    s = 1 - P[:, 0]; p = (s >= 0.5).astype(int)
    pc = np.where(p == 1, np.array(CATS)[P[:, 1:].argmax(1)], "None")
    return metrics(d.is_risk, p, s, d.category, pc)


rows = []
for name, seeds in runs.items():
    for sp, d in splits.items():
        per = pd.DataFrame([score(r[sp], d) for r in seeds.values()])
        ens = score(np.mean([r[sp] for r in seeds.values()], 0), d)
        for k in ["risk_precision", "risk_recall", "risk_f1", "risk_pr_auc", "category_macro_f1"]:
            rows.append({"config": name, "split": sp, "metric": k, "n_seeds": len(seeds),
                         "mean": round(per[k].mean(), 3), "sd": round(per[k].std(ddof=0), 3), "ensemble": ens[k]})
df = pd.DataFrame(rows)
df.to_csv(REPORTS / "model_comparison.csv", index=False)
piv = df.pivot_table(index="config", columns=["split", "metric"], values=["mean", "ensemble"])
for sp in ["val", "test"]:
    t = pd.DataFrame({
        "seeds": df[df.split == sp].groupby("config").n_seeds.first(),
        "F1 mean±sd": df[(df.split == sp) & (df.metric == "risk_f1")].set_index("config").apply(lambda r: f"{r['mean']:.3f}±{r.sd:.3f}", axis=1),
        "F1 ens": df[(df.split == sp) & (df.metric == "risk_f1")].set_index("config").ensemble,
        "P ens": df[(df.split == sp) & (df.metric == "risk_precision")].set_index("config").ensemble,
        "R ens": df[(df.split == sp) & (df.metric == "risk_recall")].set_index("config").ensemble,
        "PR-AUC mean": df[(df.split == sp) & (df.metric == "risk_pr_auc")].set_index("config")["mean"],
        "PR-AUC ens": df[(df.split == sp) & (df.metric == "risk_pr_auc")].set_index("config").ensemble,
        "catF1 ens": df[(df.split == sp) & (df.metric == "category_macro_f1")].set_index("config").ensemble,
    })
    print(f"\n== {sp.upper()} ==\n" + t.sort_values("PR-AUC ens", ascending=False).to_string())
v = df[(df.split == "val") & (df.metric == "risk_pr_auc")].set_index("config")
print("\nSelected on validation PR-AUC (ensemble):", v.ensemble.idxmax(), "| single-model:", v["mean"].idxmax())
