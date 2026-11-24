"""Shared data loading, metrics and experiment logging for all models."""
import csv, json, subprocess, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/labels/v1.0"
REPORTS = ROOT / "reports"
CATS = ["Financial", "Legal", "Regulatory", "Operational", "Business"]
SEED = 42
DATA_VERSION = "v1.0"


def load(split: str) -> pd.DataFrame:
    d = pd.read_json(DATA / f"{split}.jsonl", lines=True)
    d["is_risk"] = d.is_risk.astype(int)
    d["category"] = d.category.fillna("None").astype(str)
    d["specificity"] = d.specificity.fillna("None").astype(str)
    d["input"] = "[" + d.section.astype(str) + "] " + d.text.astype(str)  # section is a useful signal
    return d


def metrics(y_risk, p_risk, score=None, y_cat=None, p_cat=None, y_spec=None, p_spec=None) -> dict:
    """Risk detection P/R/F1 (+PR-AUC), category macro-F1 over all rows (None counts as a class
    only through the risk gate), specificity F1 on true risk rows."""
    m = {
        "risk_precision": precision_score(y_risk, p_risk, zero_division=0),
        "risk_recall": recall_score(y_risk, p_risk, zero_division=0),
        "risk_f1": f1_score(y_risk, p_risk, zero_division=0),
    }
    if score is not None:
        m["risk_pr_auc"] = average_precision_score(y_risk, score)
    if y_cat is not None:
        present = [c for c in CATS if c in set(np.asarray(y_cat))]  # macro over categories that occur in this split
        m["category_macro_f1"] = f1_score(y_cat, p_cat, labels=present, average="macro", zero_division=0)
        both = (np.asarray(y_risk) == 1) & (np.asarray(p_risk) == 1)
        m["category_acc_on_detected"] = float((np.asarray(y_cat)[both] == np.asarray(p_cat)[both]).mean()) if both.any() else 0.0
    if y_spec is not None:
        mask = np.asarray(y_risk) == 1
        m["specificity_f1_specific"] = f1_score(np.asarray(y_spec)[mask], np.asarray(p_spec)[mask],
                                                labels=["Specific"], average="micro", zero_division=0)
    return {k: round(float(v), 4) for k, v in m.items()}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def log_run(model: str, params: dict, split_metrics: dict, notes: str = "") -> None:
    """Append one row per (run, split) to reports/experiments.csv."""
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / "experiments.csv"
    run_id = f"{model}-{datetime.now():%Y%m%d-%H%M%S}"
    new = not path.exists()
    cols = ["run_id", "date", "git_commit", "model", "data_version", "split", "params", "risk_precision", "risk_recall",
            "risk_f1", "risk_pr_auc", "category_macro_f1", "category_acc_on_detected", "specificity_f1_specific", "notes"]
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        for split, m in split_metrics.items():
            w.writerow({"run_id": run_id, "date": f"{datetime.now():%Y-%m-%d %H:%M}", "git_commit": git_commit(),
                        "model": model, "data_version": DATA_VERSION, "split": split, "params": json.dumps(params),
                        "notes": notes, **m})
    print(f"logged {run_id}")


def save_preds(model: str, df: pd.DataFrame, p_risk, score, p_cat, p_spec, split: str = "test") -> None:
    out = df[["sentence_id", "company", "section", "text", "is_risk", "category", "specificity"]].copy()
    out["pred_risk"], out["score"], out["pred_category"], out["pred_specificity"] = p_risk, score, p_cat, p_spec
    (REPORTS / "preds").mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / f"preds/{model}_{split}.csv", index=False)


class Timer:
    def __enter__(self):
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t
