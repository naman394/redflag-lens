"""Step 5: exploratory + statistical analysis for M2. Writes figures to reports/figures and numbers to reports/eda_stats.json."""
import json, re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/figures"
FIG.mkdir(parents=True, exist_ok=True)

# palette (validated reference palette, light mode)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HEALTHY, DISTRESSED = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150,
})

RED_FLAGS = {
    "Going concern doubt": r"material uncertainty|significant doubt|cast\w* doubt|ability to continue as a going concern",
    "Default / overdue": r"\bdefault(ed|s)?\b(?! risk)|overdue|delay(ed)? in (re)?payment|not been (deposited|repaid)",
    "Insolvency / NCLT": r"insolvency|\bNCLT\b|\bIBC\b|corporate insolvency|resolution professional|winding.?up",
    "Qualified / adverse opinion": r"qualified opinion|basis for qualified|adverse opinion|disclaimer of opinion|except for the (possible )?effects",
    "Penalty / show cause": r"penalt(y|ies) (of|was|has been|imposed|levied)|show.?cause|imposed a (fine|penalty)",
    "Fraud reported": r"fraud (was|has been|on the company|by the company) (noticed|reported|detected|identified)|forensic audit",
    "Rating downgrade": r"downgrad|default grade|rating .{0,40}\bD\b",
}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)


def main():
    out = {}
    man = pd.read_csv(ROOT / "data/raw/manifest.csv")
    sent = pd.read_json(ROOT / "data/interim/sentences.jsonl", lines=True)
    split_of = {}
    for s in ["train", "val", "test", "backtest"]:
        for c in (ROOT / f"data/splits/{s}_companies.txt").read_text().split():
            split_of[c] = s
    sent["split"] = sent.company.map(split_of)
    sent["report"] = sent.company + "_" + sent.fy
    pilot = pd.read_json(ROOT / "data/annotations/pilot_300_labelled.jsonl", lines=True)

    # ---------- corpus numbers ----------
    out["corpus"] = {
        "reports": int(len(man)), "companies": int(man.company.nunique()),
        "distressed_reports": int((man.group == "distressed").sum()), "healthy_reports": int((man.group == "healthy").sum()),
        "pages": int(man.pages.sum()), "pdf_mb": float(man.size_mb.sum()),
        "pages_without_text_pct": float(round(man.pct_pages_no_text.mean(), 2)),
        "sentences": int(len(sent)), "median_words": float(sent.n_words.median()),
        "pct_with_money": float(round(100 * sent.has_money.mean(), 1)), "pct_with_date": float(round(100 * sent.has_date.mean(), 1)),
        "pct_kw_hit": float(round(100 * sent.kw_hit.mean(), 1)),
    }
    split_tbl = sent.groupby(["split", "group"]).agg(companies=("company", "nunique"), reports=("report", "nunique"), sentences=("text", "size")).reset_index()
    split_tbl.to_csv(ROOT / "data/splits/split_summary.csv", index=False)
    out["splits"] = split_tbl.to_dict(orient="records")

    # ---------- Fig 1: sentences per report ----------
    per = sent.groupby(["report", "group"]).size().reset_index(name="n").sort_values("n")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(per.report, per.n, color=[DISTRESSED if g == "distressed" else HEALTHY for g in per.group], height=0.7)
    ax.set_title("Sentences extracted per annual report")
    ax.set_xlabel("sentences after cleaning"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=HEALTHY), plt.Rectangle((0, 0), 1, 1, color=DISTRESSED)],
              labels=["Healthy", "Distressed"], loc="lower right")
    save(fig, "01_sentences_per_report.png")

    # ---------- Fig 2: sentences by section ----------
    sec = sent.section.value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(sec.index, sec.values, color=HEALTHY, height=0.7)
    for y, v in enumerate(sec.values):
        ax.text(v, y, f" {v:,}", va="center", color=INK2, fontsize=9)
    ax.set_title("Where the text comes from (detected report section)")
    ax.set_xlabel("sentences"); ax.grid(axis="y", visible=False)
    save(fig, "02_sentences_by_section.png")
    out["sentences_by_section"] = sec.sort_values(ascending=False).to_dict()

    # ---------- Fig 3: sentence length ----------
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.hist(sent.n_words.clip(upper=80), bins=40, color=HEALTHY, edgecolor=SURFACE, linewidth=1)
    ax.axvline(sent.n_words.median(), color=INK2, linestyle="--", linewidth=1.2)
    ax.text(sent.n_words.median() + 1, ax.get_ylim()[1] * 0.9, f"median {sent.n_words.median():.0f} words", color=INK2)
    ax.set_title("Sentence length"); ax.set_xlabel("words per sentence (80+ grouped)"); ax.set_ylabel("sentences")
    save(fig, "03_sentence_length.png")

    # ---------- Red-flag phrase rates: distressed vs healthy ----------
    rows = []
    for rep, g in sent.groupby("report"):
        r = {"report": rep, "group": g.group.iloc[0], "n": len(g)}
        for name, pat in RED_FLAGS.items():
            r[name] = int(g.text.str.contains(pat, case=False, regex=True).sum())
        r["total_flags"] = sum(r[k] for k in RED_FLAGS)
        r["flags_per_1k"] = 1000 * r["total_flags"] / r["n"]
        rows.append(r)
    rf = pd.DataFrame(rows)
    rf.to_csv(ROOT / "reports/red_flag_phrases_per_report.csv", index=False)
    d, h = rf[rf.group == "distressed"], rf[rf.group == "healthy"]
    mw = stats.mannwhitneyu(d.flags_per_1k, h.flags_per_1k, alternative="greater")
    out["red_flag_phrases"] = {
        "median_per_1k_distressed": round(float(d.flags_per_1k.median()), 2),
        "median_per_1k_healthy": round(float(h.flags_per_1k.median()), 2),
        "mann_whitney_U": float(mw.statistic), "p_value_one_sided": float(mw.pvalue),
        "per_type_reports_with_any": {k: {"distressed": f"{(d[k] > 0).sum()}/{len(d)}", "healthy": f"{(h[k] > 0).sum()}/{len(h)}"} for k in RED_FLAGS},
    }

    # Fig 4: share of reports containing each red-flag type
    names = list(RED_FLAGS)
    dv = [100 * (d[k] > 0).mean() for k in names]
    hv = [100 * (h[k] > 0).mean() for k in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(y - 0.19, dv, height=0.36, color=DISTRESSED, label=f"Distressed (n={len(d)})")
    ax.barh(y + 0.19, hv, height=0.36, color=HEALTHY, label=f"Healthy (n={len(h)})")
    for yy, a, b in zip(y, dv, hv):
        ax.text(a + 1, yy - 0.19, f"{a:.0f}%", va="center", fontsize=8.5, color=INK2)
        ax.text(b + 1, yy + 0.19, f"{b:.0f}%", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y, names); ax.invert_yaxis(); ax.set_xlim(0, 112)
    ax.set_xlabel("% of reports containing at least one such sentence"); ax.grid(axis="y", visible=False)
    ax.set_title("Keyword rules: a few flags separate the groups, most also fire on healthy reports")
    ax.legend(loc="lower right")
    save(fig, "04_red_flag_types_by_group.png")

    # Fig 5: red-flag sentences per 1,000 sentences, per report
    rf2 = rf.sort_values("flags_per_1k")
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(rf2.report, rf2.flags_per_1k, color=[DISTRESSED if g == "distressed" else HEALTHY for g in rf2.group], height=0.7)
    ax.set_title("Red-flag phrase density per report")
    ax.set_xlabel("red-flag sentences per 1,000 sentences (keyword rules)"); ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=HEALTHY), plt.Rectangle((0, 0), 1, 1, color=DISTRESSED)],
              labels=["Healthy", "Distressed"], loc="lower right")
    ax.text(0.98, 0.12, f"Mann-Whitney U, one-sided p = {mw.pvalue:.4f}", transform=ax.transAxes, ha="right", color=INK2, fontsize=9)
    save(fig, "05_red_flag_density_per_report.png")

    # ---------- Pilot labels ----------
    pr = pilot[pilot.is_risk == 1]
    out["pilot"] = {
        "n": int(len(pilot)), "risk": int(pilot.is_risk.sum()), "risk_pct": round(100 * pilot.is_risk.mean(), 1),
        "risk_pct_by_stratum": (100 * pilot.groupby("stratum").is_risk.mean()).round(1).to_dict(),
        "risk_pct_by_group": (100 * pilot.groupby("group").is_risk.mean()).round(1).to_dict(),
        "categories": pr.category.value_counts().to_dict(), "specificity": pr.specificity.value_counts().to_dict(),
        "bad_extraction": int((pilot.notes == "BAD_EXTRACTION").sum()),
        "imbalance_ratio_nonrisk_to_risk": round(float((pilot.is_risk == 0).sum() / max(pilot.is_risk.sum(), 1)), 1),
    }
    ct = pd.crosstab(pilot.group, pilot.is_risk)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    out["pilot"]["chi2_group_vs_risk"] = {"chi2": round(float(chi2), 2), "p": float(p)}

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    strata = ["random", "high_signal", "keyword"]
    vals = [100 * pilot[pilot.stratum == s].is_risk.mean() for s in strata]
    ax.bar(["Random", "High-signal\nsections", "Keyword\nmatch"], vals, color=HEALTHY, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.6, f"{v:.0f}%", ha="center", color=INK2)
    ax.set_ylim(0, max(vals) * 1.18); ax.set_title("Risk sentences by sampling stratum"); ax.set_ylabel("% labelled risk"); ax.grid(axis="x", visible=False)
    ax = axes[1]
    cats = ["Financial", "Legal", "Regulatory", "Operational", "Business"]
    cv = [int((pr.category == c).sum()) for c in cats]
    ax.bar(cats, cv, color=HEALTHY, width=0.6)
    for i, v in enumerate(cv):
        ax.text(i, v + 0.2, str(v), ha="center", color=INK2)
    ax.set_ylim(0, max(cv) * 1.15); ax.set_title(f"Pilot risk categories (n={len(pr)})"); ax.set_ylabel("sentences"); ax.grid(axis="x", visible=False)
    save(fig, "06_pilot_labels.png")

    # ---------- Year-over-year reuse (DHFL FY2018 vs FY2019) ----------
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    a = set(sent[(sent.company == "DHFL") & (sent.fy == "FY2018")].text.map(norm))
    b = sent[(sent.company == "DHFL") & (sent.fy == "FY2019")].text.map(norm)
    out["yoy_dhfl_pct_fy2019_sentences_repeated_from_fy2018"] = round(100 * b.isin(a).mean(), 1)

    # ---------- Vocabulary overlap train vs test ----------
    tok = lambda s: set(re.findall(r"[a-z]{3,}", " ".join(s).lower()))
    vtr, vte = tok(sent[sent.split == "train"].text), tok(sent[sent.split == "test"].text)
    out["vocab"] = {"train": len(vtr), "test": len(vte), "test_oov_pct": round(100 * len(vte - vtr) / len(vte), 1)}

    # ---------- Distinctive words in auditor sections: distressed vs healthy (log-odds) ----------
    aud = sent[sent.section.isin(["AUDITOR_REPORT", "CARO", "IFC_REPORT"])]
    cd = Counter(w for t in aud[aud.group == "distressed"].text for w in re.findall(r"[a-z]{4,}", t.lower()))
    ch = Counter(w for t in aud[aud.group == "healthy"].text for w in re.findall(r"[a-z]{4,}", t.lower()))
    nd, nh = sum(cd.values()), sum(ch.values())
    lo = {w: np.log((cd[w] + 1) / (nd + 1)) - np.log((ch[w] + 1) / (nh + 1)) for w in set(cd) | set(ch) if cd[w] + ch[w] >= 25}
    top = sorted(lo.items(), key=lambda x: -x[1])[:15]
    out["auditor_words_distressed_logodds_top15"] = [(w, round(v, 2)) for w, v in top]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ws, vs = zip(*top[::-1])
    ax.barh(ws, vs, color=DISTRESSED, height=0.65)
    ax.set_title("Words most typical of distressed auditor reports")
    ax.set_xlabel("log-odds ratio, distressed vs healthy (auditor report + CARO)"); ax.grid(axis="y", visible=False)
    save(fig, "07_auditor_words_distressed.png")

    (ROOT / "reports/eda_stats.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
