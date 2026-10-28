# Milestone Plan: Project vs NLP Handbook

*Maps the Corporate Risk & Red-Flag Analyzer proposal onto the course's NLP Project Milestone Handbook. Companion to [RESEARCH.md](RESEARCH.md).*

---

## 1. What the handbook asks

- Pick **one** track: **Research** (paper-driven: reproduce an anchor paper, find a gap, test one hypothesis, write an IEEE paper) or **Industry** (solve a business problem end to end, deploy it, show business impact).
- Six milestones, each reviewed during the semester.
- Final deliverable: an IEEE paper + viva (Research), or a technical report + GitHub repo + live demo + business PPT + viva (Industry).

### Inconsistencies in the handbook
The detailed milestone pages and the checklist on the last pages don't always match. **Cover both lists** so nothing is missed:

| Milestone | Detailed page says | Checklist says |
|---|---|---|
| Industry M2 | Statistical analysis | Train / validation / test split |
| Industry M4 | Model optimization, model inferencing | Evaluation report, error analysis |
| Industry M5 | Model deployment | Performance measurements |

---

## 2. Which track fits this project

| | Industry Track | Research Track |
|---|---|---|
| Fit with the current proposal | **Very high.** The proposal is already an end-to-end system with a dashboard. | **Low as written.** Needs an anchor paper to reproduce first; the agents and dashboard become mostly out of scope. |
| What must be added | Business case, users, privacy, deployment, monitoring, failure handling, robustness tests | Anchor paper + reproduction, one hypothesis with a prediction recorded beforehand, IEEE paper |
| What gets cut | Nothing essential | Most of the agent layer, the Streamlit app and probably semantic similarity (one hypothesis only) |
| Risk | Scope creep (many components) | Finding an anchor paper with public code + data that's close to this task |

**Recommendation: Industry Track.** The proposal (upload a PDF → risk dashboard) is already a product. The main change is reframing from "academic prototype" to "tool for a named user with measurable impact".

If you must take the Research Track, a workable version is:
- **Anchor paper:** a FinBERT financial-text classification paper with public data (e.g. Huang, Wang & Yang 2023 FinBERT, Financial PhraseBank / analyst-report sentences) or FiNER-ORD for NER. Both have public datasets and models, so M2 reproduction is feasible.
- **Gap:** these models are trained on US/English financial text; there is no evaluation on Indian annual reports, which have different structure, regulators and number formats.
- **One hypothesis (example):** "A FinBERT classifier fine-tuned on US risk sentences loses ≥10 macro-F1 points on Indian annual-report risk sentences, and adding ~500 Indian in-domain labels recovers most of that gap." Record the prediction before running it.

---

## 3. Industry Track: milestone-by-milestone plan

### M1 — Problem Definition & Business Case
| Deliverable | What to write for this project |
|---|---|
| **Problem brief** | Indian listed-company annual reports run 200–400 pages. Red flags (going-concern doubts, auditor qualifications, unpaid statutory dues, loan defaults, regulator penalties, large contingent liabilities) are scattered across the auditor's report, CARO annex, notes and governance report. Manual screening takes an analyst hours per report and is easy to get wrong. |
| **User identification** | Primary: **credit analysts at banks/NBFCs** (lending decisions, annual reviews). Secondary: equity research analysts, internal auditors/compliance teams, retail investors. Pick one primary user and design around them. |
| **Business objective** | Cut first-pass red-flag screening from hours to minutes per report, without missing material red flags, and with every finding traceable to page and sentence so the analyst can trust and verify it. |
| **Success metrics** | *Model:* risk-sentence recall ≥ 0.85 (missing a red flag is costlier than a false alarm), category macro-F1 ≥ 0.75, ₹ amount extraction accuracy ≥ 0.90, ≥ 95% of summary claims supported by cited evidence. *Business:* processing time < 5 min per report; recall of analyst-identified red flags on ~5 fully reviewed reports; time saved vs manual review in a small user test. *Set targets now, before building.* |
| **Feasibility analysis** | Data: public annual reports from BSE/NSE and company sites (free). Labelling: ~2,000 sentences ≈ 25–35 person-hours with LLM pre-labelling. Compute: one 8–16 GB GPU or Colab for base-size models. Cost: LLM API spend per report (estimate). Risks: PDF quality, category overlap, scope creep. Mitigations from RESEARCH.md §4. |

### M2 — Data Understanding & Preparation
| Deliverable | What to do |
|---|---|
| **Data card** | Follow the "Datasheets for Datasets" / Hugging Face data card format: source and URLs, companies/years/sectors, collection date, labelling guide, label definitions, annotators, kappa, known biases (e.g. large-cap heavy), licence notes. |
| **Exploratory analysis** | Pages and sentences per report; sentence length distribution; sentences per section; label and category distribution; % sentences with ₹ amounts; most frequent risk terms per category; how often the same sentence repeats across years. |
| **Statistical analysis** | Class imbalance ratio; Cohen's kappa on 200 double-labelled sentences; per-category counts; vocabulary overlap between train and test companies; chi-square or similar for category vs section. |
| **Data cleaning** | Header/footer/page-number removal, ₹ glyph repair, abbreviation-safe sentence splitting, table filtering, near-duplicate removal. Document every step and its effect (sentences before/after). |
| **Train / val / test split** | **Split by company** (no company in two splits); remove near-duplicates across splits; test set labelled by humans without LLM suggestions. State the split sizes in the data card. |
| **Privacy considerations** | Reports are public, but they contain personal data (directors, KMP, auditors' names, remuneration). Under India's **DPDP Act 2023**, state that the system analyses companies, not individuals; don't build person-level profiles. If a hosted LLM API is used, note what text leaves the machine and offer a local-model option. Note copyright: store PDFs for research use; don't redistribute them, distribute URLs instead. |

### M3 — Baseline & Model Development
| Deliverable | What to do |
|---|---|
| **Baseline model** | TF-IDF (word + char n-grams) + Logistic Regression for risk / non-risk and category. Also a keyword/rule baseline, since that's what an analyst would do with Ctrl+F. |
| **Candidate approaches** | FinBERT, DeBERTa-v3-base, SetFit, zero/few-shot LLM. Two-stage vs single 6-way classifier. |
| **Version-controlled implementation** | GitHub repo from week 1, layout as in RESEARCH.md §8. Fixed random seeds, `requirements.txt` or `pyproject.toml`, config files per experiment. |
| **Experiment log** | MLflow, Weights & Biases or a simple `experiments.csv`: run ID, date, git commit, model, hyperparameters, data version, val/test metrics, notes. Log failed runs too. |

### M4 — Evaluation & Optimization
| Deliverable | What to do |
|---|---|
| **Evaluation report** | Per-component metrics (RESEARCH.md §7): classifier P/R/F1/PR-AUC, NER entity F1, money value accuracy, similarity P@5/MRR, summary support rate. Model comparison table with mean ± std over seeds. |
| **Error analysis** | Categorise errors: Legal↔Regulatory confusion, negation ("no material litigation"), boilerplate flagged as risk, broken sentences from PDF, missed amounts in tables, wrong counterparty. Count each type on the test set. |
| **Model optimization** | Hyperparameter search; **decision-threshold tuning for high recall**; class weights; optionally ONNX export or quantisation for faster CPU inference. |
| **Model inferencing** | Measure latency per sentence and per full report, on CPU and GPU; memory use; LLM tokens and cost per report. |
| **Improvement iterations** | Document each change → metric delta, e.g. v1 baseline → v2 class weights → v3 FinBERT → v4 threshold tuning → v5 added section feature. |
| **Robustness testing** | Unseen sectors (e.g. train on manufacturing, test on banks); older vs newer report formats; ₹ glyph variants (`₹`, `Rs.`, `INR`, broken glyphs); negation and hedging; small typo/OCR-style noise; very long sentences; reports from distressed vs healthy firms. |

### M5 — Deployment Prototype
| Deliverable | What to do |
|---|---|
| **Working application / API** | FastAPI backend (`POST /analyze` → JSON findings) + Streamlit UI (upload PDF → dashboard; click a finding → highlighted source page). |
| **Model deployment** | Docker image; deploy to Hugging Face Spaces, Streamlit Community Cloud or a small VM. Models loaded once at startup. |
| **Performance measurements** | End-to-end time per report (split by stage), throughput, memory, cost per report; measured on 5+ real reports. |
| **Monitoring strategy** | Log per report: # findings per category, mean classifier confidence, % low-confidence predictions, verifier reject rate, stage latency, LLM errors. Alert if the distribution drifts far from the test-set baseline. Add a 👍/👎 feedback button per finding to collect new labels. |
| **Failure handling** | Scanned PDF (no text layer) → clear message; encrypted/corrupt PDF → reject; very large file → page limit; LLM timeout/failure → fall back to a template summary built from structured findings; verifier rejects a claim → drop it and flag it; no risks found → say so explicitly rather than showing an empty page. |

### M6 — Final Delivery
| Deliverable | Notes |
|---|---|
| **Technical report** | Problem, users, data card, method, results, error analysis, robustness, deployment, monitoring, limitations, future work. |
| **GitHub repository** | README with setup + one-command demo, results table, architecture diagram, data card, licence. |
| **Live demonstration** | Pick 2–3 reports with known red flags (e.g. a company that later defaulted) and show the system finding them, with page highlights. Keep a recorded backup video in case the live demo fails. |
| **Business presentation (PPT)** | User pain → solution → demo → accuracy + time saved → cost per report → limitations → roadmap. |
| **Viva** | Be ready for: why not just prompt an LLM; how you prevented leakage; why recall over precision; how the evidence is verified; what "agentic" adds (show the ablation). |

---

## 4. Changes to the original proposal for the Industry Track

1. Replace "academic prototype" framing with a **named primary user** (credit analyst) and **business metrics** (time saved, red-flag recall).
2. Add the **business-relevant label**: Generic vs Specific (see RESEARCH.md §4.1).
3. Add **privacy (DPDP Act), deployment, monitoring and failure handling**. None of these are in the proposal yet.
4. Add **robustness testing** and an **experiment log**; neither is in the proposal.
5. Keep the TF-IDF vs Transformer comparison. It covers "baseline + candidate approaches" in M3.
6. Keep the agents, but justify them with the verifier ablation, since examiners will judge business value, not architecture.

---

## 5. Open questions
- Confirm the track (Industry recommended).
- Milestone due dates and team size, to fit the 12-week plan in RESEARCH.md §9 onto the six checkpoints.
- Is a hosted LLM API allowed, or must everything run locally?
