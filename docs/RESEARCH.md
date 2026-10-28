# Research Brief: Agentic NLP Corporate Risk & Red-Flag Analyzer

*Prepared 2026-10-27. Covers the project proposal: what it is, what prior work exists, where the proposal is weak, and concrete recommendations for building it.*

---

## 1. The project in one paragraph

Input: one corporate annual report (PDF, likely Indian, since the proposal uses ₹ crore amounts). Output: a dashboard of risk findings. Each finding has a category (Financial / Legal / Regulatory / Operational / Business), entities, amounts, dates, the event, the source sentence and page, related statements found elsewhere in the report, and an LLM summary that stays tied to the evidence. The pipeline runs: PDF extraction → preprocessing → sentence segmentation → risk / non-risk classifier → category classifier → NER + information extraction → embedding similarity → three "agents" (Document, Risk, Report) → report. Training data is 30–50 reports with 1,500–2,500 hand-labelled sentences. The core experiment compares TF-IDF + Logistic Regression against a Transformer.

**Verdict:** the project is feasible and well scoped for an academic prototype. Most components have established tools. The main risks are not in modelling. They are (a) a fuzzy definition of "risk", (b) messy PDFs and Indian money formats, (c) data leakage between train and test, and (d) an "agentic" layer that is really a fixed pipeline. Sections 4–5 cover fixes.

> Note: the ■ symbol in the proposal is a broken ₹ glyph. The same thing will happen when you extract real Indian PDFs (₹ is often a custom font glyph or the legacy "`" character in Rupee Foradian fonts). Plan for it (see §5.3).

---

## 2. Prior work and how this project relates to it

| Work | What it does | Relevance |
|---|---|---|
| **GRAB** (NeurIPS 2025 workshop, arXiv 2509.21698) | 1.6M sentences from 10-K Item 1A (2001–2025), weakly labelled with a 193-term risk taxonomy → 21 fine types → 5 macro classes, using FinBERT attention + YAKE | Closest prior taxonomy. Borrow its 5 macro / 21 fine structure to justify your categories. Its labels are weak (no manual annotation), so it's for pre-training / silver data, not gold evaluation. |
| **10-K risk-sentence dataset** (NAACL 2018 demo, N18-5017) | Financial sentences labelled for risk by 8 finance specialists, from a 40,708-report 10-K corpus (1996–2013) | Human-labelled risk/non-risk sentences: useful extra training data and a citation for your binary task. |
| **FinBERT-XRC** (Tan & Kok, ICIS 2023, arXiv 2405.01881) | Predicts post-filing volatility risk from 10-K text with word/sentence-level explanations | Shows explainability at sentence level; different target (volatility, not category). |
| **RiskBERT** (Chinese A-share study) | Risk-factor classifier on 119k sentences from 2,400 Chinese listed firms | Shows the task transfers outside the US; good for related work. |
| **EDGAR-CORPUS** (Loukas et al. 2021) + **edgar-crawler** | All US 10-Ks 1993–2020 split into items, on Hugging Face (`eloukas/edgar-corpus`) | Unlimited unlabelled risk text (Item 1A) for weak labels, domain pre-training, or a cross-market test. |
| **FiNER-139** (ACL 2022) | 1.1M sentences, 139 XBRL numeric entity types | Key finding: BERT subword-splits numbers badly. Supports your choice to handle money/dates with rules + spaCy rather than pure Transformer NER. |
| **FiNER-ORD** (Shah et al. 2023) | Manually labelled financial NER (PER/ORG/LOC), CC BY-NC 4.0 | Ready-made gold set to evaluate your ORG/PER/LOC NER on financial text before you label your own. |
| **FinMTEB** (EMNLP 2025) | 64-dataset finance embedding benchmark; Fin-E5 ranks first | Two findings matter for you: general benchmarks predict finance performance poorly, and **bag-of-words beat dense embeddings on financial STS**. So include TF-IDF cosine as a similarity baseline; it may win. |
| **MiniCheck** (EMNLP 2024) | 770M-param fact checker that matches GPT-4 at judging if a claim is supported by a document | Use it to automatically score whether each summary sentence is supported by its cited evidence. |
| **LLMs as financial annotators** (arXiv 2403.18152); **MulFSA** (arXiv 2504.02429) | LLM pre-labelling vs humans, measured with Cohen's/Fleiss' kappa; MulFSA reports GPT-vs-final kappa 0.86 but warns of anchoring bias | Supports an LLM-pre-label + human-correct workflow, with a fully human-labelled test set. |
| **Indian risk disclosure studies**: Khandelwal et al. 2021 (206 BSE-500 firms, 1,854 reports, NVivo); Shivaani et al. 2019 (3,872 reports, 69-item risk disclosure index) | Manual content analysis of Indian annual reports. They find disclosures are mostly vague, qualitative and negative in tone | Strong motivation: no public NLP dataset of Indian annual-report risk sentences exists. **Your labelled dataset is itself a contribution.** The 69-item index can inform your taxonomy. |

**Gap you fill:** prior work is almost all US 10-K Item 1A, which is a clean, dedicated risk section. Indian annual reports have no single risk section. Risk is spread across MD&A, the auditor's report, the CARO annexure, notes to accounts and the governance report. No public, labelled, evidence-grounded system for Indian annual reports turned up in the search.

---

## 3. Where risk actually lives in an Indian annual report

This matters more than model choice. A "Document Analyst" that knows these sections will find the strongest red flags.

| Section | What to look for | Signal strength |
|---|---|---|
| **Independent Auditor's Report** | Qualified/adverse opinion, "Emphasis of Matter", "Material uncertainty related to going concern", Key Audit Matters (SA 701) | Very high (true red flags) |
| **CARO 2020 annexure** (part of the auditor's report) | Statutory dues not deposited / disputed (often a table with amount, period and forum), defaults in loan repayment, fraud reported, wilful defaulter status, cash losses | Very high; very structured |
| **Notes: Contingent liabilities & commitments** | Claims not acknowledged as debt, tax demands under appeal, guarantees, with ₹ amounts | High; best source of amounts |
| **Notes: Related party transactions** | Large loans/guarantees to promoter entities | Medium–high |
| **Corporate Governance Report** | Mandatory disclosure of penalties or strictures by SEBI/stock exchanges in the last 3 years; auditor resignations | High |
| **Secretarial Audit Report (Form MR-3)** | Observations of non-compliance with Companies Act / LODR, fines | High |
| **Board's / Directors' Report** | Material changes, fraud reported by auditors, defaults, change in auditors | Medium |
| **MD&A — "Risks and Concerns"** (required by SEBI LODR Reg. 34) | Forward-looking, generic risk language (competition, forex, raw materials) | Low–medium (mostly boilerplate) |
| **Risk Management Committee disclosure** (mandatory for top 1,000 listed entities since 2021) | Risk framework description | Low (process text, rarely risks) |
| **BRSR** (mandatory for top 1,000 since FY2022-23) | ESG / environmental / labour issues, complaints | Medium (good for Operational/Regulatory) |

---

## 4. Main weaknesses in the proposal (and fixes)

### 4.1 "Risk" vs "red flag" is undefined
The title promises red flags, but the classifier as described will mostly find **generic risk disclosure** ("The company faces competition…"). Those sentences show up in every report, so they are not red flags.

**Fix:** add one more label dimension, which is cheap to annotate and becomes your most interesting result:

- **Disclosure type:** `Generic` (hypothetical / boilerplate) vs `Specific` (a concrete event, amount, counterparty or date that has happened or is pending).
- Optionally **Severity** (Low/Med/High), but only if annotation time allows.

Then the dashboard can rank Specific + High findings first. That is the "red-flag" part.

### 4.2 The five categories overlap
Legal vs Regulatory (a SEBI penalty is both) and Business vs Operational overlap heavily. Without written rules, annotator agreement will be low.

**Fix:** write a 2-page annotation guide with decision rules, e.g.:
- *Regulatory* = action by a regulator or tax authority (SEBI, RBI, GST, Income Tax, pollution boards). *Legal* = disputes with private parties, courts and arbitration.
- *Financial* = liquidity, debt, defaults, losses, impairment, going concern, forex/interest exposure.
- *Operational* = plants, supply chain, safety, IT/cyber, people. *Business* = market, competition, demand, strategy, concentration.
- Allow **multi-label** or a primary + secondary label. Measure **Cohen's kappa** on 200 double-annotated sentences. Aim for ≥0.6; revise the guide if it's lower.

### 4.3 Class imbalance and sampling
Probably fewer than 10% of sentences in a full report are risk-bearing. Random sampling of 2,000 sentences would give too few positives per category (perhaps 20–40 per class).

**Fix:** stratified sampling for the training pool: about 40% from high-signal sections (§3), about 20% keyword-seeded from elsewhere, and about 40% random (to keep realistic negatives). **Keep the test set as full, naturally distributed sections from unseen companies**, so the reported precision reflects real use. Report **macro-F1 and PR-AUC**, not just accuracy, since accuracy will look high even for a useless model.

### 4.4 Data leakage
The same company's reports from different years are often 60–80% identical. The proposal's "where possible, unseen companies" should be mandatory:
- Split by company (`GroupShuffleSplit` / `GroupKFold` on company ID).
- Remove near-duplicate sentences across splits (e.g. MinHash or TF-IDF cosine > 0.9).

### 4.5 The "agentic" layer is a fixed pipeline
As drawn, the three agents run in fixed order with no decisions, so an examiner may say it's a pipeline relabelled as agents. Make it agentic in a way you can measure:

- **Give agents tools and choices.** For example, the Risk Analyst gets `search_report(query)` and `get_section(name)`. When it sees "litigation" in MD&A, it decides to look up the contingent-liabilities note to find the amount, and records which tool calls it made.
- **Add a Verifier step.** Each summary sentence is checked against its cited evidence with MiniCheck (or an NLI model). Unsupported claims go back to the Report Analyst for a rewrite or are dropped. This gives a measurable loop.
- **Ablation:** report faithfulness and completeness with and without the tool use and the verifier. That turns the agentic layer into a result rather than a diagram.
- Keep orchestration in plain Python (or LangGraph if you want a visible graph). Agents exchange **typed JSON (Pydantic models)**, never free text, so each stage can be evaluated separately.

### 4.6 Missing third baseline
Add **zero-/few-shot LLM classification** as Model 3. In 2026, reviewers will ask "why not just prompt an LLM?" Run it on the test set and compare F1, cost and latency. A fine-tuned small model matching or beating it is a strong finding. Losing to it is still an honest finding.

---

## 5. Component-by-component recommendations

### 5.1 PDF extraction
- **PyMuPDF (`fitz`)**: fast, gives page numbers and block coordinates. Use it as the main extractor. Store `(page, block_bbox, text)` for each sentence so evidence links back to the exact location.
- **Docling**: slower (seconds per page) but much better at tables and reading order according to informal comparisons. Use it only on note pages with tables (contingent liabilities, CARO disputed-dues table), or skip it for v1.
- Remove running headers/footers (lines repeated on most pages), page numbers, and the table of contents.
- **Keep tables out of the sentence classifier.** Classify narrative text only, and handle the CARO and contingent-liability tables with a separate rule-based extractor (a stretch goal).
- Scanned reports: rare for listed Indian companies after about 2015; detect "no text layer" and skip them rather than adding OCR.

### 5.2 Sentence segmentation
Financial text breaks naive splitters on `Rs.`, `Ltd.`, `Pvt.`, `No.`, `i.e.`, `viz.`, `Sr.`, `Co.`, `crore.` and numbered clauses like `(iii)`. Use spaCy with custom exceptions, or **PySBD** (rule-based, handles abbreviations well), and add an abbreviation list. Evaluate it on about 100 hand-checked sentences, since segmentation errors propagate to every later stage.

### 5.3 Money, dates and Indian formats
spaCy's MONEY entity does not reliably handle Indian formats. Build a **regex + spaCy EntityRuler** layer for:
- Symbols: `₹`, `Rs.`, `Rs`, `INR`, plus the mis-extracted ₹ glyph (`■`, `` ` ``, private-use Unicode characters). Check what PyMuPDF returns for ₹ in each sample report.
- Units: `lakh(s)`, `lac(s)`, `crore(s)`, `cr`, `mn`, `million`, `bn`.
- Indian digit grouping: `1,25,000`.
- Implied units from table and section headers: "(₹ in crore)". Carry the unit from the nearest header.
- **Normalise to a numeric INR value** (8.5 crore → 85,000,000) so amounts can be sorted and compared.

Dates: the financial-year forms `FY 2024-25`, `FY25`, `year ended March 31, 2025`, `AY 2019-20` (tax assessment years appear in disputes). Use **dateparser** plus custom rules.

ORG / PERSON / LOCATION: spaCy `en_core_web_trf` (or `en_core_web_lg` for speed). Check it on FiNER-ORD first, then on about 150 of your own sentences. Add an EntityRuler gazetteer of Indian regulators and courts: SEBI, RBI, NCLT, NCLAT, CCI, GST Council, CBDT, ITAT, High Court, Supreme Court, MCA, ROC, DGGI, Enforcement Directorate.

### 5.4 Risk event extraction
Don't train a model for this. Use a small **trigger lexicon mapped to event types**, with spaCy dependency patterns, or a constrained LLM extraction step that must quote its span:

| Event type | Example triggers |
|---|---|
| Litigation | suit, petition, arbitration, claim, proceedings |
| Regulatory action | penalty, show-cause notice, order, levied, demand |
| Default / liquidity | default, delay in repayment, NPA, restructuring, going concern |
| Tax dispute | demand, assessment, appeal before CIT(A)/ITAT |
| Fraud | fraud, misappropriation, forensic audit |
| Audit qualification | qualified opinion, emphasis of matter, except for |
| Impairment / loss | impairment, write-off, net loss, erosion of net worth |

Counterparty: the ORG entity nearest the trigger that is not the reporting company. Resolve the reporting company's own name from the cover page.

### 5.5 Classifiers (the main experiment)

| Model | Notes |
|---|---|
| **M1: TF-IDF + Logistic Regression** | Use word 1–2 grams plus character 3–5 grams. Character n-grams help with numbers and abbreviations. Also try a Linear SVM. Lemmatisation and stop-word removal belong here only. **Don't remove stop words for Transformers**, because negation ("no material litigation") flips meaning. |
| **M2a: FinBERT** (`yiyanghkust/finbert-pretrain` or `ProsusAI/finbert`) | Pre-trained on financial text. Fine-tune it on your labels. |
| **M2b: DeBERTa-v3-base** | General-purpose, but often the strongest small encoder for classification. A good contrast with domain pre-training. |
| **M2c: SetFit** (sentence-transformer + head) | Strongest when labels are scarce (8–64 per class). Good for the smaller categories, and gives a learning curve. |
| **M3: Zero/few-shot LLM** | Same label definitions as the annotation guide, structured output. |

Design: a **two-stage** model (risk/non-risk, then category) vs a **single 6-way** model (5 categories + None). Try both; it's a cheap extra result. Also plot a **learning curve** (F1 at 250 / 500 / 1,000 / all training sentences). It shows whether more labelling would help and is easy to produce.

### 5.6 Semantic similarity
- Embedders: `all-mpnet-base-v2` or `BAAI/bge-base-en-v1.5` (fast, strong), plus a finance-tuned one from the FinMTEB leaderboard if it fits your GPU. **Include a TF-IDF cosine baseline**, since FinMTEB found bag-of-words beats dense models on financial STS.
- Use FAISS or plain NumPy (a single report has only a few thousand sentences).
- **Show only cross-section matches.** Neighbouring sentences in the same paragraph are trivially similar and not useful. The useful case is MD&A ↔ contingent-liabilities note ↔ CARO table.
- Combine embedding similarity with **entity/amount overlap** (same counterparty or same ₹ amount means very likely related).
- Evaluation: about 300 labelled pairs (related / not related) → precision@k, MRR, and ROC-AUC on pair scores.

### 5.7 Evidence-grounded summaries
- The LLM receives only the structured findings plus their evidence sentences, never the whole report.
- Each output sentence must cite finding IDs. The prompt forbids numbers that don't appear in the evidence, and a **post-check verifies every ₹ amount and date in the summary exists in the cited evidence** (a simple string/number match, but very effective).
- MiniCheck scores support for each sentence. Report the % supported.
- LLM options: a hosted API, or a local model (e.g. a Qwen- or Llama-family 7–8B instruct model via Ollama) if data must stay offline. Pin the model version in the report for reproducibility.

---

## 6. Dataset plan (concrete)

| Item | Target |
|---|---|
| Reports | 40 Indian listed companies (BSE/NSE), mix of sectors, mix of healthy and distressed firms (include a few firms with known defaults, insolvency or SEBI orders so there are real red flags). 1 report per company for train/val; test companies never seen in training. |
| Sentences | ~2,000 gold: train ~1,400 / val ~300 / test ~300, **split by company** |
| Labels per sentence | `is_risk`, `category` (primary + optional secondary), `specificity` (Generic/Specific) |
| Span labels (subset of ~400 risk sentences) | ORG, PERSON, LOC, MONEY, DATE, EVENT_TRIGGER, for NER/IE evaluation |
| Agreement | 200 sentences double-labelled → Cohen's kappa |
| Labelling workflow | LLM pre-labels the train/val pool → human corrects (fast). **Test set labelled by humans without LLM suggestions**, to avoid the anchoring bias MulFSA reports. |
| Tools | Label Studio or Doccano (both free, export JSON/CoNLL) |
| Optional extras | NAACL-2018 10-K risk sentences, GRAB silver labels and EDGAR Item 1A for pre-training or a "US → India transfer" experiment |

Sources for reports: company investor-relations pages, BSE/NSE corporate filings (annual report section). Record each PDF's URL and download date in a `manifest.csv`.

---

## 7. Evaluation plan (expanded)

| Component | Metrics |
|---|---|
| Segmentation | Boundary precision/recall on ~100 hand-checked sentences |
| Risk detection | Precision, recall, F1, **PR-AUC**, confusion matrix |
| Categorisation | **Macro-F1** (headline), per-class F1, confusion matrix (expect Legal↔Regulatory confusion; discuss it) |
| Model comparison | M1 vs M2a/b/c vs M3, with mean ± std over 3–5 seeds; McNemar's test or bootstrap CI for significance |
| NER / IE | Entity-level P/R/F1 with `seqeval` (strict match); for money, also **value accuracy after normalisation** |
| Similarity | Precision@5, MRR, pair ROC-AUC; dense vs TF-IDF |
| Report / agents | (1) Evidence correctness: does the cited sentence/page contain the claim? (2) Number accuracy: % of ₹/dates in the summary found in the evidence. (3) MiniCheck support rate. (4) Completeness: recall of gold "key findings" on 5 fully annotated test reports. (5) Ablation: no-tools / no-verifier vs full agent |
| Practical | Total runtime per report, LLM tokens/cost per report |

---

## 8. Suggested architecture and repo layout

```
corporate-risk-analyzer/
├── data/
│   ├── raw_pdfs/            # + manifest.csv (company, year, url, sha256)
│   ├── extracted/           # per-report JSONL: {sent_id, page, bbox, section, text}
│   └── annotations/         # label studio exports, guide.md
├── src/
│   ├── extract/             # pymupdf reader, header/footer removal, section detector
│   ├── preprocess/          # segmentation (pysbd), normalisation, ₹ fix
│   ├── classify/            # tfidf_lr.py, transformer.py, setfit.py, llm_zero_shot.py
│   ├── extract_info/        # spaCy pipeline + EntityRuler + money/date normalisers + event triggers
│   ├── similarity/          # embed.py, search.py
│   ├── agents/              # schemas.py (Pydantic), document_analyst.py, risk_analyst.py,
│   │                        # report_analyst.py, verifier.py, tools.py, orchestrator.py
│   └── eval/                # metrics scripts per component
├── app/streamlit_app.py     # upload → dashboard; click a finding → highlighted PDF page
├── notebooks/               # EDA, error analysis
└── reports/                 # results tables, figures
```

Core data object passed between agents:

```python
class Finding(BaseModel):
    id: str
    sentence: str
    page: int
    section: str
    is_risk: bool
    risk_prob: float
    category: Literal["Financial","Legal","Regulatory","Operational","Business"]
    specificity: Literal["Generic","Specific"]
    entities: list[Entity]          # text, label, char span
    amounts: list[Money]            # raw text, value_inr, unit
    dates: list[DateRef]
    event: str | None
    counterparty: str | None
    related_ids: list[str]
    tool_trace: list[str]           # what the agent looked up, for transparency
```

Streamlit tip: render the source page with PyMuPDF and draw a highlight box over the evidence bbox. Showing the exact highlighted sentence makes the "traceable evidence" claim obvious in a demo.

---

## 9. Suggested timeline (12 weeks)

| Weeks | Work |
|---|---|
| 1–2 | Collect 40 reports, extraction + section detection, ₹/segmentation fixes, annotation guide v1 |
| 3–4 | Annotate (LLM pre-label + correct), kappa check, guide v2, test set by hand |
| 5–6 | M1 baseline, transformer models, SetFit, LLM zero-shot; learning curves |
| 7 | NER/IE layer + evaluation |
| 8 | Similarity + evaluation |
| 9–10 | Agents, tools, verifier, ablation |
| 11 | Streamlit dashboard, end-to-end runs on test reports |
| 12 | Error analysis, write-up |

---

## 10. Possible contributions to claim in the write-up
1. The first (to our knowledge) labelled sentence-level risk dataset for **Indian** annual reports, with a Generic/Specific dimension.
2. A comparison of classical, domain Transformer, few-shot and LLM classifiers on this task, with learning curves.
3. An Indian-format financial IE layer (₹/lakh/crore normalisation, FY dates, regulator gazetteer).
4. A measurable agentic layer: tool-using analysts plus an automatic evidence verifier, with an ablation on faithfulness.

## 11. Decisions still open
- **Indian or US reports?** The brief assumes Indian (₹ crore examples). With US 10-Ks, labelled data and Item 1A sections make things easier but the novelty is lower.
- **LLM:** hosted API vs local model (affects cost, privacy and reproducibility).
- **Compute:** a single 8–16 GB GPU (or Colab) is enough for base-size encoders and SetFit. Fin-E5 (7B) embeddings would need more.
- **Team size and deadline:** these decide whether the stretch goals (table extraction, severity labels, US→India transfer) are in scope.

---

## Sources
- GRAB benchmark — https://arxiv.org/abs/2509.21698
- FinBERT-XRC — https://arxiv.org/abs/2405.01881
- NAACL 2018 10-K risk sentence system — https://aclanthology.org/N18-5017
- EDGAR-CORPUS — https://arxiv.org/abs/2109.14394 · https://huggingface.co/datasets/eloukas/edgar-corpus · https://github.com/nlpaueb/edgar-crawler
- FiNER-139 — https://aclanthology.org/2022.acl-long.303 · https://huggingface.co/datasets/nlpaueb/finer-139
- FiNER-ORD — https://arxiv.org/abs/2302.11157
- FinMTEB — https://arxiv.org/abs/2502.10990
- MiniCheck — https://arxiv.org/abs/2404.10774
- LLMs as financial data annotators — https://arxiv.org/abs/2403.18152
- MulFSA (annotation, kappa, anchoring bias) — https://arxiv.org/abs/2504.02429
- Financial risk disclosure in Indian listed companies — https://www.emerald.com/insight/content/doi/10.1108/QRFM-11-2019-0140/full/html
- Risk disclosure index, Indian context — https://www.emerald.com/insight/content/doi/10.1108/MAJ-07-2016-1403/full/html
- IndicFinNLP — https://aclanthology.org/2024.lrec-main.789
- SEBI LODR overview — https://irisbusiness.com/?p=30170
- Risk Management Committee (top 1,000) — https://www2.deloitte.com/content/dam/Deloitte/in/Documents/risk/in-ra-complying-with-revised-risk-management-committee-requirements-for-listed-entities-noexp.pdf
- BRSR — https://assets.kpmg.com/content/dam/kpmg/in/pdf/2021/06/firstnotes-sebi-business-responsibility-sustainability-reporting-listed-companies.pdf
- CARO 2020 amendments — https://assets.kpmg.com/content/dam/kpmg/in/pdf/2021/07/vor-sebi-companies-act-updates-covid-19.pdf
- Docling vs PyMuPDF4LLM (informal) — https://www.file2markdown.ai/blog/pymupdf4llm-vs-docling
- SetFit replication — https://arxiv.org/abs/2306.08058

*Caveats: the PDF-tool comparisons are informal blog tests. Details of the CARO clauses, MR-3 and governance-report penalty disclosure come from general domain knowledge and should be checked against the current MCA/SEBI texts before you cite them.*
